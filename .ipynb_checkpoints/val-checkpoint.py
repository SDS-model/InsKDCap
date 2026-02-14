# -*- coding: utf-8 -*-
"""
验证 + 知识使用度可视化（cross-attention 热力图）
- 默认使用 Greedy 以获取完整 T×S 注意力；如需 Beam，可切换 VIZ_USE_GREEDY=False
- 兼容 Knowledge-only（以 data['KNOWLEDGE'] 作为 K/V）
- 同时支持 NPY 与 JSON（属性: texts+groups+mappings；背景: images 下三段768）
- 横轴使用英文槽位标签，避免中文字体告警
"""

import os
import json
import tqdm
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xmodaler.utils.comm as comm
from xmodaler.checkpoint import XmodalerCheckpointer
from xmodaler.config import get_cfg, kfg
from xmodaler.engine import default_setup, build_engine
from xmodaler.modeling import add_config

# =========================
# 0) 可改的超参与路径
# =========================
CFG_PATH   = "configs/image_caption/transformer/transformer.yaml"
CKPT_PATH  = "output/model_final.pth"

# 你的知识文件（验证集）
FG_NPY     = "./mscoco_dataset/foreground_val.npy"     # {img_id:int -> np.array(5 or 6, 768)}
BG_NPY     = "./mscoco_dataset/background_val.npy"     # {img_id:int -> np.array(3, 768)}
FG_JSON    = "./mscoco_dataset/foreground_val.json"    # 属性 JSON（texts+groups+mappings）
BG_JSON    = "./mscoco_dataset/background_val.json"    # 背景 JSON（images/imgX.jpg 下 env/problem/solution）

OUTDIR         = "./attn_viz"
VIS_ONE_ID     = None         # 指定只画某张（int）；None 表示每 batch 画一张
SAVE_PER_BATCH = True

# 可视化使用 greedy（建议），以确保每一步 attention 都被捕获；评测你仍可用 beam
VIZ_USE_GREEDY = True

# 英文槽位标签（固定9槽）
SLOT_NAMES_EN = [
    "Strings", "Color", "Shape", "Material", "DefectLoc", "DefectType",
    "EnvFeat", "Impact", "Measure"
]

# =========================
# 1) 配置与模型
# =========================
def setup_cfg(cfg_path: str):
    cfg = get_cfg()
    tmp = cfg.load_from_file_tmp(cfg_path)
    add_config(cfg, tmp)
    cfg.merge_from_file(cfg_path)
    # 生成模式
    cfg.merge_from_list(['INFERENCE.GENERATION_MODE', True])
    cfg.freeze()
    default_setup(cfg, None)
    return cfg

def build_model_from_cfg(cfg_path: str, ckpt_path: str):
    cfg = setup_cfg(cfg_path)
    trainer = build_engine(cfg)
    model = trainer.model
    model.eval()
    XmodalerCheckpointer(model, save_dir=os.path.dirname(ckpt_path)).resume_or_load(
        ckpt_path, resume=False
    )
    return cfg, model, trainer.test_data_loader, trainer.test_evaluator

# =========================
# 2) 解析你的 JSON/NPY
# =========================
def _safe_vec(x, dim=768):
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    out = np.zeros((dim,), dtype=np.float32)
    out[:min(dim, v.size)] = v[:min(dim, v.size)]
    return out

def load_fg_from_json(path):
    """
    前景（属性）JSON:
    {
      "texts": [N,768],
      "groups": {"number of strings":[...], "color":[...], "shape":[...], ...},
      "mappings": [{"visual_id":"360.npz","text_ids":[... (通常5~6个)]}, ...]
    }
    返回:
      texts: np.ndarray [N,768]
      map_by_vid: dict[str_visual_id] -> list[text_id]
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    texts = np.asarray(obj["texts"], dtype=np.float32)
    mp = {}
    for m in obj.get("mappings", []):
        mp[m["visual_id"]] = m["text_ids"]
    return texts, mp

def load_bg_from_json(path):
    """
    背景（场景）JSON:
    {
      "images": {
        "img360.jpg": {"env_feature":[768], "problem":[768], "solution":[768]}, ...
      }
    }
    返回:
      dict[str_img] -> (env, prob, solu) 三个 768 向量
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    d = {}
    for name, triple in obj.get("images", {}).items():
        env = _safe_vec(triple.get("env_feature", []))
        prob = _safe_vec(triple.get("problem", []))
        solu = _safe_vec(triple.get("solution", []))
        d[name] = (env, prob, solu)
    return d

def get_knowledge_for_id(img_id: int,
                         fg_npy_dict, bg_npy_dict,
                         fg_json_pack, bg_json_dict):
    """
    返回：
      K: np.array(9,768)  按 SLOT_NAMES_EN 顺序
      labels: 9 个英文标签（若缺 shape 则置 Shape*）
    数据优先级：NPY > JSON
    """
    K = []
    labels = SLOT_NAMES_EN.copy()

    # ---------- 前景（最多6槽：Strings/Color/Shape/Material/DefectLoc/DefectType） ----------
    fg_used = False
    if fg_npy_dict is not None and (img_id in fg_npy_dict):
        arr = np.asarray(fg_npy_dict[img_id], dtype=np.float32)  # [5 or 6,768]
        # 标准到6槽（不足补零，多的截断）
        for i in range(6):
            v = arr[i] if i < arr.shape[0] else np.zeros((768,), dtype=np.float32)
            K.append(v)
        fg_used = True
    else:
        texts, map_by_vid = fg_json_pack if fg_json_pack is not None else (None, None)
        if texts is not None:
            as_npz = f"{img_id}.npz"
            as_jpg = f"img{img_id}.jpg"
            ids = map_by_vid.get(as_npz) or map_by_vid.get(as_jpg) or []
            # 把 text_ids 取到向量，最多6个
            vecs = [ _safe_vec(texts[t]) for t in ids[:6] ]
            while len(vecs) < 6: vecs.append(np.zeros((768,), dtype=np.float32))
            K.extend(vecs[:6])
            fg_used = True

    if not fg_used:
        K.extend([np.zeros((768,), dtype=np.float32) for _ in range(6)])
        # 若缺 shape，标注成 Shape*（只是可视化标签，模型不受影响）
        labels[2] = "Shape*"

    # 如果是老版本只有5槽，第三槽缺失 Shape，占位零向量已在上面处理

    # ---------- 背景（3槽：EnvFeat/Impact/Measure） ----------
    bg_used = False
    if bg_npy_dict is not None and (img_id in bg_npy_dict):
        arr = np.asarray(bg_npy_dict[img_id], dtype=np.float32)  # [3,768]
        for i in range(3):
            v = arr[i] if i < arr.shape[0] else np.zeros((768,), dtype=np.float32)
            K.append(v)
        bg_used = True
    else:
        if bg_json_dict is not None:
            key = f"img{img_id}.jpg"
            if key in bg_json_dict:
                env, prob, solu = bg_json_dict[key]
                K.extend([env, prob, solu])
                bg_used = True

    if not bg_used:
        K.extend([np.zeros((768,), dtype=np.float32) for _ in range(3)])

    K = np.stack(K, axis=0)  # [9,768]
    return K, labels

# =========================
# 3) Cross-Attn Hook
# =========================
CROSS_ATTN_CACHE = []

def register_last_cross_attn_hook(model: nn.Module, layer_idx: int = -1):
    modules = dict(model.named_modules())
    dec = modules['decoder']
    glayers = dec.g_layers
    last = layer_idx if layer_idx >= 0 else (len(glayers) - 1)
    target_name = f"decoder.g_layers.{last}.x_att.self"
    mod = modules[target_name]

    def fwd_hook(module, inputs, outputs):
        attn = None
        for attr in ('attention_probs', 'attn_probs', 'attn', 'attn_weights'):
            if hasattr(module, attr):
                v = getattr(module, attr)
                if torch.is_tensor(v) and v.dim() == 4:  # [B,H,T,S]
                    attn = v
                    break
        if attn is None and isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
            if torch.is_tensor(outputs[1]) and outputs[1].dim() == 4:
                attn = outputs[1]
        if attn is not None:
            CROSS_ATTN_CACHE.append(attn.detach())

    return mod.register_forward_hook(fwd_hook)

# =========================
# 4) 画图
# =========================
def _ascii_list(labels, fallback_prefix):
    out = []
    for i, s in enumerate(labels):
        try:
            t = str(s).encode('ascii', 'ignore').decode('ascii').strip()
            out.append(t if t else f"{fallback_prefix}{i+1}")
        except Exception:
            out.append(f"{fallback_prefix}{i+1}")
    return out

def plot_heatmap(mat_ts: np.ndarray, ylabels, xlabels, title, save_path):
    T, S = mat_ts.shape
    xlabels = _ascii_list(xlabels, "x")
    ylabels = _ascii_list(ylabels[:T], "t")

    plt.figure(figsize=(max(6, 0.5 * S), max(4, 0.35 * T)))
    plt.imshow(mat_ts, aspect='auto')
    plt.colorbar()
    plt.xticks(range(S), xlabels, rotation=45, ha='right', fontsize=9)
    plt.yticks(range(T), ylabels, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=220)
    plt.close()

import types
import torch
from xmodaler.config import kfg

def patch_visual_embed_forward_only(model):
    # 备份原 forward
    orig_forward = model.visual_embed.forward

    def wrapped_forward(self, batched_inputs):
        if 'KNOWLEDGE' in batched_inputs:
            att = batched_inputs['KNOWLEDGE']             # [B,S,768]
            att_masks = torch.ones(att.size(0), att.size(1), device=att.device)
            return { kfg.ATT_FEATS: att, kfg.ATT_MASKS: att_masks }
        # 否则走原有视觉分支
        return orig_forward(batched_inputs)

    # 把 wrapped 绑定成 bound method（而不是替换整个模块）
    model.visual_embed.forward = types.MethodType(wrapped_forward, model.visual_embed)


# =========================
# 5) 主流程（验证+可视化）
# =========================
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cfg, model, test_loader, evaluator = build_model_from_cfg(CFG_PATH, CKPT_PATH)
    patch_visual_embed_forward_only(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 先尝试加载 NPY；失败再落到 JSON
    fg_npy = np.load(FG_NPY, allow_pickle=True).item() if os.path.exists(FG_NPY) else None
    bg_npy = np.load(BG_NPY, allow_pickle=True).item() if os.path.exists(BG_NPY) else None
    fg_json_pack = None
    if os.path.exists(FG_JSON):
        try:
            fg_json_pack = load_fg_from_json(FG_JSON)
        except Exception:
            fg_json_pack = None
    bg_json_dict = None
    if os.path.exists(BG_JSON):
        try:
            bg_json_dict = load_bg_from_json(BG_JSON)
        except Exception:
            bg_json_dict = None

    model.eval()
    results = []
    epoch = -1

    with torch.no_grad():
        for data in tqdm.tqdm(test_loader):
            data = comm.unwrap_model(model).preprocess_batch(data)

            # 取样本 id 列表（int）
            ids_field = data[kfg.IDS]
            if isinstance(ids_field, torch.Tensor):
                ids_list = [int(i) for i in ids_field.cpu().tolist()]
            else:
                ids_list = [int(i) for i in ids_field]

            # 组装 KNOWLEDGE / MASKS / 视觉接口（K-only）
            B = len(ids_list)
            Ks = []
            xlabels_per_sample = []
            for img_id in ids_list:
                K, labels = get_knowledge_for_id(
                    img_id, fg_npy, bg_npy, fg_json_pack, bg_json_dict
                )  # [9,768]
                Ks.append(K)
                xlabels_per_sample.append(labels)
            K_all = np.stack(Ks, axis=0)                         # [B,9,768]
            data['KNOWLEDGE'] = torch.tensor(K_all, device=device)
            B, S = data['KNOWLEDGE'].shape[:2]
            data['KNOWLEDGE_MASKS'] = torch.zeros(B, 1, 1, S, device=device)  # 0=可见（与你现有实现一致）

            # 给 beam/decoder 补视觉键：用知识代替视觉特征（K-only）
            data[kfg.ATT_FEATS] = data['KNOWLEDGE']             # [B,9,768]
            data[kfg.ATT_MASKS] = torch.ones(B, S, device=device)

            # 注册 cross-attn 钩子
            h = register_last_cross_attn_hook(model, layer_idx=-1)

            # 推理：可视化建议 greedy，保证每步 attention 都可抓到
            if VIZ_USE_GREEDY:
                res = model(data, use_beam_search=False, output_sents=True)
            else:
                res = model(data, use_beam_search=True, output_sents=True)

            # 卸钩
            h.remove()

            # 收集输出
            outputs = res[kfg.OUTPUT]  # list[str]
            for img_id, sent in zip(ids_list, outputs):
                results.append({"image_id": int(img_id), "caption": sent})

            # 绘图
            if len(CROSS_ATTN_CACHE) > 0 and (SAVE_PER_BATCH or VIS_ONE_ID is not None):
                attn_bhts = CROSS_ATTN_CACHE[-1]    # [B,H,T,S]
                attn_bts  = attn_bhts.mean(dim=1)   # [B,T,S]

                # 选要画的样本
                if VIS_ONE_ID is not None:
                    bidx = ids_list.index(VIS_ONE_ID) if VIS_ONE_ID in ids_list else None
                else:
                    bidx = 0

                if bidx is not None:
                    attn_ts = attn_bts[bidx]                    # [T,S]
                    attn_ts = attn_ts / (attn_ts.sum(dim=-1, keepdim=True) + 1e-8)

                    gen_text = outputs[bidx]
                    gen_tokens = str(gen_text).split()

                    xlabels = xlabels_per_sample[bidx] if xlabels_per_sample else SLOT_NAMES_EN
                    save_path = os.path.join(OUTDIR, f"attn_words_vs_nodes_{ids_list[bidx]}.png")
                    plot_heatmap(attn_ts.cpu().numpy(), gen_tokens, xlabels,
                                 title=f"Cross-Attention (image {ids_list[bidx]})",
                                 save_path=save_path)

            CROSS_ATTN_CACHE.clear()

    # 可选：跑 evaluator
    if evaluator is not None:
        try:
            eval_res = evaluator.eval(results, epoch)
            if comm.is_main_process():
                print(eval_res)
        except Exception as e:
            if comm.is_main_process():
                print("Evaluator 调用失败，仅统计生成条数：", len(results))
    else:
        if comm.is_main_process():
            print("生成条数：", len(results))


if __name__ == "__main__":
    main()
