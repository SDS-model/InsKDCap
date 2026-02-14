# -*- coding: utf-8 -*-
"""
Scene Causal-KG Network (Only 1x512 visual feature)
- 因果/相关双通道 + 门控
- 环境文本原型零样本先验
- 可选 do(E) 干预
依赖: torch, torch_geometric, transformers, numpy
"""

import os
import json
import math
import numpy as np
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from transformers import BertModel, BertTokenizer


# ===================== 配置 =====================
config = {
    "data_path": "./background_new.json",   # 你的知识与图像索引JSON
    "bert_model": "./bert-base-uncased",      # 中文BERT；英文可改为 bert-base-uncased
    "vit_dim": 512,                         # 仅用 1x512 全局视觉特征
    "bert_dim": 768,
    "hidden_dim": 256,                      # 节点隐层维度
    "num_epochs": 100,
    "lr": 1e-3,
    "knn_k": 5,                             # 同层KNN相关边
    "tau_init": 10.0,                       # 环境原型温度
    "gate_hidden": 128,                     # 门控中间维度
    "proto_ce_weight": 0.3,                 # 原型先验CE权重
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ===================== 工具：文本编码（BERT） =====================
class FrozenBERT(object):
    def __init__(self, model_name: str):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """返回 [N, bert_dim]，句向量=token平均。"""
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=24
        )
        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)  # [N, 768]
        return emb.detach()


# ===================== 视觉特征读取（仅1x512） =====================
def load_global_512(feat_path: str) -> np.ndarray:
    """
    兼容两种键:
    - arr['global'] -> [512] 或 [1,512]
    - arr['g_feature'] -> [512] 或 [1,512]
    """
    try:
        arr = np.load(feat_path, allow_pickle=True)
        if 'global' in arr:
            g = arr['global']
        elif 'g_feature' in arr:
            g = arr['g_feature']
        else:
            raise KeyError("missing global/g_feature in npz")
        g = g.astype(np.float32)
        g = g.squeeze()  # [512]
        if g.ndim != 1 or g.shape[0] != 512:
            raise ValueError(f"expect [512], got {g.shape}")
        return g
    except Exception as e:
        print(f"[WARN] 加载特征失败 {feat_path}: {e}，使用随机特征兜底")
        return np.random.randn(512).astype(np.float32)
# ===================== 从知识图构建合法掩码（A→I 与 I→M） =====================
def build_masks_from_graph(graph):
    n_a = graph['feature'].x.size(0)
    n_i = graph['problem'].x.size(0)
    n_m = graph['solution'].x.size(0)

    A2I = torch.zeros(n_a, n_i, dtype=torch.bool)
    I2M = torch.zeros(n_i, n_m, dtype=torch.bool)

    # A -> I
    ei = graph['feature', 'to', 'problem'].edge_index  # [2, E_ai]
    A2I[ei[0], ei[1]] = True

    # I -> M
    ei = graph['problem', 'to', 'solution'].edge_index  # [2, E_im]
    I2M[ei[0], ei[1]] = True

    return A2I, I2M
# ===================== 计算 C-Cons 指标并可视化 =====================
import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def ccons_doA_for_image(model, graph, idx_img: int, delta_I=0.0, delta_M=0.0, delta_lab_I=0.0, delta_lab_M=0.0):
    device = next(model.parameters()).device
    A2I, I2M = build_masks_from_graph(graph)
    A2I = A2I.to(device); I2M = I2M.to(device)

    # === 基线：无干预 ===
    out_base = model.forward_with_doA(graph.x_dict, graph.edge_index_dict, force_a_idx=None, do_env=False)
    pI_base = out_base["I"].softmax(dim=1)[idx_img]  # [n_I]
    pM_base = out_base["M"].softmax(dim=1)[idx_img]  # [n_M]

    n_a = graph['feature'].x.size(0)
    deltas_I, deltas_M = [], []
    label_hits = []
    struct_hits = []

    # 存储每个 a 的 I/M 预测，便于可视化
    yI_list, yM_list = [], []

    for a in range(n_a):
        force_a = torch.full((graph['visual'].x.size(0),), a, dtype=torch.long, device=device)
        out = model.forward_with_doA(graph.x_dict, graph.edge_index_dict, force_a_idx=force_a, do_env=False)

        pI = out["I"].softmax(dim=1)[idx_img]  # [n_I]
        yI = int(torch.argmax(pI))
        pM = out["M"].softmax(dim=1)[idx_img]  # [n_M]
        yM = int(torch.argmax(pM))

        # —— 结构方向增量（对合法集合求和）——
        mask_I = A2I[a]                  # 合法的 I
        alpha_I      = pI[mask_I].sum().item()
        alpha_I_base = pI_base[mask_I].sum().item()
        dI = alpha_I - alpha_I_base

        mask_M = I2M[yI]                 # 以干预后的 yI 决定合法 M
        alpha_M      = pM[mask_M].sum().item()
        alpha_M_base = pM_base[mask_M].sum().item()
        dM = alpha_M - alpha_M_base

        deltas_I.append(dI)
        deltas_M.append(dM)
        struct_hits.append( (dI>=delta_I) and (dM>=delta_M) )

        # —— 标签级一致性 ——（命中合法集合 + 概率提升可选）
        hit_I = mask_I[yI].item()
        hit_M = mask_M[yM].item()
        gain_I = (pI[mask_I].sum()-pI_base[mask_I].sum()).item()
        gain_M = (pM[mask_M].sum()-pM_base[mask_M].sum()).item()
        label_hits.append( hit_I and hit_M and (gain_I>=delta_lab_I) and (gain_M>=delta_lab_M) )

        yI_list.append(yI); yM_list.append(yM)

    # === 指标 ===
    C_struct = float(np.mean(struct_hits))
    C_label  = float(np.mean(label_hits))

    # === 可视化 ===
    xs = np.arange(n_a)

    plt.figure(figsize=(8,4))
    plt.plot(xs, deltas_I, marker='o', label=r'$\Delta_I(a)$')
    plt.plot(xs, deltas_M, marker='s', label=r'$\Delta_M(a)$')
    plt.axhline(0.0, color='gray', linewidth=1)
    plt.xlabel('forced A class index a')
    plt.ylabel('probability mass gain on legal set')
    plt.title('C-Cons (structural) per intervention')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ccons_struct_deltas.png', dpi=180)

    # 命中可视化：1/0 条形
    plt.figure(figsize=(8,3))
    bar_vals = np.array(label_hits, dtype=float)
    plt.bar(xs, bar_vals)
    plt.ylim(0,1.1)
    plt.xlabel('forced A class index a')
    plt.ylabel('label-consistent (0/1)')
    plt.title(f'Label-level consistency per a | mean={C_label:.2f}')
    plt.tight_layout()
    plt.savefig('ccons_label_hits.png', dpi=180)

    return {
        "C_struct": C_struct,
        "C_label": C_label,
        "deltas_I": deltas_I,
        "deltas_M": deltas_M,
        "label_hits": label_hits,
        "yI": yI_list,
        "yM": yM_list
    }

# ===================== 数据集 -> HeteroData =====================
class PowerlineDataset:
    """
    读取 JSON，构建异构图：
    节点类型: visual, subtype(E), feature(A), problem(I), solution(M)
    边: 因果边 E->A->I->M + 反向 E->visual； 同层相关边(KNN)
    """
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        with open(cfg["data_path"], "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.bert = FrozenBERT(cfg["bert_model"])

    def _encode_text_nodes(self, node_dict: Dict[str, Dict]) -> Tuple[torch.Tensor, Dict[str, int], List[str]]:
        """
        node_dict形如: {"id1": {"name": "xxx"}, "id2": {"name": "yyy"}, ...}
        返回: [N, bert_dim], id->idx映射, 名称表
        """
        items = list(node_dict.items())
        names = [v["name"] for _, v in items]
        emb = self.bert.encode(names)                       # [N, bert_dim]
        mapping = {k: i for i, (k, _) in enumerate(items)}  # id -> idx
        return emb, mapping, names

    @staticmethod
    def _build_knn_edges(x: torch.Tensor, k: int) -> torch.Tensor:
        """
        x: [N, D], 归一化后做KNN（相似度top-k）
        返回 edge_index [2, E]
        """
        with torch.no_grad():
            if x.size(0) <= 1 or k <= 0:
                return torch.empty(2, 0, dtype=torch.long)
            x_n = F.normalize(x, dim=-1)
            sim = x_n @ x_n.t()                     # [N,N]
            idx = torch.arange(x.size(0))
            sim[idx, idx] = -1e9                    # 去自环
            nbr = torch.topk(sim, k=min(k, x.size(0)-1), dim=1).indices  # [N,k]
            src = torch.repeat_interleave(idx, nbr.size(1))
            dst = nbr.reshape(-1)
            ei = torch.stack([src, dst], dim=0)     # [2, N*k]
        return ei

    def build_hetero_graph(self) -> Tuple[HeteroData, Dict]:
        data = HeteroData()

        # ---- 1) 文本节点：E/A/I/M ----
        # E: 环境子类型
        e_x, e_map, e_names = self._encode_text_nodes(self.data["knowledge_graph"]["environment_subtypes"])
        # A: 环境特征
        a_x, a_map, a_names = self._encode_text_nodes(self.data["knowledge_graph"]["environment_features"])
        # I: 问题/影响
        i_x, i_map, i_names = self._encode_text_nodes(self.data["knowledge_graph"]["problems"])
        # M: 措施
        m_x, m_map, m_names = self._encode_text_nodes(self.data["knowledge_graph"]["solutions"])

        data['subtype'].x = e_x
        data['feature'].x = a_x
        data['problem'].x = i_x
        data['solution'].x = m_x

        # ---- 2) 视觉节点：仅 1x512 ----
        vis_feats = []
        y_e, y_a, y_i, y_m = [], [], [], []
        image_items = list(self.data["images"].items())
        for img_id, info in image_items:
            g = load_global_512(info["g_feature_path"])
            vis_feats.append(torch.tensor(g).float())
            # 标签索引（不存在则为 -1）
            y_e.append(e_map.get(info.get("env_subtype", ""), -1))
            y_a.append(a_map.get(info.get("env_feature", ""), -1))
            y_i.append(i_map.get(info.get("problem", ""), -1))
            y_m.append(m_map.get(info.get("solution", ""), -1))

        data['visual'].x = torch.stack(vis_feats, dim=0)                 # [Nvis, 512]
        data['visual'].y_subtype = torch.tensor(y_e, dtype=torch.long)
        data['visual'].y_feature = torch.tensor(y_a, dtype=torch.long)
        data['visual'].y_problem = torch.tensor(y_i, dtype=torch.long)
        data['visual'].y_solution = torch.tensor(y_m, dtype=torch.long)

        # ---- 3) 因果边 ----
        # visual -> E
        ve_edges = []
        for idx, (_, info) in enumerate(image_items):
            eid = info.get("env_subtype", "")
            if eid in e_map:
                ve_edges.append([idx, e_map[eid]])
        data['visual', 'to', 'subtype'].edge_index = torch.tensor(ve_edges, dtype=torch.long).t().contiguous() \
            if len(ve_edges) else torch.empty(2, 0, dtype=torch.long)

        # E -> A
        ea_edges = []
        for eid, einfo in self.data["knowledge_graph"]["environment_subtypes"].items():
            if eid not in e_map: continue
            eidx = e_map[eid]
            for fid in einfo.get("features", []):
                if fid in a_map:
                    ea_edges.append([eidx, a_map[fid]])
        data['subtype', 'to', 'feature'].edge_index = torch.tensor(ea_edges, dtype=torch.long).t().contiguous() \
            if len(ea_edges) else torch.empty(2, 0, dtype=torch.long)

        # A -> I
        ai_edges = []
        for fid, finfo in self.data["knowledge_graph"]["environment_features"].items():
            if fid not in a_map: continue
            fidx = a_map[fid]
            for pid in finfo.get("problems", []):
                if pid in i_map:
                    ai_edges.append([fidx, i_map[pid]])
        data['feature', 'to', 'problem'].edge_index = torch.tensor(ai_edges, dtype=torch.long).t().contiguous() \
            if len(ai_edges) else torch.empty(2, 0, dtype=torch.long)

        # I -> M
        im_edges = []
        for pid, pinfo in self.data["knowledge_graph"]["problems"].items():
            if pid not in i_map: continue
            pidx = i_map[pid]
            for sid in pinfo.get("solutions", []):
                if sid in m_map:
                    im_edges.append([pidx, m_map[sid]])
        data['problem', 'to', 'solution'].edge_index = torch.tensor(im_edges, dtype=torch.long).t().contiguous() \
            if len(im_edges) else torch.empty(2, 0, dtype=torch.long)

        # 反向边：E -> visual（可选，帮助特征回流）
        ev_edges = []
        for idx, (_, info) in enumerate(image_items):
            eid = info.get("env_subtype", "")
            if eid in e_map:
                ev_edges.append([e_map[eid], idx])
        data['subtype', 'reverse_to', 'visual'].edge_index = torch.tensor(ev_edges, dtype=torch.long).t().contiguous() \
            if len(ev_edges) else torch.empty(2, 0, dtype=torch.long)

        # ---- 4) 相关边（KNN，同层）----
        # 使用 BERT 向量构造，再进入网络前会被线性投到 hidden_dim
        data['subtype',  'corr', 'subtype'].edge_index  = self._build_knn_edges(data['subtype'].x,  config["knn_k"])
        data['feature',  'corr', 'feature'].edge_index  = self._build_knn_edges(data['feature'].x,  config["knn_k"])
        data['problem',  'corr', 'problem'].edge_index  = self._build_knn_edges(data['problem'].x,  config["knn_k"])
        data['solution', 'corr', 'solution'].edge_index = self._build_knn_edges(data['solution'].x, config["knn_k"])

        meta = {
            "e_map": e_map, "a_map": a_map, "i_map": i_map, "m_map": m_map,
            "e_names": e_names, "a_names": a_names, "i_names": i_names, "m_names": m_names
        }
        return data, meta


# ===================== 环境文本原型零样本先验 =====================
class EnvTextPrototypes(nn.Module):
    """
    将 E 类的文本原型（BERT向量）投到 hidden_dim，并与视觉 hidden 做余弦匹配
    """
    def __init__(self, proto_emb: torch.Tensor, bert_dim: int, hidden_dim: int, tau_init: float = 10.0):
        super().__init__()
        self.proj = nn.Linear(bert_dim, hidden_dim)
        self.register_buffer("proto_raw", proto_emb)  # [C, bert_dim]
        self.tau = nn.Parameter(torch.tensor(tau_init, dtype=torch.float32))

    def get_proj(self) -> torch.Tensor:
        t = self.proj(self.proto_raw)        # [C, hidden]
        return F.normalize(t, dim=-1)

    def logits(self, visual_h: torch.Tensor) -> torch.Tensor:
        """
        visual_h: [B, hidden_dim]
        return: [B, C]
        """
        v = F.normalize(visual_h, dim=-1)
        t = self.get_proj()
        return self.tau * (v @ t.t())


# ===================== 双通道卷积 + 门控 =====================
class DualPathConv(nn.Module):
    def __init__(self, hidden_dim: int, gate_hidden: int):
        super().__init__()
        # 因果边
        self.causal = HeteroConv({
            ('visual', 'to', 'subtype'):      SAGEConv(hidden_dim, hidden_dim),
            ('subtype', 'to', 'feature'):     SAGEConv(hidden_dim, hidden_dim),
            ('feature', 'to', 'problem'):     SAGEConv(hidden_dim, hidden_dim),
            ('problem', 'to', 'solution'):    SAGEConv(hidden_dim, hidden_dim),
            ('subtype', 'reverse_to', 'visual'): SAGEConv(hidden_dim, hidden_dim),
        }, aggr='sum')

        # 相关边（同层KNN）
        self.corr = HeteroConv({
            ('subtype',  'corr', 'subtype'):  SAGEConv(hidden_dim, hidden_dim),
            ('feature',  'corr', 'feature'):  SAGEConv(hidden_dim, hidden_dim),
            ('problem',  'corr', 'problem'):  SAGEConv(hidden_dim, hidden_dim),
            ('solution', 'corr', 'solution'): SAGEConv(hidden_dim, hidden_dim),
        }, aggr='sum')

        # 门控
        self.gate = nn.ModuleDict({
            'visual':   nn.Sequential(nn.Linear(hidden_dim*2, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 1)),
            'subtype':  nn.Sequential(nn.Linear(hidden_dim*2, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 1)),
            'feature':  nn.Sequential(nn.Linear(hidden_dim*2, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 1)),
            'problem':  nn.Sequential(nn.Linear(hidden_dim*2, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 1)),
            'solution': nn.Sequential(nn.Linear(hidden_dim*2, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 1)),
        })

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict) -> Dict[str, torch.Tensor]:
        x_causal = self.causal(x_dict, edge_index_dict)
        x_corr   = self.corr(x_dict, edge_index_dict)

        out = {}
        for ntype, x in x_dict.items():
            if ntype not in x_corr:    # 视觉通常无同层相关边
                out[ntype] = F.elu(x_causal[ntype])
                continue
            cat = torch.cat([x_causal[ntype], x_corr[ntype]], dim=-1)
            g = torch.sigmoid(self.gate[ntype](cat))
            out[ntype] = F.elu(g * x_causal[ntype] + (1 - g) * x_corr[ntype])
        return out


# ===================== do(E) 干预（可选） =====================
class CausalInterventionLayer(nn.Module):
    """
    将全局拼接后的 [E;A;I;M] 特征中，屏蔽对 E 的外来输入，并可注入指定 E 值（简化版）
    """
    def __init__(self, n_e: int, n_a: int, n_i: int, n_m: int):
        super().__init__()
        self.n_e, self.n_a, self.n_i, self.n_m = n_e, n_a, n_i, n_m
        self.active = False
        self.e_value = None   # [hidden_dim] or None

    def set_do_env(self, e_value: Optional[torch.Tensor] = None):
        self.active = True
        self.e_value = e_value

    def clear(self):
        self.active = False
        self.e_value = None

    def apply(self, global_feat: torch.Tensor) -> torch.Tensor:
        """
        global_feat: [N_all, hidden_dim]，拼接顺序 E, A, I, M
        """
        if not self.active:
            return global_feat
        out = global_feat.clone()
        s0, s1 = 0, self.n_e
        if self.e_value is None:
            out[s0:s1] = 0.0
        else:
            out[s0:s1] = self.e_value.expand(self.n_e, -1)
        return out


# ===================== 主网络：CausalChainGNN =====================
class CausalChainGNN(nn.Module):
    def __init__(self,
                 n_e: int, n_a: int, n_i: int, n_m: int,
                 bert_dim: int, hidden_dim: int, gate_hidden: int,
                 e_proto_raw: torch.Tensor):
        super().__init__()
        self.n_e, self.n_a, self.n_i, self.n_m = n_e, n_a, n_i, n_m

        # 投影到 hidden_dim
        self.visual_proj = nn.Linear(config["vit_dim"], hidden_dim)
        self.text_proj   = nn.Linear(bert_dim, hidden_dim)

        # 双层双通道卷积 + 门控
        self.block1 = DualPathConv(hidden_dim, gate_hidden)
        self.block2 = DualPathConv(hidden_dim, gate_hidden)

        # do(E)
        self.intervene = CausalInterventionLayer(n_e, n_a, n_i, n_m)

        # 环境原型先验
        self.env_proto = EnvTextPrototypes(e_proto_raw, bert_dim, hidden_dim, tau_init=config["tau_init"])
        # 原型与GNN融合(逐类融合)：将 [logits_proto, logits_gnn] -> 1个logit
        self.env_fuse = nn.Linear(2, 1)

        # 链式头（E->A->I->M）
        self.subtype_classifier  = nn.Linear(hidden_dim, n_e)
        self.feature_classifier  = nn.Linear(hidden_dim*2, n_a)
        self.problem_classifier  = nn.Linear(hidden_dim*2, n_i)
        self.solution_classifier = nn.Linear(hidden_dim*2, n_m)

        # 嵌入（用于链式条件）
        self.e_embed = nn.Embedding(n_e, hidden_dim)
        self.a_embed = nn.Embedding(n_a, hidden_dim)
        self.i_embed = nn.Embedding(n_i, hidden_dim)

    def forward(self, x_dict, edge_index_dict, do_env: bool = False):
        # 0) 投影
        x_dict['visual']  = F.elu(self.visual_proj(x_dict['visual']))
        for ntype in ['subtype', 'feature', 'problem', 'solution']:
            x_dict[ntype] = F.elu(self.text_proj(x_dict[ntype]))

        # 1) 第1层双通道
        x_dict = self.block1(x_dict, edge_index_dict)

        # 2) 全局拼接 -> 可选 do(E)
        global_feat = torch.cat([x_dict['subtype'], x_dict['feature'], x_dict['problem'], x_dict['solution']], dim=0)
        if do_env:
            self.intervene.set_do_env(e_value=None)    # 置零干预；也可注入指定 e 嵌入
            global_feat = self.intervene.apply(global_feat)
            self.intervene.clear()
        # 拆回
        s0, s1 = 0, self.n_e
        x_dict['subtype'] = global_feat[s0:s1]
        s0, s1 = s1, s1 + self.n_a
        x_dict['feature'] = global_feat[s0:s1]
        s0, s1 = s1, s1 + self.n_i
        x_dict['problem'] = global_feat[s0:s1]
        s0, s1 = s1, s1 + self.n_m
        x_dict['solution'] = global_feat[s0:s1]

        # 3) 第2层双通道
        x_dict = self.block2(x_dict, edge_index_dict)

        # 4) E 原型 + GNN 融合
        logits_e_gnn   = self.subtype_classifier(x_dict['visual'])            # [B, n_e]
        logits_e_proto = self.env_proto.logits(x_dict['visual'])              # [B, n_e]
        cat = torch.stack([logits_e_proto, logits_e_gnn], dim=-1)             # [B, n_e, 2]
        logits_e = self.env_fuse(cat).squeeze(-1)                             # [B, n_e]

        # 5) 链式条件 A/I/M
        e_idx = torch.argmax(logits_e, dim=1)
        e_emb = self.e_embed(e_idx)
        logits_a = self.feature_classifier(torch.cat([x_dict['visual'], e_emb], dim=1))

        a_idx = torch.argmax(logits_a, dim=1)
        a_emb = self.a_embed(a_idx)
        logits_i = self.problem_classifier(torch.cat([x_dict['visual'], a_emb], dim=1))

        i_idx = torch.argmax(logits_i, dim=1)
        i_emb = self.i_embed(i_idx)
        logits_m = self.solution_classifier(torch.cat([x_dict['visual'], i_emb], dim=1))

        return {"E": logits_e, "A": logits_a, "I": logits_i, "M": logits_m, "E_proto": logits_e_proto}

    # 解析某张图片的预测索引（返回各类的下标）
    def interpret_idx(self, outputs: Dict[str, torch.Tensor], img_idx: int) -> Dict[str, int]:
        e = torch.argmax(outputs["E"][img_idx]).item()
        a = torch.argmax(outputs["A"][img_idx]).item()
        i = torch.argmax(outputs["I"][img_idx]).item()
        m = torch.argmax(outputs["M"][img_idx]).item()
        return {"E": e, "A": a, "I": i, "M": m}
    @torch.no_grad()
    def forward_with_doA(self, x_dict, edge_index_dict, force_a_idx: Optional[torch.Tensor], do_env: bool=False):
        """
        force_a_idx: LongTensor[B] 或 None。
        - None: 正常链式（A 用 argmax）
        - 给定: 强制把 A 的类别设为该张量，再用其嵌入计算 I/M
        返回: 同 forward 的 dict（包含 E/A/I/M 的 logits）
        """
        self.eval()

        # === 与 forward 前半段一致 ===
        x_local = {k: v.clone() for k, v in x_dict.items()}
        x_local['visual']  = F.elu(self.visual_proj(x_local['visual']))
        for ntype in ['subtype', 'feature', 'problem', 'solution']:
            x_local[ntype] = F.elu(self.text_proj(x_local[ntype]))

        x_local = self.block1(x_local, edge_index_dict)

        # do(E) 可选：按需保留/禁用
        global_feat = torch.cat([x_local['subtype'], x_local['feature'], x_local['problem'], x_local['solution']], dim=0)
        if do_env:
            self.intervene.set_do_env(e_value=None)
            global_feat = self.intervene.apply(global_feat)
            self.intervene.clear()
        s0, s1 = 0, self.n_e
        x_local['subtype'] = global_feat[s0:s1]
        s0, s1 = s1, s1 + self.n_a
        x_local['feature'] = global_feat[s0:s1]
        s0, s1 = s1, s1 + self.n_i
        x_local['problem'] = global_feat[s0:s1]
        s0, s1 = s1, s1 + self.n_m
        x_local['solution'] = global_feat[s0:s1]

        x_local = self.block2(x_local, edge_index_dict)

        # === 读出（与 forward 相同），但 A 的类别可被强制 ===
        logits_e_gnn   = self.subtype_classifier(x_local['visual'])
        logits_e_proto = self.env_proto.logits(x_local['visual'])
        cat = torch.stack([logits_e_proto, logits_e_gnn], dim=-1)
        logits_e = self.env_fuse(cat).squeeze(-1)

        # 正常算 A 的 logits（不用于选择类别时可忽略）
        e_idx = torch.argmax(logits_e, dim=1)
        e_emb = self.e_embed(e_idx)
        logits_a = self.feature_classifier(torch.cat([x_local['visual'], e_emb], dim=1))

        # === 关键：强制 A 的类别 ===
        if force_a_idx is None:
            a_idx = torch.argmax(logits_a, dim=1)
        else:
            a_idx = force_a_idx.to(x_local['visual'].device)

        a_emb = self.a_embed(a_idx)
        logits_i = self.problem_classifier(torch.cat([x_local['visual'], a_emb], dim=1))

        i_idx = torch.argmax(logits_i, dim=1)
        i_emb = self.i_embed(i_idx)
        logits_m = self.solution_classifier(torch.cat([x_local['visual'], i_emb], dim=1))

        return {"E": logits_e, "A": logits_a, "I": logits_i, "M": logits_m, "A_forced": a_idx}


# ===================== 训练/测试 =====================
def train_scene_kg():
    device = config["device"]

    # 构图
    dataset = PowerlineDataset(config)
    graph, meta = dataset.build_hetero_graph()

    # 准备模型
    n_e = graph['subtype'].x.size(0)
    n_a = graph['feature'].x.size(0)
    n_i = graph['problem'].x.size(0)
    n_m = graph['solution'].x.size(0)

    model = CausalChainGNN(
        n_e, n_a, n_i, n_m,
        bert_dim=config["bert_dim"],
        hidden_dim=config["hidden_dim"],
        gate_hidden=config["gate_hidden"],
        e_proto_raw=graph['subtype'].x.clone(),  # 直接用 E 的BERT向量做原型
    ).to(device)

    # 将图上的节点特征转到 device
    for k in graph.x_dict.keys():
        graph[k].x = graph[k].x.to(device)
    for triplet in graph.edge_index_dict:
        graph[triplet].edge_index = graph[triplet].edge_index.to(device)
    graph['visual'].y_subtype = graph['visual'].y_subtype.to(device)
    graph['visual'].y_feature = graph['visual'].y_feature.to(device)
    graph['visual'].y_problem = graph['visual'].y_problem.to(device)
    graph['visual'].y_solution = graph['visual'].y_solution.to(device)

    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # 训练循环
    model.train()
    for ep in range(1, config["num_epochs"] + 1):
        optim.zero_grad()
        out = model(graph.x_dict, graph.edge_index_dict, do_env=False)

        loss = torch.tensor(0.0, device=device)
        # 有效掩码（忽略无效标签=-1）
        def add_ce(logits, ys):
            mask = ys != -1
            if mask.sum() == 0: 
                return torch.tensor(0.0, device=device)
            return F.cross_entropy(logits[mask], ys[mask])

        lE  = add_ce(out["E"], graph['visual'].y_subtype)
        lA  = add_ce(out["A"], graph['visual'].y_feature)
        lI  = add_ce(out["I"], graph['visual'].y_problem)
        lM  = add_ce(out["M"], graph['visual'].y_solution)
        lEp = config["proto_ce_weight"] * add_ce(out["E_proto"], graph['visual'].y_subtype)

        loss = lE + lA + lI + lM + lEp
        loss.backward()
        optim.step()

        print(f"Epoch {ep:03d} | Loss {loss.item():.4f} | "
              f"E {lE.item():.4f} A {lA.item():.4f} I {lI.item():.4f} M {lM.item():.4f} Ep {lEp.item():.4f}")

    torch.save(model.state_dict(), "scene_causal_kg_only512.pth")
    print("训练完成，已保存: scene_causal_kg_only512.pth")
    return model, graph, meta


@torch.no_grad()
def test_once(model: CausalChainGNN, graph: HeteroData, meta: Dict, img_idx: int = 1):
    model.eval()
    out = model(graph.x_dict, graph.edge_index_dict, do_env=False)
    pred_idx = model.interpret_idx(out, img_idx)

    # 将 idx 转为 id/name（若需要）
    e_name = meta["e_names"][pred_idx["E"]] if pred_idx["E"] < len(meta["e_names"]) else "未知E"
    a_name = meta["a_names"][pred_idx["A"]] if pred_idx["A"] < len(meta["a_names"]) else "未知A"
    i_name = meta["i_names"][pred_idx["I"]] if pred_idx["I"] < len(meta["i_names"]) else "未知I"
    m_name = meta["m_names"][pred_idx["M"]] if pred_idx["M"] < len(meta["m_names"]) else "未知M"

    print("\n==== 单样本预测 ====")
    print(f"E(环境) : {pred_idx['E']} - {e_name}")
    print(f"A(特点) : {pred_idx['A']} - {a_name}")
    print(f"I(问题) : {pred_idx['I']} - {i_name}")
    print(f"M(措施) : {pred_idx['M']} - {m_name}")


if __name__ == "__main__":
    # 1) 训练或载入已训好的模型
    model, graph, meta = train_scene_kg()   # 或 torch.load_state_dict(...); model.eval()

    # 2) 选一张图做 C-Cons（idx 可换）
    idx_img = 0
    stats = ccons_doA_for_image(
        model, graph, idx_img,
        delta_I=0.02, delta_M=0.02,         # 结构方向阈值（可设 0 或小正数去噪）
        delta_lab_I=0.00, delta_lab_M=0.00  # 标签级增益阈值（可为 0）
    )

    print(f"[C-Cons] structural={stats['C_struct']:.3f}, label={stats['C_label']:.3f}")
    print("保存图：ccons_struct_deltas.png, ccons_label_hits.png")

