# -*- coding: utf-8 -*-
import json, math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ==========================
# 配置：哈希位长与“属性槽”切块
# ==========================
config = {
    "features_dir": "./data",             # ROI特征目录（.npz或.npy），向量维度D
    "annotation_file": "./data_converted.json",     # 标注文件
    "batch_size": 32,
    "num_epochs": 100,
    "lr": 5e-4,
    "hash_bits": 64,                      # K
    "use_adam": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # 槽位定义（顺序即位段顺序）
    # 每个槽有：name（用于记录/可视化）、num_classes（原子属性数）、bits（该槽位段位数）
    "slot_spec": [
            {"name": "number of strings", "num_classes": 3, "bits": 12},  # 0/1/2 -> "one/two/three strings"
            {"name": "color",             "num_classes": 4, "bits": 12},  # gray/green/white/brown
            {"name": "material",          "num_classes": 2, "bits": 10},  # glass/ceramic
            {"name": "defect location",   "num_classes": 5, "bits": 14},  # top/middle/left end/right end/bottom
            {"name": "defect type",       "num_classes": 3, "bits": 16},  # damage/flashover/self explosion
    ],

    # 损失权重
    "w_ml": 2.0,           # 同属性聚合 + 属性分离（槽内）
    "w_sep": 0.7,          # 中心-中心分离（槽内）
    "w_quant": 0.05,        # 量化
    "w_reg": 0.01,         # L2正则
    "w_decorr": 0.0,       # 跨槽去相关（可先置0）

    # 边际与超参
    "margin": 1.75,         # 属性分离边际
    "gamma_sep": 3.5,      # 中心-中心最小间隔
    "grad_clip": 1.0,

    # 位置编码的放置：'pre'（推荐，前置）或 'mid'（在sign之前）
    "posenc_mode": "pre",
}


# ==========================
# 数据集
# 支持两种标注：扁平多标签 -> 槽映射；或 labels_per_slot 直接给出每槽类别id
# ==========================
class PowerLineSlotDataset(Dataset):
    def __init__(self, features_dir: str, annotation_file: str, slot_spec: List[Dict]):
        self.features_dir = Path(features_dir)
        with open(annotation_file, "r", encoding="utf-8") as f:
            self.ann = json.load(f)

        self.slot_spec = slot_spec
        self.S = len(slot_spec)
        self.samples = []
        self.text_to_idx = {t: i for i, t in enumerate(self.ann.get("texts", []))}

        # 读取映射（两种情况兼容）
        for m in self.ann["mappings"]:
            visual_id = m["visual_id"]  # 对应特征文件名
            # 优先使用新格式：labels_per_slot = [c0, c1, ..., c_{S-1}]
            if "labels_per_slot" in m:
                labels_per_slot = m["labels_per_slot"]
                assert len(labels_per_slot) == self.S, "labels_per_slot长度需与slot数一致"
                self.samples.append({"visual_id": visual_id, "labels_per_slot": labels_per_slot})
            else:
                # 旧格式：text_ids多热，需要由调用方提供 texts->(slot_id,class_id) 的映射
                # 这里约定在 ann 里提供 text_to_slot_class: { text_string: [slot_id, class_id] }
                tmap = self.ann.get("text_to_slot_class", {})
                text_ids = m["text_ids"]
                labels_per_slot = [-1] * self.S
                for tid in text_ids:
                    text_string = self.ann["texts"][tid]
                    if text_string in tmap:
                        s_id, c_id = tmap[text_string]
                        labels_per_slot[s_id] = int(c_id)
                self.samples.append({"visual_id": visual_id, "labels_per_slot": labels_per_slot})

    def __len__(self): return len(self.samples)

    def _load_vec(self, path: Path) -> np.ndarray:
        # 支持 .npz{'features'或'g_feature'或'global'} 或 .npy
        if path.suffix == ".npy":
            v = np.load(path).astype(np.float32)
            return v.squeeze()
        arr = np.load(path, allow_pickle=True)
        for k in ["g_feature","features", "global"]:
            if k in arr:
                v = arr[k].astype(np.float32).squeeze()
                return v
        raise KeyError(f"未在 {path} 找到可用键")

    def __getitem__(self, idx):
        s = self.samples[idx]
        feat_path = self.features_dir / s["visual_id"]
        vec = self._load_vec(feat_path)            # [D]
        x = torch.tensor(vec, dtype=torch.float32)
        labels = torch.tensor(s["labels_per_slot"], dtype=torch.long)  # [S], 每槽一个类id（-1表示缺失）
        return x, labels


# ==========================
# 位置编码（长度=K）
# ==========================
class PositionalEncoding1D(nn.Module):
    def __init__(self, K: int, learnable: bool = True, dropout: float = 0.0):
        super().__init__()
        pe = torch.zeros(K)
        if learnable:
            self.pe = nn.Parameter(pe)    # 可学习
        else:
            self.register_buffer("pe", pe)  # 固定
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K]
        return self.dropout(x + self.pe)


# ==========================
# 混沌映射（逐位/逐段）
# ==========================
class SimplifiedChaosMappings(nn.Module):
    def __init__(self, steps: int = 6):
        super().__init__()
        self.steps = steps
        # 融合权重（可学习）
        self.w_logistic = nn.Parameter(torch.randn(1))
        self.w_sine = nn.Parameter(torch.randn(1))

    def logistic(self, x):
        out = x
        for _ in range(self.steps):
            out = 4.0 * out * (1.0 - out)
            out = torch.clamp(out, 0.0, 1.0)
        return out

    def sine(self, x):
        out = x * 2.0 - 1.0  # 映射到 [-1,1]
        for _ in range(self.steps):
            out = 4.0 * torch.sin(math.pi * out)
            out = torch.clamp(out, -1.0, 1.0)
        out = (out + 1.0) / 2.0  # 回到 [0,1]
        return out

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: [B, K] 先sigmoid到[0,1]
        x = torch.sigmoid(u)
        s1 = self.logistic(x)
        s2 = self.sine(x)
        w = torch.softmax(torch.stack([self.w_logistic, self.w_sine]), dim=0)
        return w[0] * s1 + w[1] * s2


# ==========================
# 属性感知序列层（对齐到K + 位置编码 + 混沌 + 线性头）
# ==========================
class AttributeAwareHashBackbone(nn.Module):
    def __init__(self, in_dim: int, K: int, posenc_mode: str = "pre"):
        super().__init__()
        self.align = nn.Linear(in_dim, K)
        self.posenc = PositionalEncoding1D(K, learnable=True, dropout=0.0)
        self.posenc_mode = posenc_mode
        self.proj1 = nn.Linear(K, K)
        self.chaos = SimplifiedChaosMappings(steps=6)
        self.proj2 = nn.Linear(K, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回实值哈希码 b ∈ R^K（训练时用 b；测试时 sign(b)）
        """
        zK = self.align(x)          # [B, K]
        if self.posenc_mode == "pre":
            h0 = self.posenc(zK)
            u = self.proj1(h0)
        else:
            u = self.proj1(zK)
        c = self.chaos(u)
        b = self.proj2(c)
        if self.posenc_mode == "mid":
            b = self.posenc(b)      # 二值化前中置
        return b                     # [B, K]


# ==========================
# 原子中心与“位段掩码”
# ==========================
def build_slot_masks(slot_spec: List[Dict], K: int) -> torch.Tensor:
    # 返回 [S, K] 0/1 掩码
    masks = []
    start = 0
    for s in slot_spec:
        kb = s["bits"]
        m = torch.zeros(K)
        m[start:start + kb] = 1.0
        masks.append(m)
        start += kb
    assert start == K, f"位数不匹配，sum(bits)={start} != K={K}"
    return torch.stack(masks, dim=0)  # [S, K]


class AtomicHashCenterTable(nn.Module):
    """
    维护每槽的原子属性中心向量（全K维，训练时通过掩码只看本槽位段）
    """
    def __init__(self, slot_spec: List[Dict], K: int):
        super().__init__()
        self.slot_spec = slot_spec
        self.S = len(slot_spec)
        self.K = K
        # 拼一个大表：总类数 × K
        self.slot_offsets = []
        self.class_offsets = []
        all_centers = []
        class_offset = 0
        for s_id, s in enumerate(slot_spec):
            C_s = s["num_classes"]
            mu_s = nn.Parameter(torch.randn(C_s, K) * 0.01)  # 小方差初始化
            self.register_parameter(f"centers_{s_id}", mu_s)
            self.class_offsets.append(class_offset)
            class_offset += C_s
        self.total_classes = class_offset

    def centers_of_slot(self, s_id: int) -> torch.Tensor:
        return getattr(self, f"centers_{s_id}")  # [C_s, K]


# ==========================
# 槽内损失（同属性聚合 + 属性分离 + 中心分离）
# ==========================

def slot_l1_dist(b: torch.Tensor, mu: torch.Tensor, M_s: torch.Tensor) -> torch.Tensor:

    #b: [B, K], mu: [*, K], M_s: [K]
    #返回逐元素 L1（带掩码）：
    #  - 若 mu维度为 [C_s, K]，输出 [B, C_s]
    #  - 若 mu维度为 [B, K]，输出 [B]

    if mu.dim() == 3:
        # not used here
        raise NotImplementedError
    if mu.size(0) == b.size(0):  # [B,K] vs [B,K]
        return torch.abs((b * M_s) - (mu * M_s)).sum(dim=-1)
    else:
        # [B,K] vs [C_s,K] -> [B,C_s]
        B, K = b.shape
        C = mu.size(0)
        bb = (b * M_s).unsqueeze(1).expand(B, C, K)
        mm = (mu * M_s).unsqueeze(0).expand(B, C, K)
        return torch.abs(bb - mm).sum(dim=-1)  # [B, C]
"""
def slot_l1_dist(b, mu, M_s):
    if mu.size(0) == b.size(0):  # [B,K] vs [B,K]
        diff = torch.abs((b * M_s) - (mu * M_s)).sum(dim=-1)
    else:                        # [B,K] vs [C_s,K] -> [B,C_s]
        B, K = b.shape; C = mu.size(0)
        bb = (b * M_s).unsqueeze(1).expand(B, C, K)
        mm = (mu * M_s).unsqueeze(0).expand(B, C, K)
        diff = torch.abs(bb - mm).sum(dim=-1)      # [B,C_s]
    k_s = M_s.sum().clamp(min=1.0)
    return diff / k_s
"""

def insuhash_losses(
    b: torch.Tensor,                 # [B, K] 实值码
    labels_per_slot: torch.Tensor,   # [B, S], 每槽一个类id（-1表示缺失）
    center_table: AtomicHashCenterTable,
    slot_masks: torch.Tensor,        # [S, K]
    margin: float = 1.75,
    gamma_sep: float = 3.5,
    w_ml: float = 2.0,
    w_sep: float = 0.7,
    w_quant: float = 0.05,
    w_reg: float = 0.01,
    w_decorr: float = 0.0,
):
    device = b.device
    B, K = b.shape
    S = slot_masks.size(0)

    loss_pos = torch.tensor(0.0, device=device)
    loss_neg = torch.tensor(0.0, device=device)
    loss_sep = torch.tensor(0.0, device=device)
    valid_slots = 0

    # 槽内聚合与分离
    for s_id in range(S):
        labels_s = labels_per_slot[:, s_id]  # [B]
        mask_valid = labels_s >= 0
        if mask_valid.sum() == 0:
            continue
        valid_slots += 1
        M_s = slot_masks[s_id].to(device)  # [K]
        centers_s = center_table.centers_of_slot(s_id)  # [C_s, K]

        # 正类中心
        mu_pos = centers_s[labels_s[mask_valid]]  # [Bv, K]
        b_valid = b[mask_valid]                   # [Bv, K]
        d_pos = slot_l1_dist(b_valid, mu_pos, M_s)  # [Bv]
        loss_pos += d_pos.mean()

        # 负类分离：对所有类计算距离，然后对非真值类施加 margin
        d_all = slot_l1_dist(b_valid, centers_s, M_s)  # [Bv, C_s]
        Bv, Cs = d_all.shape
        rows = torch.arange(Bv, device=device)
        d_true = d_all[rows, labels_s[mask_valid]]
        # 构造负样本掩码
        neg_mask = torch.ones_like(d_all, dtype=torch.bool)
        neg_mask[rows, labels_s[mask_valid]] = False
        d_neg = d_all[neg_mask]
        loss_neg += F.relu(margin - d_neg).mean()

        # 中心-中心分离（同槽）
        if Cs > 1:
            mu_i = centers_s.unsqueeze(1).expand(Cs, Cs, K)
            mu_j = centers_s.unsqueeze(0).expand(Cs, Cs, K)
            d_cc = torch.abs((mu_i - mu_j) * M_s).sum(dim=-1)  # [Cs, Cs]
            tri = torch.triu(torch.ones_like(d_cc, dtype=torch.bool), diagonal=1)
            if tri.sum() > 0:
                loss_sep += F.relu(gamma_sep - d_cc[tri]).mean()

    if valid_slots == 0:  # 全部缺失，避免除0
        valid_slots = 1

    loss_ml = (loss_pos + loss_neg) / valid_slots
    loss_sep = loss_sep / valid_slots

    # 量化：|b|-1 -> 0
    loss_quant = torch.mean((torch.abs(b) - 1.0) ** 2)

    # 正则
    reg = torch.norm(b, p=2)
    for s_id in range(S):
        reg = reg + torch.norm(center_table.centers_of_slot(s_id), p=2)

    # 跨槽去相关（可选）
    loss_decorr = torch.tensor(0.0, device=device)
    if w_decorr > 0:
        # 简单的协方差F范数
        for s1 in range(S):
            M1 = slot_masks[s1].to(device)
            b1 = b * M1
            for s2 in range(s1 + 1, S):
                M2 = slot_masks[s2].to(device)
                b2 = b * M2
                # 去均值
                b1c = b1 - b1.mean(dim=0, keepdim=True)
                b2c = b2 - b2.mean(dim=0, keepdim=True)
                cov = (b1c.t() @ b2c) / (B - 1 + 1e-6)  # [K,K]
                loss_decorr += torch.norm(cov, p='fro')

    total = (
        w_ml * loss_ml
        + w_sep * loss_sep
        + w_quant * loss_quant
        + w_reg * reg
        + w_decorr * loss_decorr
    )
    stats = {
        "loss_total": total.item(),
        "loss_ml": loss_ml.item(),
        "loss_sep": loss_sep.item(),
        "loss_quant": loss_quant.item(),
        "loss_reg": reg.item(),
        "loss_decorr": loss_decorr.item(),
    }
    return total, stats


# ==========================
# 主模型：InsuHash（槽位段）
# ==========================
class MultiSlotSeqHash(nn.Module):
    def __init__(self, feature_dim: int, slot_spec: List[Dict], K: int, posenc_mode: str = "pre"):
        super().__init__()
        self.K = K
        self.slot_spec = slot_spec
        self.S = len(slot_spec)
        # 位段掩码（buffer）
        self.register_buffer("slot_masks", build_slot_masks(slot_spec, K))  # [S, K]
        # 主干（对齐到K + 位置编码 + 混沌 + 线性头）
        self.backbone = AttributeAwareHashBackbone(feature_dim, K, posenc_mode=posenc_mode)
        # 原子中心表
        self.center_table = AtomicHashCenterTable(slot_spec, K)

    def forward(self, x: torch.Tensor):
        """
        x: [B, D]  ->  b: [B, K], centers_s: list of [C_s, K]
        """
        b = self.backbone(x)
        centers = [self.center_table.centers_of_slot(s) for s in range(self.S)]
        return b, centers, self.slot_masks

    @staticmethod
    def binarize(b: torch.Tensor) -> torch.Tensor:
        return torch.sign(b)  # {-1, +1}


# ==========================
# 训练与评估
# ==========================
def train_insuhash():
    ds = PowerLineSlotDataset(config["features_dir"], config["annotation_file"], config["slot_spec"])
    dl = DataLoader(ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    # 特征维度
    D = ds[0][0].shape[0]
    K = config["hash_bits"]

    model = MultiSlotSeqHash(D, config["slot_spec"], K, posenc_mode=config["posenc_mode"]).to(config["device"])
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"]) if config["use_adam"] \
        else torch.optim.RMSprop(model.parameters(), lr=config["lr"])

    for ep in range(1, config["num_epochs"] + 1):
        model.train()
        agg = {"loss_total": 0, "loss_ml": 0, "loss_sep": 0, "loss_quant": 0, "loss_reg": 0, "loss_decorr": 0}
        n_batches = 0
        for feats, labels_per_slot in dl:
            feats = feats.to(config["device"])                # [B, D]
            labels_per_slot = labels_per_slot.to(config["device"])  # [B, S]
            opt.zero_grad()
            b, _, slot_masks = model(feats)                  # b:[B,K]
            loss, stats = insuhash_losses(
                b, labels_per_slot,
                model.center_table, slot_masks,
                margin=config["margin"], gamma_sep=config["gamma_sep"],
                w_ml=config["w_ml"], w_sep=config["w_sep"], w_quant=config["w_quant"],
                w_reg=config["w_reg"], w_decorr=config["w_decorr"]
            )
            loss.backward()
            if config["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            opt.step()
            for k in agg: agg[k] += stats[k]
            n_batches += 1

        print(f"Epoch {ep:02d} | " + " ".join([f"{k}={agg[k]/n_batches:.4f}" for k in agg]))

    torch.save(model.state_dict(), "insuhash_slots.pth")
    print("已保存模型到 insuhash_slots.pth")
    return model, ds


@torch.no_grad()
def evaluate_slots(model: MultiSlotSeqHash, ds: PowerLineSlotDataset, top_k: int = 5):
    """
    简单示例：计算每个槽的Top-1准确率与mAP@K（把“同槽同类”为相关）
    """
    model.eval()
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    all_codes, all_labels = [], []
    for feats, labels in dl:
        feats = feats.to(config["device"])
        b, _, _ = model(feats)
        bh = MultiSlotSeqHash.binarize(b).cpu()  # [-1,+1]
        all_codes.append(bh)
        all_labels.append(labels)
    codes = torch.cat(all_codes, dim=0)          # [N,K]
    labels = torch.cat(all_labels, dim=0)        # [N,S]

    N, K = codes.shape
    S = labels.shape[1]
    # 内积等价排序（汉明）：
    sims = codes @ codes.t()                     # [N,N]
    # 自身屏蔽
    sims.fill_diagonal_(-1e9)

    # 每槽Top-1 Acc
    acc_slots = []
    for s in range(S):
        correct = 0; total = 0
        for i in range(N):
            gt = labels[i, s].item()
            if gt < 0: continue
            # Top-1近邻
            j = torch.argmax(sims[i]).item()
            if labels[j, s].item() == gt:
                correct += 1
            total += 1
        acc = correct / max(total, 1)
        acc_slots.append(acc)

    print("槽级Top-1 Acc:", ["{:.3f}".format(a) for a in acc_slots])
    print("平均Acc: {:.3f}".format(sum(acc_slots) / len(acc_slots) if acc_slots else 0.0))

import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_slot_scatter(B_slot, y, centers, title="slot scatter (PCA-2D)", save_dir="figs", basename="slot"):
    """
    将槽位嵌入降到2D并绘制散点；中心用★。会同时显示与保存。
    保存路径：{save_dir}/{basename}.png
    """
    _ensure_dir(save_dir)

    X = np.vstack([B_slot, centers])     # [N+C, K]
    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(X)
    N = B_slot.shape[0]
    C = centers.shape[0]
    pts, ctr = X2[:N], X2[N:N+C]

    plt.figure(figsize=(6.6, 5.2))
    cmap = plt.get_cmap("tab10")
    # 样本
    for c in range(C):
        idx = (y == c)
        if idx.sum() == 0: 
            continue
        plt.scatter(pts[idx,0], pts[idx,1], s=10, alpha=0.65, label=f"class {c}", color=cmap(c))
    # 中心
    plt.scatter(ctr[:,0], ctr[:,1], s=180, marker="*", color="k", edgecolors="w", linewidths=0.8, label="centers")

    plt.title(title)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.legend(loc="best", fontsize=9, ncol=2)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{basename}.png")
    plt.savefig(out_path, dpi=220)
    print(f"[Saved] {out_path}")
    try:
        plt.show()   # 有GUI时会弹窗；服务器上没有GUI也没关系，文件已保存
    except Exception:
        pass
    plt.close()

@torch.no_grad()
def evaluate_slot_geometry(model, ds, slot_id: int, slot_name: str, save_dir="figs"):
    """抽取该槽位的实值嵌入，打印聚类指标，并调用 plot 保存+显示。"""
    model.eval()
    dl = DataLoader(ds, batch_size=128, shuffle=False)
    device = next(model.parameters()).device

    all_b, all_y = [], []
    for feats, labels in dl:
        feats = feats.to(device)
        b, _, _ = model(feats)              # [B,K] 实值码（未sign）
        all_b.append(b.cpu())
        all_y.append(labels[:, slot_id].cpu())
    B = torch.cat(all_b, 0).cpu().numpy()   # [N,K]
    y = torch.cat(all_y, 0).cpu().numpy()   # [N]
    mask = model.slot_masks[slot_id].cpu().numpy()  # [K]

    valid = (y >= 0)
    B_slot = (B[valid] * mask)              # [N_valid, K]
    y_valid = y[valid]
    centers = (model.center_table.centers_of_slot(slot_id).cpu().numpy() * mask)  # [C_s, K]

    # --- 计算几何指标 ---
    # 类内散度
    mu_y = centers[y_valid]                         # [N_valid,K]
    D_intra = np.abs(B_slot - mu_y).sum(axis=-1).mean()

    # 类间最小中心距
    C = centers.shape[0]
    D_inter = np.inf
    for i in range(C):
        for j in range(i+1, C):
            D_inter = min(D_inter, np.abs(centers[i]-centers[j]).sum())
    F_ratio = D_inter / (D_intra + 1e-8)

    # Silhouette（欧氏近似足够；想要L1可自行写pairwise L1）
    try:
        sil = silhouette_score(B_slot, y_valid, metric='euclidean')
    except Exception:
        sil = float("nan")

    print(f"[{slot_name}] intra={D_intra:.4f}  inter_min={D_inter:.4f}  Fisher={F_ratio:.3f}  Silhouette={sil:.3f}")

    # --- 画图并保存 ---
    basename = f"slot_{slot_id}_{slot_name.replace(' ', '_')}"
    plot_slot_scatter(B_slot, y_valid, centers, title=f"{slot_name}: PCA scatter", save_dir=save_dir, basename=basename)


if __name__ == "__main__":
    model, ds = train_insuhash()
    # 选几个关键槽可视化，比如 颜色、材质、缺陷类型
    slot_names = [s["name"] for s in config["slot_spec"]]
    for s_id, s_name in enumerate(slot_names):
        evaluate_slot_geometry(model, ds, s_id, s_name)
    #evaluate_slots(model, ds, top_k=5)
