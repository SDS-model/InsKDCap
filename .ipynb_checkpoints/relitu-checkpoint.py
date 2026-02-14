# -*- coding: utf-8 -*-
import re, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# =========================
# 1) 槽位定义（10个，含 SpatialRel）
# =========================
SLOTS = ["Strings","Color","Shape","Material","SpatialRel","DefectLoc","DefectType","EnvFeat","Impact","Measure"]
slot2idx = {s:i for i,s in enumerate(SLOTS)}

# =========================
# 2) 句子（可随时改）
# =========================
SENTENCE = (
    "A string of white disc-shaped ceramic insulators are vertically connected to the transmission line, "
    "and one of the insulators in the middle is damaged. "
    "This environment has the characteristics of high temperature and little rain, "
    "which can easily cause flashover due to surface contamination of insulators. "
    "Regular cleaning of insulator strings is recommended."
)

# =========================
# 3)（可选）自动规则：正则 -> 目标槽分布
#    提供合理默认，和手动标注一起使用（相加后再归一化）
# =========================
def dist(**kw):
    """便捷指定某些槽的权重，其余给极小泄露；最后会再归一化"""
    v = np.full(len(SLOTS), 1e-6, dtype=np.float32)
    for k, w in kw.items():
        v[slot2idx[k]] = w
    return v

RULES = [
    # 属性 / 外观
    (r"\bstring(s)?\b|\binsulator strings?\b",                dist(Strings=0.85)),
    (r"\bwhite\b",                                            dist(Color=0.9)),
    (r"\bdisc-?shaped\b|\bdisc\b",                            dist(Shape=0.9)),
    (r"\bceramic\b",                                          dist(Material=0.9)),
    (r"\bin the middle\b|\bmiddle\b|\bright end\b|\bleft end\b|\btop\b|\bbottom\b",
                                                             dist(DefectLoc=0.9)),
    (r"\bdamaged\b|\bbroken\b|\bchipped\b",                   dist(DefectType=0.9)),
    # 结构/空间关系
    (r"\bvertically connected to the transmission line\b|\bconnected to the transmission line\b",
                                                             dist(SpatialRel=0.75, Strings=0.15, Shape=0.05)),
    # 环境 & 影响 & 措施
    (r"\bhigh temperature\b|\blittle rain\b|\bdry\b",         dist(EnvFeat=0.9)),
    (r"\bflashover\b|\bsurface contamination\b",              dist(Impact=0.85, EnvFeat=0.05)),
    (r"\bregular cleaning\b|\bcleaning\b|\binspection\b",     dist(Measure=0.9)),
]

# =========================
# 4)（可选）手动标注：token索引/正则 -> 主槽
# =========================
MANUAL = {
    r"\bstring\b": "Strings",
    r"\bwhite\b": "Color",
    r"\bdisc-?shaped\b": "Shape",
    r"\bceramic\b": "Material",
    r"\bvertically connected to the transmission line\b": "SpatialRel",
    r"\bin the middle\b": "DefectLoc",
    r"\bdamaged\b": "DefectType",
    r"\bhigh temperature\b|\blittle rain\b": "EnvFeat",
    r"\bflashover\b": "Impact",
    r"\bRegular cleaning\b|\bcleaning\b": "Measure",
    # 也可写 {12: "SpatialRel"} 之类的精确索引
}

# =========================
# 5) 停用词：降低底噪即可避免“整列均匀”
# =========================
STOPWORDS = {
    "a","an","and","the","of","to","in","is","are","one","has","this","that",
    "which","can","due","be","with","for","on","at","by","or","as"
}

# =========================
# 6) 分词（保留标点；绘图时会显示）
# =========================
def tokenize(text):
    toks, spans = [], []
    for m in re.finditer(r"[A-Za-z\-]+|[.,;:!?]", text):
        toks.append(m.group(0))
        spans.append((m.start(), m.end()))
    return toks, spans

# =========================
# 7) 归一化与“多色但主槽更深”的控制
# =========================
def softmax_dense(v, temperature=1.0):
    x = v / max(1e-6, temperature)
    x = x - x.max()
    p = np.exp(x)
    p /= p.sum() + 1e-8
    return p

def apply_peak_and_floor(p, peak_gamma=1.4, epsilon_floor=0.01):
    """
    p: 归一化后的概率向量
    peak_gamma>1 会让主槽更深（p^gamma 再归一化）
    epsilon_floor 给所有槽一个下限，确保同列“多色”
    """
    # 先峰化
    if peak_gamma is not None and peak_gamma > 1.0:
        q = np.power(np.maximum(p, 1e-12), peak_gamma)
        p = q / (q.sum() + 1e-8)
    # 再加下限
    if epsilon_floor and epsilon_floor > 0.0:
        S = len(p)
        p = (1.0 - epsilon_floor) * p + epsilon_floor * (1.0 / S)
        p = p / (p.sum() + 1e-8)
    return p

# =========================
# 8) 构造 token×slot 软分布
# =========================
def build_soft_attn(sentence, rules, manual,
                    bg=0.015,          # 全局背景泄露
                    rule_peak=1.0,     # 规则命中强度
                    manual_boost=1.2,  # 手工主槽强化
                    sigma=0.8,         # 短语内扩散（越小越不外溢）
                    tau=0.8,           # 温度：控制“尖锐程度”
                    peak_gamma=1.4,    # >1 主槽更深，但仍保留副槽
                    epsilon_floor=0.01,# 为“多色列”提供下限
                    stop_bg=2e-4,      # 停用词的底噪更低
                    stop_tau=1.3       # 停用词温度更高（更平）
                    ):
    tokens, spans = tokenize(sentence)
    T, S = len(tokens), len(SLOTS)

    scores = np.full((T, S), bg, dtype=np.float32)

    # 1) 规则匹配 + 高斯扩散
    for pat, target in rules:
        for m in re.finditer(pat, sentence, flags=re.IGNORECASE):
            s, e = m.start(), m.end()
            covered = [ti for ti,(ts,te) in enumerate(spans) if not (te<=s or e<=ts)]
            if not covered:
                continue
            center = np.mean(covered)
            for t in range(T):
                g = np.exp(-0.5 * ((t - center) / max(1e-6, sigma))**2)
                scores[t] += rule_peak * g * target

    # 2) 手工标注加强
    for k, v in manual.items():
        if isinstance(k, int):
            idxs = [k]
        else:
            idxs = []
            for m in re.finditer(k, sentence, flags=re.IGNORECASE):
                s, e = m.start(), m.end()
                idxs += [ti for ti,(ts,te) in enumerate(spans) if not (te<=s or e<=ts)]
        if not idxs:
            continue
        slot = slot2idx[v]
        for t in idxs:
            scores[t, slot] += manual_boost

    # 3) 逐 token 归一化（带峰化与下限，保证“一列多色但主槽最深”）
    probs = np.zeros_like(scores)
    for t in range(T):
        tok = tokens[t].lower()
        vec = scores[t].copy()

        cur_tau = stop_tau if (tok in STOPWORDS or re.fullmatch(r"[.,;:!?]", tok)) else tau
        p = softmax_dense(vec, cur_tau)
        p = apply_peak_and_floor(p, peak_gamma=peak_gamma, epsilon_floor=epsilon_floor)

        # 停用词进一步抑制（让它们更“透明”，但仍保留轻微多色）
        if tok in STOPWORDS or re.fullmatch(r"[.,;:!?]", tok):
            p = 0.5 * p + 0.5 * (np.ones_like(p) / S)  # 更平，但因为 stop_bg 很小，总体仍很淡

        probs[t] = p

    return tokens, probs

# =========================
# 9) 自定义 colormap（浅黄→橙红→洋红→深紫→黑，越深越高）
# =========================
def make_my_cmap():
    colors = [
        "#fff3b0",  # very light yellow (low)
        "#f6b26b",  # orange
        "#d04b8b",  # magenta
        "#6f1d7a",  # deep purple
        "#000000",  # black (high)
    ]
    return LinearSegmentedColormap.from_list("y2black", colors, N=256)

# =========================
# 10) 绘图：槽(行) × token(列)
# =========================
def plot_slots_by_tokens(tokens, probs, slots, save="attn_demo/slots_by_tokens_soft.png", title=None):
    mat = probs.T  # [S,T]
    S, T = mat.shape
    plt.figure(figsize=(max(8, 0.45*T), max(5.8, 0.40*S)))
    plt.imshow(mat, aspect='auto', cmap=make_my_cmap(), vmin=0.0, vmax=1.0)
    cbar = plt.colorbar()
    cbar.set_label("Attention")
    xt = [tok if len(tok)<=14 else tok[:12]+"…" for tok in tokens]
    plt.xticks(range(T), xt, rotation=45, ha='right', fontsize=9)
    plt.yticks(range(S), slots, fontsize=10)
    plt.title(title or "Knowledge Slots × Tokens (Soft Columns, Peaked Max)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save), exist_ok=True)
    plt.savefig(save, dpi=240)
    plt.close()
    print(f"[Saved] {save}")

# =========================
# 11) 运行
# =========================
if __name__ == "__main__":
    tokens, probs = build_soft_attn(
        SENTENCE, RULES, MANUAL,
        bg=0.015,        # 全局底噪
        sigma=0.8,       # 扩散略窄，减少外溢
        tau=0.8,         # 非停用词温度（越小越尖锐）
        peak_gamma=1.45, # 主槽更深（>1），同时保留次要槽
        epsilon_floor=0.012, # 给每列所有槽一个小下限，确保“多色”
        stop_bg=2e-4,
        stop_tau=1.3
    )
    plot_slots_by_tokens(tokens, probs, SLOTS,
                         save="./attn_demo/slots_by_tokens_soft.png",
                         title="Distribution of Attention Weights in Different Regions of Displayed Knowledge")
