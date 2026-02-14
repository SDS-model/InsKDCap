import json
from collections import defaultdict

def analyze_vocab(vocab_file, vocab_format="txt"):
    """
    分析词汇表文件，返回大小及统计信息
    
    参数:
        vocab_file: 词汇表文件路径
        vocab_format: 文件格式，支持 'txt'、'json' 或 'dict'
    """
    vocab = {}
    
    # 1. 加载词汇表
    try:
        if vocab_format == "txt":
            # 每行一个token的txt文件（如：hello\nworld\n<pad>\n）
            with open(vocab_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    token = line.strip()
                    vocab[token] = idx
        
        elif vocab_format == "json":
            # JSON格式（如：{"<pad>": 0, "hello": 1, ...}）
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
        
        elif vocab_format == "dict":
            # 直接传入Python字典
            vocab = vocab_file
        else:
            raise ValueError("不支持的格式，请选择 'txt'、'json' 或 'dict'")
    
    except Exception as e:
        print(f"加载词汇表失败: {e}")
        return None

    # 2. 基础检查
    vocab_size = len(vocab)
    print(f"词汇表大小: {vocab_size}")
    
    # 3. 检查特殊Token
    special_tokens = {
        "pad_token": ("<pad>", 0),
        "bos_token": ("<bos>", 1),
        "eos_token": ("<eos>", 2),
        "unk_token": ("<unk>", 3)
    }
    
    print("\n特殊Token检查:")
    for name, (default_token, expected_id) in special_tokens.items():
        found = False
        for token, idx in vocab.items():
            if token == default_token and idx == expected_id:
                print(f"✓ {name}: '{token}' (ID={idx})")
                found = True
                break
        if not found:
            print(f"× {name}: 未找到或ID不匹配 (期望 '{default_token}' ID={expected_id})")

    # 4. 统计Token长度分布
    len_dist = defaultdict(int)
    for token in vocab.keys():
        len_dist[len(token)] += 1
    
    print("\nToken长度分布:")
    for length, count in sorted(len_dist.items()):
        print(f"{length}字符: {count}个Token")

    # 5. 示例展示
    print("\n示例Token (前5个):")
    for i, (token, idx) in enumerate(list(vocab.items())[:5]):
        print(f"{idx}: '{token}'")

    return vocab_size

# 使用示例 ==============================================

# 情况1：txt格式词汇表（每行一个token）
print("="*50 + "\n测试txt格式词汇表\n" + "="*50)
analyze_vocab("./mscoco_dataset/vocabulary.txt", vocab_format="txt")