#!/usr/bin/env python3
"""
t-SNE 特征可视化
================
对比原始 ET-BERT 和双塔模型的特征空间分布
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uer.utils.vocab import Vocab
from uer.layers import str2embedding
from uer.encoders import str2encoder

from model import DualTowerETBERT
from data import HexTokenizer, load_tsv_data, TRAFFIC_FEATURE_DIM

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("Error: Please install scikit-learn and matplotlib")
    print("pip install scikit-learn matplotlib")
    sys.exit(1)


def extract_features(model, data, tokenizer, vocab, device, max_sem_len=128, max_pkt_len=100, ablation='both'):
    """
    提取模型特征向量（分类头之前的pooled features）
    
    Returns:
        features: [N, hidden_size]
        labels: [N]
    """
    model.eval()
    all_features = []
    all_labels = []
    
    w2i = vocab.w2i
    unk_id = w2i.get("[UNK]", 1)
    pad_id = w2i.get("[PAD]", 0)
    
    with torch.no_grad():
        for item in tqdm(data, desc="Extracting features"):
            # 处理语义特征
            tokens = tokenizer.tokenize(item['hex'])
            sem_ids = [w2i.get(t, unk_id) for t in tokens[:max_sem_len]]
            sem_ids += [pad_id] * (max_sem_len - len(sem_ids))
            sem_ids = torch.tensor([sem_ids[:max_sem_len]], dtype=torch.long, device=device)
            seg = torch.ones(1, max_sem_len, dtype=torch.long, device=device)
            
            # 流量矩阵
            traffic_matrix = item.get('traffic_matrix', 
                              np.zeros((max_pkt_len, TRAFFIC_FEATURE_DIM), dtype=np.float32))
            traffic = torch.tensor([traffic_matrix], dtype=torch.float32, device=device)
            
            # ====== 提取特征（在分类头之前） ======
            if ablation == 'stat_only':
                traffic_enc = model.traffic_encoder(traffic)
                feature = model.stat_pooling(traffic_enc)
            
            elif ablation == 'semantic_only':
                sem = model.embedding(sem_ids, seg)
                sem_seq = model.encoder(sem, seg)
                feature = model.sem_pooling(sem_seq)
            
            else:  # both
                # 语义塔
                sem = model.embedding(sem_ids, seg)
                sem_seq = model.encoder(sem, seg)
                sem_pooled = model.sem_pooling(sem_seq)
                
                # 统计塔
                traffic_enc = model.traffic_encoder(traffic)
                stat_pooled = model.stat_pooling(traffic_enc)
                
                # 融合
                if model.fusion_type == 'attention':
                    feature = model.fusion(sem_pooled, stat_pooled)
                else:
                    feature = model.fusion(torch.cat([sem_pooled, stat_pooled], dim=-1))
            
            all_features.append(feature.cpu().numpy())
            all_labels.append(item['label'])
    
    features = np.vstack(all_features)
    labels = np.array(all_labels)
    
    return features, labels


def plot_tsne(features_dict, labels, title="t-SNE Visualization", save_path=None, 
              n_samples=2000, perplexity=30, random_state=42):
    """
    绘制 t-SNE 降维可视化
    
    Args:
        features_dict: {model_name: features_array}
        labels: [N] 标签数组（所有模型共享）
        n_samples: 采样数量（太多点会很慢）
    """
    n_models = len(features_dict)
    
    # 采样（如果样本太多）
    if len(labels) > n_samples:
        indices = np.random.choice(len(labels), n_samples, replace=False)
        labels = labels[indices]
        features_dict = {name: feat[indices] for name, feat in features_dict.items()}
    
    # 创建子图
    fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 6))
    if n_models == 1:
        axes = [axes]
    
    n_classes = labels.max() + 1
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_classes, 20)))
    if n_classes > 20:
        colors = plt.cm.hsv(np.linspace(0, 1, n_classes))
    
    for ax, (model_name, features) in zip(axes, features_dict.items()):
        print(f"\nRunning t-SNE for {model_name}...")
        
        # t-SNE 降维
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, 
                    n_iter=1000, verbose=1)
        features_2d = tsne.fit_transform(features)
        
        # 绘制散点图
        for i in range(n_classes):
            mask = labels == i
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      c=[colors[i]], label=f'Class {i}', alpha=0.6, s=20, edgecolors='none')
        
        ax.set_title(f'{model_name}\n({len(features)} samples, {n_classes} classes)', 
                     fontsize=12)
        ax.set_xlabel('t-SNE Dim 1')
        ax.set_ylabel('t-SNE Dim 2')
        
        # 只在第一个子图显示图例（类别太多会挡住图）
        if n_classes <= 10 and ax == axes[0]:
            ax.legend(loc='upper right', fontsize=8, ncol=2)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="t-SNE Feature Visualization")
    
    # 模型路径
    parser.add_argument("--model1", type=str, required=True, help="模型1路径 (如原始ET-BERT)")
    parser.add_argument("--model1_name", type=str, default="Original ET-BERT")
    parser.add_argument("--model1_ablation", type=str, default="semantic_only", 
                        choices=['both', 'semantic_only', 'stat_only'])
    
    parser.add_argument("--model2", type=str, help="模型2路径 (如双塔)")
    parser.add_argument("--model2_name", type=str, default="Dual-Tower ET-BERT")
    parser.add_argument("--model2_ablation", type=str, default="both",
                        choices=['both', 'semantic_only', 'stat_only'])
    
    # 数据和配置
    parser.add_argument("--test_data", type=str, required=True, help="测试集TSV")
    parser.add_argument("--config", type=str, default="../models/bert_base_config.json")
    parser.add_argument("--vocab", type=str, default="../models/encryptd_vocab.txt")
    parser.add_argument("--labels_num", type=int, default=10)
    
    # 可视化参数
    parser.add_argument("--n_samples", type=int, default=2000, help="采样数量（加速t-SNE）")
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--output", type=str, default="figures/tsne_comparison.png")
    
    # 模型架构
    parser.add_argument("--max_sem_len", type=int, default=128)
    parser.add_argument("--max_pkt_len", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 加载配置
    with open(args.config) as f:
        cfg = json.load(f)
    cfg['labels_num'] = args.labels_num
    cfg['dropout'] = 0.1
    cfg_obj = type('Config', (), cfg)()
    
    # 加载词表
    vocab = Vocab()
    vocab.load(args.vocab)
    
    # 加载数据
    tokenizer = HexTokenizer(args.max_sem_len, bigram=False)
    test_data = load_tsv_data(args.test_data, max_packets=args.max_pkt_len)
    print(f"Test samples: {len(test_data)}")
    
    # 配置 TransformerEncoder
    cfg_obj.max_seq_length = args.max_sem_len
    cfg_obj.remove_embedding_layernorm = False
    cfg_obj.emb_size = cfg_obj.hidden_size
    cfg_obj.mask = 'fully_visible'
    cfg_obj.parameter_sharing = False
    cfg_obj.factorized_embedding_parameterization = False
    cfg_obj.layernorm_positioning = 'post'
    cfg_obj.relative_position_embedding = False
    cfg_obj.remove_transformer_bias = False
    cfg_obj.layernorm = 'normal'
    cfg_obj.relative_attention_buckets_num = 32
    cfg_obj.remove_attention_scale = False
    cfg_obj.feed_forward = 'dense'
    
    features_dict = {}
    
    # ========== 加载模型1 ==========
    print(f"\n{'='*60}")
    print(f"Loading Model 1: {args.model1_name}")
    print(f"{'='*60}")
    
    if args.model1_ablation != 'stat_only':
        sem_emb1 = str2embedding['word_pos_seg'](cfg_obj, len(vocab))
        sem_enc1 = str2encoder['transformer'](cfg_obj)
    else:
        sem_emb1, sem_enc1 = None, None
    
    model1 = DualTowerETBERT(
        cfg_obj, sem_emb1, sem_enc1,
        labels_num=args.labels_num,
        seq_len=args.max_pkt_len,
        ablation=args.model1_ablation
    ).to(device)
    
    model1.load_state_dict(torch.load(args.model1, map_location=device), strict=False)
    
    features1, labels = extract_features(
        model1, test_data, tokenizer, vocab, device,
        args.max_sem_len, args.max_pkt_len, args.model1_ablation
    )
    features_dict[args.model1_name] = features1
    print(f"Features shape: {features1.shape}")
    
    # ========== 加载模型2 ==========
    if args.model2:
        print(f"\n{'='*60}")
        print(f"Loading Model 2: {args.model2_name}")
        print(f"{'='*60}")
        
        if args.model2_ablation != 'stat_only':
            sem_emb2 = str2embedding['word_pos_seg'](cfg_obj, len(vocab))
            sem_enc2 = str2encoder['transformer'](cfg_obj)
        else:
            sem_emb2, sem_enc2 = None, None
        
        model2 = DualTowerETBERT(
            cfg_obj, sem_emb2, sem_enc2,
            labels_num=args.labels_num,
            seq_len=args.max_pkt_len,
            ablation=args.model2_ablation
        ).to(device)
        
        model2.load_state_dict(torch.load(args.model2, map_location=device), strict=False)
        
        features2, _ = extract_features(
            model2, test_data, tokenizer, vocab, device,
            args.max_sem_len, args.max_pkt_len, args.model2_ablation
        )
        features_dict[args.model2_name] = features2
        print(f"Features shape: {features2.shape}")
    
    # ========== t-SNE 可视化 ==========
    print(f"\n{'='*60}")
    print(f"Generating t-SNE visualization...")
    print(f"{'='*60}")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    plot_tsne(
        features_dict, labels,
        title="Feature Space Comparison (t-SNE)",
        save_path=args.output,
        n_samples=args.n_samples,
        perplexity=args.perplexity
    )
    
    print(f"\nDone! Check {args.output}")


if __name__ == "__main__":
    main()
