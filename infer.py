#!/usr/bin/env python3
"""
双塔多模态 ET-BERT 推理脚本
===========================
支持:
  1. PCAP 文件推理
  2. TSV 测试集评估 (输出精度、混淆矩阵、分类报告)
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
from data import PCAPProcessor, HexTokenizer, load_tsv_data, TRAFFIC_FEATURE_DIM


def infer_flow(model, tokenizer, vocab, hex_str, traffic_matrix, device, max_sem_len=512):
    """单流推理"""
    model.eval()
    
    # 处理语义特征
    tokens = tokenizer.tokenize(hex_str)
    w2i = vocab.w2i
    unk_id = w2i.get("[UNK]", 1)
    pad_id = w2i.get("[PAD]", 0)
    sem_ids = [w2i.get(t, unk_id) for t in tokens[:max_sem_len]]
    sem_ids += [pad_id] * (max_sem_len - len(sem_ids))
    
    sem_ids = torch.tensor([sem_ids[:max_sem_len]], dtype=torch.long, device=device)
    seg = torch.ones(1, max_sem_len, dtype=torch.long, device=device)
    traffic = torch.tensor([traffic_matrix], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        _, logits = model(sem_ids, traffic, seg)
    
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred = int(probs.argmax())
    return pred, float(probs[pred]), probs.tolist()


def evaluate_tsv(model, tokenizer, vocab, tsv_path, device, max_sem_len=512, max_pkt_len=100, label_map=None):
    """
    在 TSV 测试集上评估模型
    返回: accuracy, predictions, labels, confusion_matrix
    """
    model.eval()
    
    # 加载数据
    data = load_tsv_data(tsv_path, max_packets=max_pkt_len)
    print(f"Test samples: {len(data)}")
    
    w2i = vocab.w2i
    unk_id = w2i.get("[UNK]", 1)
    pad_id = w2i.get("[PAD]", 0)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for item in tqdm(data, desc="Evaluating"):
            # 处理语义特征
            tokens = tokenizer.tokenize(item['hex'])
            sem_ids = [w2i.get(t, unk_id) for t in tokens[:max_sem_len]]
            sem_ids += [pad_id] * (max_sem_len - len(sem_ids))
            
            sem_ids = torch.tensor([sem_ids[:max_sem_len]], dtype=torch.long, device=device)
            seg = torch.ones(1, max_sem_len, dtype=torch.long, device=device)
            traffic = torch.tensor([item['traffic_matrix']], dtype=torch.float32, device=device)
            
            _, logits = model(sem_ids, traffic, seg)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred = int(probs.argmax())
            
            all_preds.append(pred)
            all_labels.append(item['label'])
            all_probs.append(probs)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算指标
    accuracy = (all_preds == all_labels).mean()
    
    # 混淆矩阵
    n_classes = max(all_labels.max(), all_preds.max()) + 1
    confusion = np.zeros((n_classes, n_classes), dtype=np.int32)
    for pred, label in zip(all_preds, all_labels):
        confusion[label, pred] += 1
    
    # 分类报告
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    
    # 计算每类指标
    precisions, recalls, f1s, supports = [], [], [], []
    for i in range(n_classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        support = confusion[i, :].sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
        
        class_name = label_map.get(str(i), f"class_{i}") if label_map else f"class_{i}"
        print(f"  {class_name:20s}  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}  support={support}")
    
    # 宏平均和加权平均
    macro_p = np.mean(precisions)
    macro_r = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    
    total_support = sum(supports)
    weighted_p = sum(p * s for p, s in zip(precisions, supports)) / total_support
    weighted_r = sum(r * s for r, s in zip(recalls, supports)) / total_support
    weighted_f1 = sum(f * s for f, s in zip(f1s, supports)) / total_support
    
    print("-"*60)
    print(f"  {'Macro Avg':20s}  P={macro_p:.4f}  R={macro_r:.4f}  F1={macro_f1:.4f}")
    print(f"  {'Weighted Avg':20s}  P={weighted_p:.4f}  R={weighted_r:.4f}  F1={weighted_f1:.4f}")
    print(f"\n  Accuracy: {accuracy:.4f} ({int(accuracy * len(data))}/{len(data)})")
    print("="*60)
    
    return {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_p),
        'macro_recall': float(macro_r),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_p),
        'weighted_recall': float(weighted_r),
        'weighted_f1': float(weighted_f1),
        'confusion_matrix': confusion.tolist(),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description="Dual-Tower ET-BERT Inference")
    
    # 模型参数
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--config", type=str, default="../models/bert_base_config.json")
    parser.add_argument("--vocab", type=str, default="../models/encryptd_vocab.txt")
    
    # 输入输出
    parser.add_argument("--pcap", type=str, help="PCAP 文件路径 (单文件推理)")
    parser.add_argument("--tsv", type=str, help="TSV 测试集路径 (批量评估)")
    parser.add_argument("--output", type=str, default="predictions.json")
    parser.add_argument("--labels", type=str, help="标签映射 JSON")
    
    # 模型架构
    parser.add_argument("--labels_num", type=int, default=10)
    parser.add_argument("--fusion", choices=['attention', 'concat'], default='attention')
    parser.add_argument("--max_pkt_len", type=int, default=100, help="流量时序矩阵长度")
    parser.add_argument("--max_sem_len", type=int, default=128, help="语义序列长度 (需与训练时一致)")
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 加载配置
    with open(args.config) as f:
        cfg = json.load(f)
    cfg['labels_num'] = args.labels_num
    cfg['dropout'] = cfg.get('dropout', 0.1)
    cfg_obj = type('Config', (), cfg)()
    
    # 加载词表
    vocab = Vocab()
    vocab.load(args.vocab)
    
    # 构建模型
    # 使用 word_pos_seg embedding (与原始 ET-BERT 一致)
    cfg_obj.max_seq_length = args.max_sem_len  # 需与训练时一致
    cfg_obj.remove_embedding_layernorm = False
    cfg_obj.emb_size = cfg_obj.hidden_size
    
    # TransformerEncoder 需要的额外配置
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
    
    sem_emb = str2embedding['word_pos_seg'](cfg_obj, len(vocab))
    sem_enc = str2encoder['transformer'](cfg_obj)
    model = DualTowerETBERT(
        cfg_obj, sem_emb, sem_enc,
        fusion=args.fusion,
        labels_num=args.labels_num,
        seq_len=args.max_pkt_len
    ).to(device)
    
    # 加载模型权重
    print(f"Loading model: {args.model}")
    model.load_state_dict(torch.load(args.model, map_location=device), strict=False)
    
    # 加载标签映射
    label_map = None
    if args.labels:
        with open(args.labels) as f:
            label_map = json.load(f)
    
    # bigram=False 因为数据已经是 bigram 格式
    tokenizer = HexTokenizer(512, bigram=False)
    
    # ========== TSV 测试集评估 ==========
    if args.tsv:
        print(f"\nEvaluating on: {args.tsv}")
        results = evaluate_tsv(
            model, tokenizer, vocab, args.tsv, device,
            max_sem_len=args.max_sem_len, max_pkt_len=args.max_pkt_len,
            label_map=label_map
        )
        
        # 保存结果
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
        return
    
    # ========== PCAP 文件推理 ==========
    elif args.pcap:
        processor = PCAPProcessor(
            max_packets=args.max_pkt_len,
            payload_packets=5,
            payload_len=128
        )
        
        print(f"Processing: {args.pcap}")
        flows = processor.process_pcap(args.pcap, use_bigram=True)
        print(f"Found {len(flows)} flows")
        
        results = {}
        for key, data in flows.items():
            pred, conf, probs = infer_flow(
                model, tokenizer, vocab,
                data['semantic_hex'], data['traffic_matrix'],
                device, max_sem_len=args.max_sem_len
            )
            
            flow_id = f"{key[0]}:{key[2]}->{key[1]}:{key[3]}"
            results[flow_id] = {
                'prediction': pred,
                'label': label_map[str(pred)] if label_map else f"class_{pred}",
                'confidence': conf,
                'probabilities': probs
            }
        
        # 保存结果
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    
    else:
        print("Error: Must provide --tsv or --pcap")
        return


if __name__ == "__main__":
    main()
