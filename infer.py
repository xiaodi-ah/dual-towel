#!/usr/bin/env python3
"""
双塔多模态 ET-BERT 推理脚本
===========================
支持 PCAP 文件和 TSV 文件推理
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uer.utils.vocab import Vocab
from uer.layers import str2embedding
from uer.encoders import str2encoder

from model import DualTowerETBERT
from data import PCAPProcessor, HexTokenizer


def infer_flow(model, tokenizer, vocab, hex_str, traffic_matrix, device, max_sem_len=512):
    """单流推理"""
    model.eval()
    
    # 处理语义特征
    tokens = tokenizer.tokenize(hex_str)
    sem_ids = [vocab.get(t, vocab.get("00", 0)) for t in tokens[:max_sem_len]]
    sem_ids += [vocab.get("[PAD]", 0)] * (max_sem_len - len(sem_ids))
    
    sem_ids = torch.tensor([sem_ids[:max_sem_len]], dtype=torch.long, device=device)
    seg = torch.ones(1, max_sem_len, dtype=torch.long, device=device)
    traffic = torch.tensor([traffic_matrix], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        _, logits = model(sem_ids, traffic, seg)
    
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred = int(probs.argmax())
    return pred, float(probs[pred]), probs.tolist()


def main():
    parser = argparse.ArgumentParser(description="Dual-Tower ET-BERT Inference")
    
    # 模型参数
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--config", type=str, default="../models/bert_base_config.json")
    parser.add_argument("--vocab", type=str, default="../models/encryptd_vocab.txt")
    
    # 输入输出
    parser.add_argument("--pcap", type=str, help="PCAP 文件路径")
    parser.add_argument("--output", type=str, default="predictions.json")
    parser.add_argument("--labels", type=str, help="标签映射 JSON")
    
    # 模型架构
    parser.add_argument("--labels_num", type=int, default=10)
    parser.add_argument("--fusion", choices=['attention', 'concat'], default='attention')
    parser.add_argument("--max_pkt_len", type=int, default=100, help="流量时序矩阵长度")
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
    cfg_obj.max_seq_length = 512
    cfg_obj.remove_embedding_layernorm = False
    cfg_obj.emb_size = cfg_obj.hidden_size
    
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
    
    # 处理 PCAP
    # bigram=False 因为 PCAPProcessor 已经输出 bigram 格式
    tokenizer = HexTokenizer(512, bigram=False)
    processor = PCAPProcessor(
        max_packets=args.max_pkt_len,
        payload_packets=5,  # 语义塔取前5个包，与原始 ET-BERT 一致
        payload_len=128     # 每包取128字节 payload
    )
    
    print(f"Processing: {args.pcap}")
    flows = processor.process_pcap(args.pcap, use_bigram=True)
    print(f"Found {len(flows)} flows")
    
    results = {}
    for key, data in flows.items():
        pred, conf, probs = infer_flow(
            model, tokenizer, vocab,
            data['semantic_hex'], data['traffic_matrix'],
            device
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


if __name__ == "__main__":
    main()
