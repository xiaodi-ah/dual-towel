#!/usr/bin/env python3
"""
双塔多模态数据预处理脚本
========================
从 PCAP 文件生成双塔训练所需的 TSV 格式数据

输出 TSV 格式:
label \t hex_bigram \t pkt_len,iat,dir;pkt_len,iat,dir;...

使用方法:
---------
# 从按类别组织的 PCAP 目录生成数据
python preprocess.py --pcap_dir /path/to/pcaps --output train.tsv

# 目录结构应该是:
# pcap_dir/
# ├── class_0/
# │   ├── session1.pcap
# │   └── session2.pcap
# ├── class_1/
# │   └── ...
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import PCAPProcessor, TRAFFIC_FEATURE_DIM


def process_pcap_directory(pcap_dir, output_path, max_packets=100, 
                           payload_packets=5, payload_len=128,
                           samples_per_class=None, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    """
    处理按类别组织的 PCAP 目录
    
    Args:
        pcap_dir: PCAP 目录，每个子文件夹是一个类别
        output_path: 输出 TSV 文件路径
        max_packets: 统计塔取前 N 个包
        payload_packets: 语义塔取前 N 个包的 payload
        payload_len: 每包取多少字节 payload
        samples_per_class: 每类最多采样多少个 (None=全部)
        train_ratio: 训练集比例
    """
    processor = PCAPProcessor(
        max_packets=max_packets,
        payload_packets=payload_packets,
        payload_len=payload_len
    )
    
    # 收集所有类别 (过滤隐藏目录和系统目录)
    label_dirs = {}
    skip_dirs = {'.ipynb_checkpoints', '__pycache__', '.git', '.DS_Store'}
    for i, name in enumerate(sorted(os.listdir(pcap_dir))):
        if name.startswith('.') or name in skip_dirs:
            continue  # 跳过隐藏目录和系统目录
        dir_path = os.path.join(pcap_dir, name)
        if os.path.isdir(dir_path):
            label_dirs[len(label_dirs)] = (name, dir_path)
    
    print(f"发现 {len(label_dirs)} 个类别:")
    for label_id, (name, path) in label_dirs.items():
        print(f"  {label_id}: {name}")
    
    # 处理每个类别
    all_samples = []
    
    for label_id, (label_name, dir_path) in label_dirs.items():
        print(f"\n处理类别 {label_id}: {label_name}")
        
        # 收集 PCAP 文件
        pcap_files = []
        for root, dirs, files in os.walk(dir_path):
            for f in files:
                if f.endswith(('.pcap', '.pcapng')):
                    pcap_files.append(os.path.join(root, f))
        
        if samples_per_class and len(pcap_files) > samples_per_class:
            import random
            pcap_files = random.sample(pcap_files, samples_per_class)
        
        print(f"  找到 {len(pcap_files)} 个 PCAP 文件")
        
        class_samples = []
        for pcap_path in tqdm(pcap_files, desc=f"  处理 {label_name}"):
            try:
                flows = processor.process_pcap(pcap_path, use_bigram=True)
                
                for flow_key, data in flows.items():
                    hex_seq = data['semantic_hex']
                    traffic_matrix = data['traffic_matrix']
                    
                    # 跳过太短的流
                    if len(hex_seq.split()) < 10:
                        continue
                    
                    # 转换 traffic_matrix 为字符串格式
                    traffic_str = format_traffic_matrix(traffic_matrix)
                    
                    class_samples.append({
                        'label': label_id,
                        'hex': hex_seq,
                        'traffic': traffic_str
                    })
            except Exception as e:
                print(f"  警告: 处理 {pcap_path} 失败: {e}")
                continue
        
        print(f"  提取 {len(class_samples)} 个流样本")
        all_samples.extend(class_samples)
    
    print(f"\n总共 {len(all_samples)} 个样本")
    
    # 划分训练集、验证集、测试集 (80/10/10)
    import random
    random.shuffle(all_samples)
    
    n_total = len(all_samples)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    
    train_samples = all_samples[:n_train]
    valid_samples = all_samples[n_train:n_train + n_valid]
    test_samples = all_samples[n_train + n_valid:]
    
    # 保存
    base_name = os.path.splitext(output_path)[0]
    train_path = f"{base_name}_train.tsv"
    valid_path = f"{base_name}_valid.tsv"
    test_path = f"{base_name}_test.tsv"
    
    save_tsv(train_samples, train_path)
    save_tsv(valid_samples, valid_path)
    save_tsv(test_samples, test_path)
    
    print(f"\n保存完成:")
    print(f"  训练集: {train_path} ({len(train_samples)} 样本)")
    print(f"  验证集: {valid_path} ({len(valid_samples)} 样本) - 用于模型选择")
    print(f"  测试集: {test_path} ({len(test_samples)} 样本) - 用于最终评估")
    
    # 保存标签映射
    labels_path = f"{base_name}_labels.json"
    import json
    with open(labels_path, 'w') as f:
        label_map = {str(i): name for i, (name, _) in label_dirs.items()}
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print(f"  标签映射: {labels_path}")


def format_traffic_matrix(matrix):
    """将 traffic_matrix 转为 TSV 字符串格式"""
    parts = []
    for i in range(matrix.shape[0]):
        if matrix[i, 0] == 0 and matrix[i, 1] == 0:
            break  # 遇到全零行停止
        pkt_len = f"{matrix[i, 0]:.4f}"
        iat = f"{matrix[i, 1]:.4f}"
        direction = f"{int(matrix[i, 2])}"
        parts.append(f"{pkt_len},{iat},{direction}")
    return ';'.join(parts) if parts else "0,0,0"


def save_tsv(samples, output_path):
    """保存为 TSV 格式"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            line = f"{sample['label']}\t{sample['hex']}\t{sample['traffic']}\n"
            f.write(line)


def main():
    parser = argparse.ArgumentParser(description="双塔多模态数据预处理")
    
    parser.add_argument("--pcap_dir", type=str, required=True,
                        help="PCAP 目录 (每个子文件夹是一个类别)")
    parser.add_argument("--output", type=str, default="dataset.tsv",
                        help="输出文件前缀 (会生成 _train.tsv 和 _valid.tsv)")
    
    # 特征提取参数
    parser.add_argument("--max_packets", type=int, default=100,
                        help="统计塔: 每个流取前 N 个包 (默认 100)")
    parser.add_argument("--payload_packets", type=int, default=5,
                        help="语义塔: 取前 N 个包的 payload (默认 5，与原始 ET-BERT 一致)")
    parser.add_argument("--payload_len", type=int, default=128,
                        help="每包取多少字节 payload (默认 128)")
    
    # 采样参数
    parser.add_argument("--samples_per_class", type=int, default=None,
                        help="每类最多采样多少个 PCAP 文件 (默认全部)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例 (默认 0.8)")
    parser.add_argument("--valid_ratio", type=float, default=0.1,
                        help="验证集比例 (默认 0.1, 用于模型选择)")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="测试集比例 (默认 0.1, 用于最终评估)")
    
    args = parser.parse_args()
    
    process_pcap_directory(
        args.pcap_dir,
        args.output,
        max_packets=args.max_packets,
        payload_packets=args.payload_packets,
        payload_len=args.payload_len,
        samples_per_class=args.samples_per_class,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio
    )


if __name__ == "__main__":
    main()
