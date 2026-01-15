#!/usr/bin/env python3
"""
随机森林 Baseline (传统机器学习对比)
====================================
用于证明深度学习方法的优势
"""

import os
import sys
import argparse
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_tsv_data

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Error: Please install scikit-learn")
    print("pip install scikit-learn")
    sys.exit(1)


def extract_traffic_features(traffic_matrix):
    """
    从流量时序矩阵提取统计特征
    
    Input: [N, 3] (pkt_len, iat, direction)
    Output: [feature_dim] 统计特征向量
    """
    features = []
    
    # 找到有效数据（非零行）
    valid_mask = (traffic_matrix[:, 0] != 0) | (traffic_matrix[:, 2] != 0)
    valid_data = traffic_matrix[valid_mask]
    
    if len(valid_data) == 0:
        return np.zeros(30)  # 返回零向量
    
    # 对每个特征维度（pkt_len, iat, direction）提取统计量
    for dim in range(3):
        col = valid_data[:, dim]
        
        # 基础统计量
        features.append(np.mean(col))      # 均值
        features.append(np.std(col))       # 标准差
        features.append(np.min(col))       # 最小值
        features.append(np.max(col))       # 最大值
        features.append(np.median(col))    # 中位数
        
        # 分位数
        features.append(np.percentile(col, 25))   # 25%分位
        features.append(np.percentile(col, 75))   # 75%分位
        
        # 其他统计量
        if len(col) > 1:
            features.append(np.var(col))   # 方差
        else:
            features.append(0)
    
    # 包数量
    features.append(len(valid_data))
    
    # 方向统计（上行/下行比例）
    directions = valid_data[:, 2]
    features.append((directions > 0).sum())   # 上行包数
    features.append((directions < 0).sum())   # 下行包数
    
    # 总共: 3*8 + 1 + 2 = 27 个特征
    # 补齐到30
    features += [0] * (30 - len(features))
    
    return np.array(features[:30])


def extract_hex_features(hex_str):
    """
    从Hex序列提取统计特征
    
    Input: "1603 0301 0100 ..."
    Output: [feature_dim] 统计特征向量
    """
    features = []
    
    if not hex_str or not hex_str.strip():
        return np.zeros(20)
    
    tokens = hex_str.split()
    
    # 基础特征
    features.append(len(tokens))  # 序列长度
    
    # Token频率统计
    token_counts = Counter(tokens)
    features.append(len(token_counts))  # 唯一token数
    
    # 最常见token的频率
    if token_counts:
        most_common = token_counts.most_common(5)
        for i in range(5):
            if i < len(most_common):
                features.append(most_common[i][1] / len(tokens))  # 归一化频率
            else:
                features.append(0)
    else:
        features += [0] * 5
    
    # 熵（衡量随机性/加密程度）
    if len(tokens) > 0:
        probs = np.array([count / len(tokens) for count in token_counts.values()])
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        features.append(entropy)
    else:
        features.append(0)
    
    # Bigram 多样性
    if len(tokens) > 1:
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        features.append(len(set(bigrams)) / len(bigrams))  # 唯一bigram比例
    else:
        features.append(0)
    
    # 填充/截断到20维
    features += [0] * (20 - len(features))
    
    return np.array(features[:20])


def extract_combined_features(data):
    """
    组合提取流量特征 + Hex特征
    """
    X = []
    y = []
    
    print("Extracting features...")
    for item in data:
        # 流量特征 (30维)
        traffic_feat = extract_traffic_features(item['traffic_matrix'])
        
        # Hex特征 (20维)
        hex_feat = extract_hex_features(item['hex'])
        
        # 拼接 (50维)
        combined = np.concatenate([traffic_feat, hex_feat])
        
        X.append(combined)
        y.append(item['label'])
    
    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser(description="Random Forest Baseline")
    
    parser.add_argument("--train", type=str, required=True, help="训练数据TSV")
    parser.add_argument("--valid", type=str, help="验证数据TSV")
    parser.add_argument("--test", type=str, help="测试数据TSV")
    parser.add_argument("--output", type=str, default="rf_model.pkl", help="模型保存路径")
    
    # RF参数
    parser.add_argument("--n_estimators", type=int, default=100, help="树的数量")
    parser.add_argument("--max_depth", type=int, default=20, help="最大深度")
    parser.add_argument("--min_samples_split", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=-1, help="并行数 (-1=全部CPU)")
    
    parser.add_argument("--max_pkt_len", type=int, default=100)
    
    args = parser.parse_args()
    
    # 加载数据
    print("Loading training data...")
    train_data = load_tsv_data(args.train, max_packets=args.max_pkt_len)
    X_train, y_train = extract_combined_features(train_data)
    print(f"Train: {X_train.shape}, {len(set(y_train))} classes")
    
    # 训练模型
    print(f"\nTraining Random Forest (n_estimators={args.n_estimators}, max_depth={args.max_depth})...")
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        n_jobs=args.n_jobs,
        random_state=42,
        verbose=1
    )
    
    rf.fit(X_train, y_train)
    
    # 训练集准确率
    train_pred = rf.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"Train Accuracy: {train_acc:.4f}")
    
    # 验证集评估
    if args.valid:
        print("\nEvaluating on validation set...")
        valid_data = load_tsv_data(args.valid, max_packets=args.max_pkt_len)
        X_valid, y_valid = extract_combined_features(valid_data)
        
        valid_pred = rf.predict(X_valid)
        valid_acc = accuracy_score(y_valid, valid_pred)
        print(f"Valid Accuracy: {valid_acc:.4f}")
        
        print("\nValidation Classification Report:")
        print(classification_report(y_valid, valid_pred, digits=4))
    
    # 测试集评估
    if args.test:
        print("\nEvaluating on test set...")
        test_data = load_tsv_data(args.test, max_packets=args.max_pkt_len)
        X_test, y_test = extract_combined_features(test_data)
        
        test_pred = rf.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        print(f"Test Accuracy: {test_acc:.4f}")
        
        print("\nTest Classification Report:")
        print(classification_report(y_test, test_pred, digits=4))
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, test_pred)
        print(f"\nConfusion Matrix shape: {cm.shape}")
        
        # 保存混淆矩阵
        np.save(args.output.replace('.pkl', '_confusion.npy'), cm)
    
    # 保存模型
    joblib.dump(rf, args.output)
    print(f"\nModel saved to {args.output}")
    
    # 特征重要性
    print("\nTop 10 Most Important Features:")
    feature_names = (
        [f'traffic_{i}' for i in range(30)] +
        [f'hex_{i}' for i in range(20)]
    )
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")


if __name__ == "__main__":
    main()
