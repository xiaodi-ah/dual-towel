#!/usr/bin/env python3
"""
双塔模型可视化脚本
==================
1. 混淆矩阵热力图
2. 各类别 F1 对比柱状图
3. t-SNE 特征可视化
4. 训练曲线
5. 模型对比柱状图
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not installed")

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False


def plot_confusion_matrix(cm, labels=None, title="Confusion Matrix", save_path=None, figsize=(14, 12)):
    """
    绘制混淆矩阵热力图
    
    Args:
        cm: 混淆矩阵 [n_classes, n_classes]
        labels: 类别标签列表
        title: 图表标题
        save_path: 保存路径
    """
    if not HAS_MPL:
        print("matplotlib not installed")
        return
    
    n_classes = cm.shape[0]
    if labels is None:
        labels = [str(i) for i in range(n_classes)]
    
    # 归一化 (按行)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if HAS_SNS:
        sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    vmin=0, vmax=1)
    else:
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        plt.colorbar(im)
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_class_f1_comparison(results_dict, labels=None, title="Per-Class F1 Score Comparison", save_path=None):
    """
    绘制各类别 F1 对比柱状图
    
    Args:
        results_dict: {model_name: {'f1_scores': [f1_0, f1_1, ...]}}
        labels: 类别标签
    """
    if not HAS_MPL:
        return
    
    models = list(results_dict.keys())
    n_models = len(models)
    n_classes = len(results_dict[models[0]]['f1_scores'])
    
    if labels is None:
        labels = [str(i) for i in range(n_classes)]
    
    x = np.arange(n_classes)
    width = 0.8 / n_models
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    
    for i, (model, data) in enumerate(results_dict.items()):
        f1_scores = data['f1_scores']
        offset = (i - n_models/2 + 0.5) * width
        ax.bar(x + offset, f1_scores, width, label=model, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='0.9 baseline')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_model_comparison(results, title="Model Comparison", save_path=None):
    """
    绘制模型对比柱状图
    
    Args:
        results: {model_name: accuracy} 或 {model_name: {'acc': ..., 'f1': ...}}
    """
    if not HAS_MPL:
        return
    
    models = list(results.keys())
    
    # 处理不同格式
    if isinstance(list(results.values())[0], dict):
        accs = [results[m].get('acc', results[m].get('accuracy', 0)) for m in models]
    else:
        accs = [results[m] for m in models]
    
    # 颜色：双塔最高用绿色
    colors = ['#4CAF50' if a == max(accs) else '#2196F3' for a in accs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(models, accs, color=colors, alpha=0.8, edgecolor='black')
    
    # 添加数值标签
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90% baseline')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_ablation_results(results, title="Ablation Study", save_path=None):
    """
    消融实验结果柱状图
    
    Args:
        results: {experiment_name: accuracy}
    """
    if not HAS_MPL:
        return
    
    # 排序：按准确率从高到低
    sorted_items = sorted(results.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_items]
    accs = [x[1] for x in sorted_items]
    
    # 颜色渐变
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(names)))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.barh(names, accs, color=colors, alpha=0.8, edgecolor='black')
    
    # 添加数值标签
    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc:.2%}', ha='left', va='center', fontsize=10)
    
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 1.15)
    ax.axvline(x=0.9, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_training_curve(history, title="Training Curve", save_path=None):
    """
    绘制训练曲线
    
    Args:
        history: {'train_loss': [...], 'valid_loss': [...], 'train_acc': [...], 'valid_acc': [...]}
    """
    if not HAS_MPL:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Loss 曲线
    if 'train_loss' in history:
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', marker='o')
    if 'valid_loss' in history:
        ax1.plot(epochs, history['valid_loss'], 'r-', label='Valid Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy 曲线
    if 'train_acc' in history:
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', marker='o')
    if 'valid_acc' in history:
        ax2.plot(epochs, history['valid_acc'], 'r-', label='Valid Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def calculate_f1_from_confusion(cm):
    """从混淆矩阵计算每类 F1"""
    n_classes = cm.shape[0]
    f1_scores = []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    return f1_scores


def main():
    parser = argparse.ArgumentParser(description="Visualization Tools")
    parser.add_argument("--results", type=str, help="推理结果 JSON 文件 (predictions.json)")
    parser.add_argument("--confusion", type=str, help="混淆矩阵 NPY 文件")
    parser.add_argument("--output_dir", type=str, default="figures", help="输出目录")
    parser.add_argument("--labels", type=str, help="标签映射 JSON")
    
    # 快速绘制消融实验结果
    parser.add_argument("--ablation", action="store_true", help="绘制消融实验结果")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载标签映射
    label_map = None
    if args.labels:
        with open(args.labels) as f:
            label_map = json.load(f)
    
    # ========== 从推理结果绘图 ==========
    if args.results:
        with open(args.results) as f:
            results = json.load(f)
        
        # 混淆矩阵
        if 'confusion_matrix' in results:
            cm = np.array(results['confusion_matrix'])
            labels = [label_map.get(str(i), f"C{i}") for i in range(cm.shape[0])] if label_map else None
            plot_confusion_matrix(cm, labels, 
                                  title="Dual-Tower ET-BERT Confusion Matrix",
                                  save_path=os.path.join(args.output_dir, "confusion_matrix.png"))
    
    # ========== 单独的混淆矩阵文件 ==========
    if args.confusion:
        cm = np.load(args.confusion)
        labels = [label_map.get(str(i), f"C{i}") for i in range(cm.shape[0])] if label_map else None
        plot_confusion_matrix(cm, labels,
                              save_path=os.path.join(args.output_dir, "confusion_matrix.png"))
    
    # ========== 消融实验快速绘图 ==========
    if args.ablation:
        # 硬编码的实验结果 (可以修改)
        ablation_results = {
            'Dual-Tower (Attention)': 0.9641,
            'Dual-Tower (Concat)': 0.9651,
            'Statistical Tower Only': 0.8286,
            'Semantic Tower (ET-BERT)': 0.7386,
        }
        
        plot_ablation_results(ablation_results, 
                              title="Ablation Study: Model Architecture",
                              save_path=os.path.join(args.output_dir, "ablation_model.png"))
        
        # 特征消融 (需要你补充实验数据)
        feature_ablation = {
            'All Features (pkt_len+iat+dir)': 0.8286,
            # 'pkt_len only': 0.xxx,
            # 'iat only': 0.xxx,
            # 'dir only': 0.xxx,
            # 'pkt_len + dir': 0.xxx,
        }
        
        if len(feature_ablation) > 1:
            plot_ablation_results(feature_ablation,
                                  title="Ablation Study: Traffic Features",
                                  save_path=os.path.join(args.output_dir, "ablation_features.png"))
        
        print("\nTo add more results, edit the ablation_results dict in visualize.py")


if __name__ == "__main__":
    main()
