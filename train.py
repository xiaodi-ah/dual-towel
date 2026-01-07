"""
双塔多模态 ET-BERT 训练脚本
===========================
支持两阶段训练策略:
  阶段1: 冻结语义塔，只训练统计塔+融合层+分类器
  阶段2: 解冻语义塔，联合微调全部参数
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed
from uer.layers import str2embedding
from uer.encoders import str2encoder

from model import DualTowerETBERT
from data import HexTokenizer, MultimodalDataset, load_tsv_data


class SimpleDataset(Dataset):
    def __init__(self, data, tokenizer, vocab, max_sem_len=512, max_pkt_len=100):
        self.ds = MultimodalDataset(data, tokenizer, vocab, max_sem_len, max_pkt_len)
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        return self.ds[idx]


def collate_fn(batch):
    return {
        'sem_ids': torch.tensor(np.stack([b['sem_ids'] for b in batch]), dtype=torch.long),
        'seg': torch.tensor(np.stack([b['seg'] for b in batch]), dtype=torch.long),
        'traffic_matrix': torch.tensor(np.stack([b['traffic_matrix'] for b in batch]), dtype=torch.float32),
        'labels': torch.tensor([b['label'] for b in batch], dtype=torch.long)
    }


def train_epoch(model, loader, optimizer, device, scaler=None, accum_steps=1):
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()
    
    for step, batch in enumerate(tqdm(loader, desc="Training")):
        sem_ids = batch['sem_ids'].to(device)
        seg = batch['seg'].to(device)
        traffic_matrix = batch['traffic_matrix'].to(device)
        labels = batch['labels'].to(device)
        
        # 混合精度前向
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss, logits = model(sem_ids, traffic_matrix, seg, labels)
                loss = loss / accum_steps  # 梯度累积需要缩放 loss
            scaler.scale(loss).backward()
        else:
            loss, logits = model(sem_ids, traffic_matrix, seg, labels)
            loss = loss / accum_steps
            loss.backward()
        
        # 梯度累积: 每 accum_steps 步更新一次
        if (step + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device, use_amp=False):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    for batch in tqdm(loader, desc="Evaluating"):
        sem_ids = batch['sem_ids'].to(device)
        seg = batch['seg'].to(device)
        traffic_matrix = batch['traffic_matrix'].to(device)
        labels = batch['labels'].to(device)
        
        if use_amp:
            with torch.cuda.amp.autocast():
                loss, logits = model(sem_ids, traffic_matrix, seg, labels)
        else:
            loss, logits = model(sem_ids, traffic_matrix, seg, labels)
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser(description="Train Dual-Tower ET-BERT")
    
    # 路径参数
    parser.add_argument("--pretrained", type=str, default=None, help="预训练模型路径")
    parser.add_argument("--config", type=str, default="../models/bert_base_config.json")
    parser.add_argument("--vocab", type=str, default="../models/encryptd_vocab.txt")
    parser.add_argument("--train", type=str, required=True, help="训练数据 TSV")
    parser.add_argument("--valid", type=str, default=None, help="验证数据 TSV")
    parser.add_argument("--output", type=str, default="dual_tower_model.bin")
    
    # 模型参数
    parser.add_argument("--labels_num", type=int, default=10)
    parser.add_argument("--fusion", choices=['attention', 'concat'], default='attention')
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8, help="实际 batch size (建议 4-8 以节省显存)")
    parser.add_argument("--accum_steps", type=int, default=4, help="梯度累积步数 (有效 batch = batch_size * accum_steps)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", help="启用 FP16 混合精度训练 (大幅节省显存)")
    
    # 两阶段训练
    parser.add_argument("--two_stage", action="store_true", help="启用两阶段训练")
    parser.add_argument("--stage1_epochs", type=int, default=3, help="阶段1 epochs")
    parser.add_argument("--stage1_lr", type=float, default=1e-3, help="阶段1学习率")
    
    # 序列长度
    parser.add_argument("--max_sem_len", type=int, default=512)
    parser.add_argument("--max_pkt_len", type=int, default=100, help="流量时序矩阵长度 (前N个包)")
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
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
    print(f"Vocab size: {len(vocab)}")
    
    # 加载数据
    # bigram=False 因为 TSV 中的 hex 已经是 bigram 格式 (与原始 ET-BERT 一致)
    tokenizer = HexTokenizer(args.max_sem_len, bigram=False)
    train_data = load_tsv_data(args.train, max_packets=args.max_pkt_len)
    train_ds = SimpleDataset(train_data, tokenizer, vocab, args.max_sem_len, args.max_pkt_len)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate_fn)
    print(f"Train samples: {len(train_ds)}")
    
    valid_loader = None
    if args.valid:
        valid_data = load_tsv_data(args.valid, max_packets=args.max_pkt_len)
        valid_ds = SimpleDataset(valid_data, tokenizer, vocab, args.max_sem_len, args.max_pkt_len)
        valid_loader = DataLoader(valid_ds, args.batch_size, collate_fn=collate_fn)
        print(f"Valid samples: {len(valid_ds)}")
    
    # 构建模型
    # 使用 word_pos_seg embedding (与原始 ET-BERT 一致)
    # 需要设置 max_seq_length 和 remove_embedding_layernorm
    cfg_obj.max_seq_length = args.max_sem_len
    cfg_obj.remove_embedding_layernorm = False
    cfg_obj.emb_size = cfg_obj.hidden_size  # embedding 维度 = hidden 维度
    
    # TransformerEncoder 需要的额外配置
    cfg_obj.mask = 'fully_visible'  # BERT 使用全可见 mask
    cfg_obj.parameter_sharing = False  # 不共享参数
    cfg_obj.factorized_embedding_parameterization = False  # 不使用因式分解
    cfg_obj.layernorm_positioning = 'post'  # LayerNorm 在后 (原始 BERT)
    cfg_obj.relative_position_embedding = False  # 不使用相对位置编码
    cfg_obj.remove_transformer_bias = False  # 保留 bias
    cfg_obj.layernorm = 'normal'  # 普通 LayerNorm
    cfg_obj.relative_attention_buckets_num = 32  # 相对位置桶数 (虽然不用)
    
    # TransformerLayer 需要的额外配置
    cfg_obj.remove_attention_scale = False  # 保留 attention scale (1/sqrt(d_k))
    cfg_obj.feed_forward = 'dense'  # 标准 FFN (不是 gated)
    
    sem_emb = str2embedding['word_pos_seg'](cfg_obj, len(vocab))
    sem_enc = str2encoder['transformer'](cfg_obj)
    
    model = DualTowerETBERT(
        cfg_obj, sem_emb, sem_enc,
        fusion=args.fusion,
        labels_num=args.labels_num,
        seq_len=args.max_pkt_len
    ).to(device)
    
    # 加载预训练权重
    if args.pretrained:
        print(f"Loading pretrained: {args.pretrained}")
        state = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(state, strict=False)
        print("Pretrained weights loaded (semantic tower)")
    
    # ========== 混合精度 scaler ==========
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    if args.fp16:
        print(f"FP16 mixed precision enabled")
    print(f"Effective batch size: {args.batch_size} x {args.accum_steps} = {args.batch_size * args.accum_steps}")
    
    # ========== 两阶段训练 ==========
    if args.two_stage:
        print("\n========== Stage 1: Train statistical tower ==========")
        model.freeze_semantic(True)
        
        # 只优化非冻结参数
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt1 = torch.optim.AdamW(trainable, lr=args.stage1_lr)
        
        for ep in range(args.stage1_epochs):
            loss, acc = train_epoch(model, train_loader, opt1, device, scaler, args.accum_steps)
            print(f"[Stage1] Epoch {ep+1}: Loss={loss:.4f}, Acc={acc:.4f}")
            if valid_loader:
                v_loss, v_acc = evaluate(model, valid_loader, device, args.fp16)
                print(f"         Valid: Loss={v_loss:.4f}, Acc={v_acc:.4f}")
        
        print("\n========== Stage 2: Joint fine-tuning ==========")
        model.freeze_semantic(False)
    
    # ========== 主训练循环 ==========
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_acc = 0
    
    for ep in range(args.epochs):
        loss, acc = train_epoch(model, train_loader, optimizer, device, scaler, args.accum_steps)
        print(f"Epoch {ep+1}: Loss={loss:.4f}, Acc={acc:.4f}")
        
        if valid_loader:
            v_loss, v_acc = evaluate(model, valid_loader, device, args.fp16)
            print(f"  Valid: Loss={v_loss:.4f}, Acc={v_acc:.4f}")
            
            if v_acc > best_acc:
                best_acc = v_acc
                torch.save(model.state_dict(), args.output)
                print(f"   Saved best model (acc={v_acc:.4f})")
    
    if not valid_loader:
        torch.save(model.state_dict(), args.output)
    
    print(f"\nTraining completed! Model saved to {args.output}")


if __name__ == "__main__":
    main()
