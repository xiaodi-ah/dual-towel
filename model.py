"""
双塔多模态 ET-BERT 模型 (流级别恶意流量检测)
==========================================
- 语义塔: BERT/DistilBERT 处理 Hex 序列
- 统计塔: 
  - 输入: 流量时序矩阵 [N, 3] (Packet Length, IAT, Direction)
  - 1D-CNN: 提取局部纹理/模式特征
  - Bi-LSTM: 提取时序依赖关系
- 融合: 注意力融合或简单拼接

流量时序矩阵概念:
================
将一个会话的前 N 个数据包表示为一个 [N, 3] 的矩阵，
类似于一张 "流量图像"，其中:
- 行: 时间维度 (第 1, 2, ..., N 个包)
- 列: 特征维度 (包长, IAT, 方向)

这种表示可以捕获:
- CNN: 局部模式/纹理 (如突发模式、固定大小心跳)
- LSTM: 长程时序依赖 (如请求-响应序列、周期性beacon)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============== 特征维度定义 ==============

# 包级别时序特征 (3 维) - 流量时序矩阵
TRAFFIC_FEATURES = {
    'pkt_len': 0,      # 包长 (归一化到 0-1)
    'iat': 1,          # 到达时间间隔 (归一化)
    'direction': 2,    # 方向 (+1 上行, -1 下行)
}
TRAFFIC_FEATURE_DIM = len(TRAFFIC_FEATURES)  # 3 维

# 默认序列长度
DEFAULT_SEQ_LEN = 100  # 前 N 个包


class TrafficImageEncoder(nn.Module):
    """
    流量时序矩阵编码器: 1D-CNN + Bi-LSTM
    
    输入: [B, N, 3] 流量时序矩阵 (N个包, 每包3个特征)
    
    架构:
    ┌──────────────────────────────────────────────┐
    │  输入: 流量时序矩阵 [B, N, 3]                 │
    │  (Packet Length, IAT, Direction)            │
    └─────────────────┬────────────────────────────┘
                      ▼
    ┌──────────────────────────────────────────────┐
    │  1D-CNN (多尺度卷积)                         │
    │  - 提取局部纹理/模式特征                     │
    │  - kernel_sizes: [3, 5, 7, 9]               │
    │  - 捕获: 突发模式、心跳包、握手序列          │
    └─────────────────┬────────────────────────────┘
                      ▼
    ┌──────────────────────────────────────────────┐
    │  Bi-LSTM                                     │
    │  - 提取长程时序依赖                          │
    │  - 捕获: 请求-响应、周期性beacon             │
    └─────────────────┬────────────────────────────┘
                      ▼
    ┌──────────────────────────────────────────────┐
    │  输出: [B, N, hidden_size]                   │
    └──────────────────────────────────────────────┘
    """
    
    def __init__(self, hidden_size, feature_dim=TRAFFIC_FEATURE_DIM, 
                 cnn_filters=64, kernel_sizes=[3, 5, 7, 9],
                 lstm_layers=2, dropout=0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # ====== 1D-CNN: 多尺度卷积提取局部模式 ======
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feature_dim, cnn_filters, k, padding=k//2),
                nn.BatchNorm1d(cnn_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for k in kernel_sizes
        ])
        
        cnn_output_dim = cnn_filters * len(kernel_sizes)
        
        # CNN 输出投影
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_output_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ====== Bi-LSTM: 时序依赖建模 ======
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # 最终投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, N, 3] 流量时序矩阵
               - N: 序列长度 (包数量)
               - 3: (pkt_len, iat, direction)
        Returns:
            [B, N, hidden_size] 编码后的序列表示
        """
        B, N, D = x.shape
        
        # ====== 1D-CNN ======
        # [B, N, D] -> [B, D, N] for Conv1d
        x_cnn = x.transpose(1, 2)
        
        # 多尺度卷积
        conv_outs = [conv(x_cnn) for conv in self.convs]  # 每个 [B, cnn_filters, N]
        x_cnn = torch.cat(conv_outs, dim=1)  # [B, cnn_filters * num_kernels, N]
        
        # [B, C, N] -> [B, N, C]
        x_cnn = x_cnn.transpose(1, 2)
        
        # 投影到 hidden_size
        x_cnn = self.cnn_proj(x_cnn)  # [B, N, hidden_size]
        
        # ====== Bi-LSTM ======
        x_lstm, _ = self.lstm(x_cnn)  # [B, N, hidden_size]
        
        # 残差连接
        output = self.output_proj(x_lstm) + x_cnn
        
        return output


class TrafficImagePooling(nn.Module):
    """流量时序矩阵池化层 - 从序列表示生成流级别表示"""
    
    def __init__(self, hidden_size, pooling='attention'):
        super().__init__()
        self.pooling = pooling
        
        if pooling == 'attention':
            # 注意力池化
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1)
            )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, N, hidden_size]
            mask: [B, N] 有效位置掩码 (可选)
        Returns:
            [B, hidden_size]
        """
        if self.pooling == 'attention':
            # 注意力加权池化
            attn_scores = self.attention(x).squeeze(-1)  # [B, N]
            if mask is not None:
                attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1)  # [B, N]
            pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # [B, hidden_size]
        elif self.pooling == 'max':
            pooled = x.max(dim=1)[0]
        else:  # mean
            pooled = x.mean(dim=1)
        
        return pooled


class AttentionFusion(nn.Module):
    """
    多模态融合模块
    
    'gated': 门控融合 (向量级别，用于池化后的特征)
    """
    
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, mode='gated'):
        super().__init__()
        self.mode = mode
        # 门控融合 (向量级别)
        self.fc_sem = nn.Linear(hidden_size, hidden_size)
        self.fc_stat = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, sem, stat):
        """
        Args:
            sem: 语义特征 [B, hidden] 或 [B, L, hidden]
            stat: 统计特征 [B, hidden] 或 [B, N, hidden]
        """
        # 向量级别门控融合
        sem_proj = self.fc_sem(sem)
        stat_proj = self.fc_stat(stat)
        gate = self.gate(torch.cat([sem_proj, stat_proj], dim=-1))
        fused = gate * sem_proj + (1 - gate) * stat_proj
        # 也拼接原始特征增强表达
        output = self.output(torch.cat([fused, sem + stat], dim=-1))
        return output


class DualTowerETBERT(nn.Module):
    """
    双塔多模态 ET-BERT (流级别恶意流量检测)
    
    关键设计: 语义塔和统计塔有不同的序列长度和语义
    =========================================================
    - 语义塔: [B, L_sem] → [B, L_sem, hidden] → 池化 → [B, hidden]
      - L_sem: token 级别 (通常 512)
      - 一个 token = 一个 hex byte pair
      
    - 统计塔: [B, N, 3] → [B, N, hidden] → 池化 → [B, hidden]
      - N: packet 级别 (通常 100)
      - 一个时间步 = 一个数据包
    
    两者语义不对应，所以各自池化后再融合！
    
    架构:
    ┌─────────────────┐     ┌──────────────────────────────┐
    │    语义塔       │     │          统计塔               │
    │  (BERT/Trans)  │     │                              │
    │                │     │  流量时序矩阵 [B, N, 3]       │
    │  Hex序列       │     │  (pkt_len, iat, direction)   │
    │  [B, L_sem]    │     │           │                  │
    │      │         │     │           ▼                  │
    │      ▼         │     │  ┌─────────────────────┐    │
    │  Embedding     │     │  │ 1D-CNN (多尺度卷积)  │    │
    │      │         │     │  │ 提取局部纹理/模式   │    │
    │      ▼         │     │  └──────────┬──────────┘    │
    │  Transformer   │     │             ▼                │
    │   Encoder      │     │  ┌─────────────────────┐    │
    │      │         │     │  │ Bi-LSTM             │    │
    │      ▼         │     │  │ 提取时序依赖        │    │
    │  [B,L_sem,H]   │     │  └──────────┬──────────┘    │
    │      │         │     │             │                │
    │      ▼         │     │             ▼                │
    │  注意力池化    │     │       注意力池化             │
    │  [B, hidden]   │     │       [B, hidden]           │
    └──────┬─────────┘     └─────────────┬───────────────┘
           │                             │
           └──────────┬──────────────────┘
                      ▼
              ┌───────────────┐
              │  门控融合      │
              │  [B, hidden]  │
              └───────┬───────┘
                      ▼
              ┌───────────────┐
              │   分类头       │
              │  → 恶意/正常   │
              └───────────────┘
    
    Args:
        args: 配置参数 (需包含 hidden_size, heads_num)
        semantic_embedding: 语义嵌入层
        semantic_encoder: 语义编码器 (BERT/Transformer)
        fusion: 'attention' (门控融合) 或 'concat' (简单拼接)
        labels_num: 分类数
        seq_len: 流量时序矩阵长度 (前 N 个包)
    """
    
    def __init__(self, args, semantic_embedding, semantic_encoder, 
                 fusion='attention', labels_num=10, seq_len=DEFAULT_SEQ_LEN):
        super().__init__()
        
        # 使用与原始 ET-BERT 相同的属性名，以便正确加载预训练权重
        # 预训练权重的 key: embedding.*, encoder.*
        self.embedding = semantic_embedding
        self.encoder = semantic_encoder
        hidden = args.hidden_size
        dropout = getattr(args, 'dropout', 0.1)
        self.seq_len = seq_len
        
        # ====== 统计塔: 流量时序矩阵编码器 (1D-CNN + LSTM) ======
        self.traffic_encoder = TrafficImageEncoder(
            hidden_size=hidden,
            feature_dim=TRAFFIC_FEATURE_DIM,
            cnn_filters=64,
            kernel_sizes=[3, 5, 7, 9],
            lstm_layers=2,
            dropout=dropout
        )
        
        # ====== 池化层 (各自独立) ======
        self.sem_pooling = TrafficImagePooling(hidden, pooling='attention')
        self.stat_pooling = TrafficImagePooling(hidden, pooling='attention')
        
        # ====== 融合模块 (向量级别) ======
        if fusion == 'attention':
            self.fusion = AttentionFusion(hidden, getattr(args, 'heads_num', 8), dropout, mode='gated')
            self.fusion_type = 'attention'
        else:
            self.fusion = nn.Sequential(
                nn.Linear(hidden * 2, hidden), 
                nn.LayerNorm(hidden),
                nn.ReLU(), 
                nn.Dropout(dropout)
            )
            self.fusion_type = 'concat'
        
        # ====== 分类头 ======
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, labels_num)
        )
        self.labels_num = labels_num
        
    def forward(self, sem_ids, traffic_matrix, seg=None, labels=None):
        """
        Args:
            sem_ids: [B, L_sem] 语义 token IDs (token 级别，通常 512)
            traffic_matrix: [B, N, 3] 流量时序矩阵 (packet 级别，通常 100)
                - N: 前 N 个数据包
                - 3: (pkt_len, iat, direction)
            seg: [B, L_sem] 段落标识
            labels: [B] 标签 (可选，用于计算loss)
        Returns:
            loss, logits
        """
        # ====== 语义塔 (ET-BERT) ======
        sem = self.embedding(sem_ids, seg)
        sem_seq = self.encoder(sem, seg)  # [B, L_sem, hidden]
        
        # 语义塔池化
        sem_pooled = self.sem_pooling(sem_seq)  # [B, hidden]
        
        # ====== 统计塔: 流量时序矩阵编码 ======
        traffic_enc = self.traffic_encoder(traffic_matrix)  # [B, N, hidden]
        
        # 统计塔池化 (独立)
        stat_pooled = self.stat_pooling(traffic_enc)  # [B, hidden]
        
        # ====== 融合 (向量级别) ======
        if self.fusion_type == 'attention':
            fused = self.fusion(sem_pooled, stat_pooled)  # [B, hidden]
        else:
            fused = self.fusion(torch.cat([sem_pooled, stat_pooled], dim=-1))  # [B, hidden]
        
        # ====== 分类 ======
        logits = self.classifier(fused)
        
        # 计算loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return loss, logits
    
    def freeze_semantic(self, freeze=True):
        """冻结/解冻语义塔 (用于两阶段训练)"""
        for p in self.embedding.parameters():
            p.requires_grad = not freeze
        for p in self.encoder.parameters():
            p.requires_grad = not freeze
