"""
Dual-Tower Multimodal ET-BERT
=============================
双塔多模态 ET-BERT 加密流量分类模型

Usage:
    from dual_tower import DualTowerETBERT, PCAPProcessor, HexTokenizer
"""

from .model import (
    DualTowerETBERT,
    CNNStatEncoder,
    LSTMStatEncoder,
    AttentionFusion
)

from .data import (
    PCAPProcessor,
    HexTokenizer,
    MultimodalDataset,
    load_tsv_data
)

__all__ = [
    'DualTowerETBERT',
    'CNNStatEncoder',
    'LSTMStatEncoder', 
    'AttentionFusion',
    'PCAPProcessor',
    'HexTokenizer',
    'MultimodalDataset',
    'load_tsv_data'
]
