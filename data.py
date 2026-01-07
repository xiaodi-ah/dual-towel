"""
多模态数据处理器 (流级别恶意加密流量识别)
=========================================
从 PCAP 提取:
- 语义特征: TLS/应用层 Hex 序列
- 流量时序矩阵: [N, 3] 每个会话前 N 个数据包

流量时序矩阵 (Traffic Matrix):
==============================
将一个流/会话表示为一张 "流量图像":
┌─────────────────────────────────────┐
│  Packet   │ pkt_len │  IAT  │ dir  │
├───────────┼─────────┼───────┼──────┤
│    1      │  0.42   │ 0.00  │ +1   │
│    2      │  0.03   │ 0.01  │ -1   │
│    3      │  0.85   │ 0.02  │ +1   │
│   ...     │  ...    │  ...  │ ...  │
│    N      │  0.12   │ 0.15  │ -1   │
└─────────────────────────────────────┘

特征说明 (3 维):
0. pkt_len   - 包长 (归一化到 0-1, /1500)
1. iat       - 到达时间间隔 (秒, 截断到 10s)
2. direction - 方向 (+1 上行/客户端发, -1 下行/服务器发)
"""

import numpy as np
from collections import defaultdict
import os

# 流量时序矩阵特征维度
TRAFFIC_FEATURE_DIM = 3   # (pkt_len, iat, direction)

try:
    from scapy.all import rdpcap, TCP, UDP, IP, IPv6
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False
    print("Warning: scapy not installed. PCAP processing unavailable.")

class PCAPProcessor:
    """
    PCAP 特征提取器 - 专为双塔模型设计
    """
    
    def __init__(self, max_packets=100, payload_packets=5, payload_len=128,
                 pkt_len_norm=1500.0, iat_clip=10.0):
        """
        Args:
            max_packets: 统计塔 - 每个流取前 N 个包
            payload_packets: 语义塔 - 提取前几个包的 payload
            payload_len: 语义塔 - 每个包取多少字节 payload
            pkt_len_norm: 包长归一化基准 (MTU)
            iat_clip: IAT 截断阈值 (秒)
        """
        self.max_packets = max_packets
        self.payload_packets = payload_packets
        self.payload_len = payload_len
        self.pkt_len_norm = pkt_len_norm
        self.iat_clip = iat_clip
    
    def extract_hex_bigram(self, payload):
        """
        提取 Hex Bigram (非重叠切分，与 ET-BERT 预训练保持一致)
        输入 bytes: b'\x16\x03\x01...'
        输出 str: "1603 0100 ..."
        """
        if not payload:
            return ""
        
        # 截断
        payload = payload[:self.payload_len]
        
        # 补齐偶数长度 (方便做 bigram)
        if len(payload) % 2 != 0:
            payload += b'\x00'
            
        # Non-overlapping Bigram: 步长为 2
        # 原理: BERT 的 Tokenizer 更容易处理固定的 2字节 词表
        tokens = []
        for i in range(0, len(payload), 2):
            chunk = payload[i:i+2]
            tokens.append(f'{chunk[0]:02x}{chunk[1]:02x}')
            
        return ' '.join(tokens)

    def extract_traffic_matrix(self, packets, first_src_ip):
        """
        提取流量时序矩阵 [N, 3] -> 供 LSTM/CNN 使用
        特征: [归一化包长, Log_IAT, 方向]
        """
        matrix = np.zeros((self.max_packets, TRAFFIC_FEATURE_DIM), dtype=np.float32)
        last_time = 0.0
        
        # 按时间戳排序 (防止乱序)
        packets.sort(key=lambda x: float(x.time))
        
        count = 0
        for i, pkt in enumerate(packets):
            if count >= self.max_packets:
                break
                
            try:
                # 1. 包长归一化
                # 使用 IP 层长度，如果没有 IP 层则用 Wire 长度
                if IP in pkt:
                    plen = pkt[IP].len
                    src = pkt[IP].src
                elif IPv6 in pkt:
                    plen = pkt[IPv6].plen + 40
                    src = pkt[IPv6].src
                else:
                    plen = len(pkt)
                    src = None

                norm_len = min(plen / self.pkt_len_norm, 1.0)
                
                # 2. IAT (对数平滑处理 !!! 重要 !!!)
                cur_time = float(pkt.time)
                if i == 0:
                    norm_iat = 0.0
                    last_time = cur_time
                else:
                    iat = cur_time - last_time
                    if iat < 0: iat = 0 # 修正乱序
                    iat = min(iat, self.iat_clip)
                    # Log 处理: 将 0.0001~10 的范围压缩到 -9 ~ 2.3，利于 LSTM 收敛
                    norm_iat = np.log(iat + 1e-7)
                    last_time = cur_time
                
                # 3. 方向 (+1: Client->Server, -1: Server->Client)
                if src:
                    direction = 1.0 if src == first_src_ip else -1.0
                else:
                    direction = 0.0
                
                matrix[count] = [norm_len, norm_iat, direction]
                count += 1
                
            except Exception:
                continue
                
        return matrix

    def get_flow_key(self, pkt):
        """提取五元组 Key (双向流聚合)"""
        try:
            if IP in pkt:
                src, dst = pkt[IP].src, pkt[IP].dst
                proto = pkt[IP].proto
            elif IPv6 in pkt:
                src, dst = pkt[IPv6].src, pkt[IPv6].dst
                proto = pkt[IPv6].nh
            else:
                return None
            
            if TCP in pkt:
                sport, dport = pkt[TCP].sport, pkt[TCP].dport
            elif UDP in pkt:
                sport, dport = pkt[UDP].sport, pkt[UDP].dport
            else:
                return None
            
            # 统一方向：小的 IP 放前面
            if src < dst:
                return (src, dst, sport, dport, proto)
            else:
                return (dst, src, dport, sport, proto)
        except:
            return None

    def process_pcap(self, pcap_path, use_bigram=True):
        """
        处理单个 PCAP 文件
        Returns: {flow_key: {'semantic_hex': ..., 'traffic_matrix': ...}}
        """
        if not HAS_SCAPY:
            raise RuntimeError("scapy not installed")
        
        try:
            # 使用 rdpcap 读取小文件 (如果文件巨大建议用 PcapReader)
            packets = rdpcap(pcap_path)
        except Exception as e:
            return {}

        # 1. 聚类：将包分到不同的流中
        raw_flows = defaultdict(list)
        for pkt in packets:
            if (TCP in pkt or UDP in pkt) and (IP in pkt or IPv6 in pkt):
                key = self.get_flow_key(pkt)
                if key:
                    raw_flows[key].append(pkt)
        
        results = {}
        for key, pkts in raw_flows.items():
            if not pkts: continue
            
            # 确定 Client IP (假设流的第一个包的发起者是 Client)
            # 注意：这里需要按时间简单排序找第一个
            pkts.sort(key=lambda x: float(x.time))
            first_pkt = pkts[0]
            if IP in first_pkt:
                first_src = first_pkt[IP].src
            elif IPv6 in first_pkt:
                first_src = first_pkt[IPv6].src
            else:
                first_src = None
            
            # A. 提取统计特征 (Traffic Matrix)
            traffic_matrix = self.extract_traffic_matrix(pkts, first_src)
            
            # B. 提取语义特征 (Payload Hex)
            hex_parts = []
            payload_count = 0
            for pkt in pkts:
                if payload_count >= self.payload_packets:
                    break
                
                # 提取应用层负载
                layer = pkt[TCP].payload if TCP in pkt else pkt[UDP].payload
                content = bytes(layer)
                
                if not content:
                    continue # 跳过空包 (纯ACK)
                    
                if use_bigram:
                    hex_str = self.extract_hex_bigram(content)
                else:
                    # 只有单字节时
                    hex_str = ' '.join(f'{b:02x}' for b in content[:self.payload_len])
                
                if hex_str:
                    hex_parts.append(hex_str)
                    payload_count += 1
            
            # 如果没有 payload (只有握手没数据)，填个占位符
            final_hex = ' '.join(hex_parts) if hex_parts else "0000"
            
            results[key] = {
                'semantic_hex': final_hex,
                'traffic_matrix': traffic_matrix
            }
            
        return results

# ==========================================
# 工具函数：用于 Dataset DataLoader
# ==========================================

def load_tsv_data(tsv_path, max_packets=100):
    """
    加载 TSV 数据 (适配 preprocess.py 的输出)
    """
    data = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) < 3: continue # 必须有 label, hex, traffic
            
            label = int(parts[0])
            hex_seq = parts[1]
            traffic_str = parts[2]
            
            # 解析 Matrix
            matrix = np.zeros((max_packets, TRAFFIC_FEATURE_DIM), dtype=np.float32)
            
            # 格式: len,iat,dir;len,iat,dir...
            if traffic_str and traffic_str != "0,0,0":
                pkts_str = traffic_str.split(';')
                for i, p_str in enumerate(pkts_str[:max_packets]):
                    vals = p_str.split(',')
                    if len(vals) == 3:
                        matrix[i, 0] = float(vals[0]) # len
                        matrix[i, 1] = float(vals[1]) # iat
                        matrix[i, 2] = float(vals[2]) # dir
            
            data.append({
                'label': label,
                'hex': hex_seq,
                'traffic_matrix': matrix
            })
    return data


class HexTokenizer:
    """
    Hex 序列分词器
    
    输入格式: "1603 0301 0100 00f1" (bigram，与 ET-BERT 预训练一致)
    输出: token 列表 ["1603", "0301", "0100", "00f1"]
    """
    
    def __init__(self, max_len=512, bigram=False):
        """
        Args:
            max_len: 最大 token 数
            bigram: 是否对输入做额外 bigram 处理
                    - False (默认): 输入已是 bigram 格式，直接 split
                    - True: 输入是单字节 "16 03 01"，需要做 bigram
        """
        self.max_len = max_len
        self.bigram = bigram
    
    def tokenize(self, hex_string):
        """将 Hex 字符串转为 token 列表"""
        if not hex_string or not hex_string.strip():
            return ["0000"] * self.max_len
        
        tokens = hex_string.split()
        
        # 如果输入是原始 hex (单字节)，需要做 bigram
        if self.bigram and len(tokens) > 1 and len(tokens[0]) == 2:
            bigram_tokens = []
            for i in range(len(tokens) - 1):
                bigram_tokens.append(tokens[i] + tokens[i+1])
            tokens = bigram_tokens
        
        # 截断/填充
        tokens = tokens[:self.max_len]
        tokens += ["0000"] * (self.max_len - len(tokens))
        return tokens


class MultimodalDataset:
    """
    多模态数据集 (双塔模型)
    
    输入 data: list of {'label': int, 'hex': str, 'traffic_matrix': np.array}
    输出: {'sem_ids': ..., 'seg': ..., 'traffic_matrix': ..., 'label': ...}
    """
    
    def __init__(self, data, tokenizer, vocab, max_sem_len=512, max_pkt_len=100):
        """
        Args:
            data: load_tsv_data() 返回的数据列表
            tokenizer: HexTokenizer 实例
            vocab: UER 的 Vocab 对象 (支持 vocab.get(token, default))
            max_sem_len: 语义序列最大长度
            max_pkt_len: 流量时序矩阵最大长度 (包数)
        """
        self.data = data
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_sem_len = max_sem_len
        self.max_pkt_len = max_pkt_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # ====== 语义特征 (Hex Bigram → Token IDs) ======
        tokens = self.tokenizer.tokenize(item['hex'])
        
        # 转换为 ID (使用 vocab.w2i 字典)
        # UER 的 Vocab 对象: vocab.w2i 是 token → id 映射
        w2i = self.vocab.w2i
        pad_id = w2i.get("[PAD]", 0)
        unk_id = w2i.get("[UNK]", 1)
        
        sem_ids = []
        for t in tokens[:self.max_sem_len]:
            # 优先查找 token，找不到用 UNK
            tid = w2i.get(t, unk_id)
            sem_ids.append(tid)
        
        # 填充到 max_sem_len
        while len(sem_ids) < self.max_sem_len:
            sem_ids.append(pad_id)
        
        # ====== 流量时序矩阵 [N, 3] ======
        traffic_matrix = item.get('traffic_matrix', 
                          np.zeros((self.max_pkt_len, TRAFFIC_FEATURE_DIM), dtype=np.float32))
        
        # 确保形状正确
        if traffic_matrix.shape[0] < self.max_pkt_len:
            pad = np.zeros((self.max_pkt_len - traffic_matrix.shape[0], TRAFFIC_FEATURE_DIM), 
                           dtype=np.float32)
            traffic_matrix = np.vstack([traffic_matrix, pad])
        traffic_matrix = traffic_matrix[:self.max_pkt_len].astype(np.float32)
        
        return {
            'sem_ids': np.array(sem_ids[:self.max_sem_len], dtype=np.int64),
            'seg': np.ones(self.max_sem_len, dtype=np.int64),  # segment ids (全1)
            'traffic_matrix': traffic_matrix,
            'label': item.get('label', 0)
        }


def visualize_traffic_matrix(traffic_matrix, title="Traffic Matrix"):
    """
    可视化流量时序矩阵 (用于调试/分析)
    
    Args:
        traffic_matrix: [N, 3] 流量时序矩阵
        title: 图表标题
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        # 找到有效包数量 (非零行)
        valid_mask = (traffic_matrix[:, 0] != 0) | (traffic_matrix[:, 2] != 0)
        valid_len = valid_mask.sum()
        if valid_len == 0:
            valid_len = 1
        
        feature_names = ['Packet Length (normalized)', 'IAT (log scale)', 'Direction']
        colors = ['blue', 'green', 'red']
        
        for i, (ax, name, color) in enumerate(zip(axes, feature_names, colors)):
            if i == 2:  # Direction 用柱状图更直观
                ax.bar(range(valid_len), traffic_matrix[:valid_len, i], 
                       color=[('blue' if d > 0 else 'red') for d in traffic_matrix[:valid_len, i]],
                       alpha=0.7)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax.set_ylabel(name)
                ax.set_ylim(-1.5, 1.5)
            else:
                ax.plot(range(valid_len), traffic_matrix[:valid_len, i], 
                        color=color, linewidth=1, marker='o', markersize=2)
                ax.fill_between(range(valid_len), traffic_matrix[:valid_len, i], 
                               alpha=0.3, color=color)
                ax.set_ylabel(name)
        
        axes[-1].set_xlabel('Packet Index')
        axes[0].set_title(f'{title} (N={valid_len} packets)')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib not installed, visualization unavailable")