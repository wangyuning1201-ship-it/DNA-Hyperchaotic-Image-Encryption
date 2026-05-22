import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing

# 设置多进程启动方法为'spawn'
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 常量定义 - 增强参数
NUM_PARTICLES = 150 if torch.cuda.is_available() else 80  # 增加粒子数
NUM_ITERATIONS = 200 if torch.cuda.is_available() else 120  # 增加迭代次数
DIMENSION = 256
MATRIX_SIZE = 16
C1 = 0.6
C2 = 0.7
W_MAX = 0.9
W_MIN = 0.2
MAX_VELOCITY = DIMENSION // 4
N = 8
MUTATION_RATE = 0.25  # 提高变异率
DL_THRESHOLD = 0.75
ELITE_SIZE = 10 if torch.cuda.is_available() else 6
LOCAL_SEARCH_INTENSITY = 8 if torch.cuda.is_available() else 5  # 增强局部搜索

# 适应度函数的权重因子 - 优化权重分配
a_s = 25.0  # 非线性度权重 (微降)
a_d = 100.0  # 差分均匀度权重 (大幅提升)
a_a = 8.0  # 代数次数权重 (保持)
a_sac = 15.0  # SAC权重 (大幅提升)
a_bic = 15.0  # BIC权重 (大幅提升)

# 自定义S盒定义 (保持不变)
CUSTOM_S_BOX = [
    0x1e, 0x2b, 0xd9, 0x88, 0xa0, 0xe9, 0x7b, 0xb3, 0x6f, 0x5d, 0xb8, 0x72, 0xf1, 0x04, 0x95, 0x6b,
    0xfb, 0x15, 0xe2, 0x83, 0x63, 0x19, 0x06, 0x30, 0xb4, 0xba, 0x3d, 0x07, 0x86, 0xbf, 0xf9, 0x6d,
    0xb1, 0x17, 0xc6, 0x69, 0x4e, 0xb7, 0x8d, 0x59, 0xfd, 0x44, 0xc0, 0x5a, 0x3c, 0x31, 0x27, 0xf0,
    0x65, 0xe1, 0x62, 0x3f, 0xd2, 0x4d, 0xcf, 0x78, 0x7c, 0x81, 0x93, 0xa2, 0xb0, 0x0b, 0xfa, 0x3b,
    0x94, 0x25, 0xc7, 0x09, 0x5c, 0x22, 0xf8, 0x35, 0x18, 0x3e, 0x97, 0xee, 0x8a, 0x52, 0xe0, 0x1a,
    0xb2, 0x2c, 0x1d, 0xce, 0x5f, 0x84, 0x12, 0x80, 0x21, 0xea, 0xd4, 0xec, 0xdf, 0xa9, 0x47, 0xaa,
    0xfe, 0xdb, 0xbc, 0x9c, 0x0e, 0xb6, 0xd3, 0x1f, 0x56, 0xa6, 0x6a, 0xf4, 0xab, 0x74, 0x03, 0xfc,
    0x01, 0xc1, 0x8c, 0x0d, 0x85, 0x7e, 0x6e, 0x23, 0x67, 0x8b, 0xc9, 0xad, 0x42, 0xa7, 0xd1, 0x9e,
    0x75, 0x46, 0xde, 0x77, 0xaf, 0xae, 0xc8, 0x51, 0x11, 0x92, 0x2e, 0xe8, 0x43, 0x57, 0xd6, 0xe6,
    0x33, 0xb5, 0x20, 0x73, 0x87, 0x37, 0x48, 0x9d, 0x7a, 0x2d, 0x16, 0x0a, 0x4f, 0x0f, 0x32, 0xbb,
    0x66, 0x53, 0x29, 0xd8, 0xc2, 0xe5, 0x9a, 0x0c, 0xe3, 0x39, 0x7d, 0x13, 0x36, 0xbe, 0x7f, 0x5e,
    0xdc, 0xc5, 0x90, 0xf2, 0x55, 0xd0, 0x49, 0xbd, 0xa3, 0x10, 0x98, 0xf3, 0xef, 0x60, 0xb9, 0x24,
    0x40, 0xf5, 0xa1, 0x2f, 0x61, 0x96, 0x71, 0xd5, 0x38, 0x82, 0x64, 0x1b, 0xa5, 0xd7, 0xf7, 0x8f,
    0x14, 0x3a, 0x6c, 0x34, 0x4a, 0xda, 0x45, 0x02, 0x99, 0x91, 0x05, 0x26, 0xcd, 0xcb, 0x41, 0xeb,
    0xcc, 0xc3, 0xac, 0x50, 0x79, 0x1c, 0xca, 0xf6, 0x8e, 0xe4, 0x00, 0x5b, 0x08, 0x68, 0xdd, 0x9b,
    0xff, 0x4b, 0x89, 0x28, 0xa8, 0x76, 0x58, 0xc4, 0x2a, 0x4c, 0x9f, 0x54, 0xa4, 0xe7, 0x70, 0xed,
]

# 初始化随机种子
random.seed(time.time())
np.random.seed(int(time.time()))
torch.manual_seed(int(time.time()))
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(int(time.time()))


# ======================== 并行计算工具函数 ========================

def parallel_fitness(s_boxes: List[List[int]], use_dl: bool) -> List[float]:
    """并行计算一组S盒的适应度"""
    # 根据是否使用GPU选择并行策略
    if torch.cuda.is_available():
        # GPU环境下使用单进程避免CUDA冲突
        return [fitness(s_box, use_dl) for s_box in tqdm(s_boxes, desc="顺序计算适应度")]
    else:
        # CPU环境下使用多进程
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(fitness, s_box, use_dl) for s_box in s_boxes]
            return [f.result() for f in tqdm(as_completed(futures), total=len(s_boxes), desc="并行计算适应度")]


# ======================== 混沌系统函数 ========================

# Logistic映射函数
def logistic_map(z: float, s: float) -> float:
    return s * z * (1 - z)


# 使用Logistic映射生成[0,1)之间的随机数
def chaotic_rand(z: List[float], s: float) -> float:
    z[0] = logistic_map(z[0], s)
    return z[0]


# 使用自定义的S盒
def use_custom_S_box() -> List[int]:
    return CUSTOM_S_BOX.copy()


# 生成初始S盒 - 使用改进的混沌映射
def generate_initial_S_box(s: float, z0: float, iterations: int) -> List[int]:
    S = [-1] * DIMENSION
    count = 0
    z = z0

    # 消除初始值影响
    for _ in range(iterations):
        z = logistic_map(z, s)

    # 使用双混沌系统增加随机性
    z2 = random.random()
    s2 = 3.99

    while count < DIMENSION:
        z = logistic_map(z, s)
        z2 = logistic_map(z2, s2)
        # 组合两个混沌值
        combined = (z + z2) % 1.0
        Z = int(combined * DIMENSION)

        if Z not in S[:count]:
            S[count] = Z
            count += 1

    return S


# 变换S盒
def transform_S_box(S: List[int]) -> List[int]:
    a = 3
    b = 1
    S_matrix = [[0] * MATRIX_SIZE for _ in range(MATRIX_SIZE)]
    new_S_matrix = [[0] * MATRIX_SIZE for _ in range(MATRIX_SIZE)]

    # 将一维数组转换为二维数组
    for r in range(MATRIX_SIZE):
        for c in range(MATRIX_SIZE):
            S_matrix[r][c] = S[r * MATRIX_SIZE + c]

    # 按公式进行变换
    for r in range(MATRIX_SIZE):
        for c in range(MATRIX_SIZE):
            r1 = (r + a * c) % MATRIX_SIZE
            c1 = (b * r + (1 + a * b) * c) % MATRIX_SIZE
            new_S_matrix[r1][c1] = S_matrix[r][c]

    # 将变换后的二维数组转换回一维数组
    for r in range(MATRIX_SIZE):
        for c in range(MATRIX_SIZE):
            S[r * MATRIX_SIZE + c] = new_S_matrix[r][c]

    return S


# 生成S盒种群
def generate_S_box_population(s: float, iterations: int) -> List[List[int]]:
    population = []

    # 对于第一个粒子，使用自定义的S盒
    population.append(use_custom_S_box())

    # 其他粒子使用混沌映射生成
    for _ in range(1, NUM_PARTICLES):
        z0 = random.random()
        population.append(generate_initial_S_box(s, z0, iterations))

    return population


# ======================== 密码学指标计算 (优化版) ========================

def hamming_weight(x: torch.Tensor) -> torch.Tensor:
    """计算整数的汉明重量（二进制表示中1的数量）"""
    # 使用位操作计算汉明重量
    x = x.to(torch.int32)
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & 0x7F


def nonlinearity_gpu(s_box: List[int]) -> int:
    """使用GPU加速计算非线性度"""
    n = 8
    min_nl = 256

    # 将S盒转换为PyTorch张量
    s_tensor = torch.tensor(s_box, dtype=torch.uint8, device=device)

    for bit in range(n):
        # 提取特定位
        truth_table = ((s_tensor >> bit) & 1).float()

        # 计算Walsh谱
        walsh_spectrum = torch.zeros(DIMENSION, device=device)
        for w in range(DIMENSION):
            # 计算点积 (x·w)
            x = torch.arange(DIMENSION, device=device)
            # 计算汉明重量后模2
            dot_product = hamming_weight(x & w) % 2
            sum_val = torch.sum(torch.where(dot_product == truth_table, 1.0, -1.0))
            walsh_spectrum[w] = torch.abs(sum_val)

        # 计算当前位的非线性度
        max_walsh = torch.max(walsh_spectrum)
        nl = 128 - max_walsh.item() / 2
        if nl < min_nl:
            min_nl = int(nl)

    return min_nl


def AC_f_gpu(s_box: List[int]) -> int:
    """优化版差分均匀度计算"""
    s_tensor = torch.tensor(s_box, dtype=torch.uint8, device=device)
    max_diff = 0

    # 使用批处理计算所有输入差分
    for a in range(1, DIMENSION):
        # 使用循环移位代替异或操作
        shifted = torch.roll(s_tensor, a)
        diff = s_tensor ^ shifted

        # 使用直方图统计
        counts = torch.bincount(diff, minlength=DIMENSION)
        max_count = torch.max(counts).item()

        # 提前终止检查
        if max_count > max_diff:
            max_diff = max_count
            # 如果已经超过阈值，提前返回
            if max_diff > 12:  # 12是典型的安全阈值
                return max_diff

    return max_diff


def compute_sbox_algebraic_degree_gpu(s_box: List[int]) -> int:
    """优化版代数次数计算"""
    n = 8
    max_degree = 0
    s_tensor = torch.tensor(s_box, dtype=torch.uint8, device=device)

    # 使用位并行处理
    for bit in range(n):
        truth_table = ((s_tensor >> bit) & 1).to(torch.uint8)

        # 执行Mobius变换
        for k in range(n):
            mask = 1 << k
            indices = torch.arange(DIMENSION, device=device)
            mask_tensor = (indices & mask) == 0
            indices_to_update = indices[mask_tensor]
            indices_target = indices_to_update | mask

            # 批量更新
            truth_table[indices_target] = truth_table[indices_to_update] ^ truth_table[indices_target]

        # 计算代数次数
        non_zero_indices = torch.nonzero(truth_table).flatten()
        if non_zero_indices.numel() > 0:
            degrees = hamming_weight(non_zero_indices)
            current_max = torch.max(degrees).item()
            if current_max > max_degree:
                max_degree = current_max

    return max_degree


def calculate_sac_gpu(s_box: List[int]) -> float:
    """优化版SAC计算"""
    n = len(s_box)
    s_tensor = torch.tensor(s_box, dtype=torch.uint8, device=device)

    # 使用矩阵运算替代循环
    input_masks = torch.tensor([1 << i for i in range(8)], dtype=torch.long, device=device)
    output_masks = torch.tensor([1 << i for i in range(8)], dtype=torch.long, device=device)

    # 批量处理所有输入比特变化
    x = torch.arange(n, dtype=torch.long, device=device)
    x_flip = x.unsqueeze(1) ^ input_masks

    # 获取原始输出和翻转后的输出
    y_orig = s_tensor[x]
    y_flip = s_tensor[x_flip]

    # 计算比特变化
    diff = y_orig.unsqueeze(1) ^ y_flip
    flip_counts = ((diff.unsqueeze(2) & output_masks) != 0)

    # 计算满足率
    sac_values = flip_counts.float().mean(dim=0)
    avg_sac = torch.mean(torch.abs(sac_values - 0.5))

    return 1.0 - avg_sac.item()


def calculate_bic_gpu(s_box: List[int]) -> float:
    """优化版BIC计算"""
    n = len(s_box)
    s_tensor = torch.tensor(s_box, dtype=torch.uint8, device=device)

    # 准备输入输出掩码
    input_masks = torch.tensor([1 << i for i in range(8)], dtype=torch.long, device=device)
    output_masks = torch.tensor([1 << i for i in range(8)], dtype=torch.long, device=device)

    # 生成输入索引
    x = torch.arange(n, dtype=torch.long, device=device)
    x_flip = x.unsqueeze(1) ^ input_masks

    # 计算输出差异
    y_orig = s_tensor[x]
    y_flip = s_tensor[x_flip]
    diff = y_orig.unsqueeze(1) ^ y_flip

    # 计算比特变化
    bit_changes = (diff.unsqueeze(2) & output_masks) != 0

    # 计算比特对独立性
    bic_values = torch.zeros((8, 8, 8), device=device)
    for i in range(8):
        for j in range(i + 1, 8):
            # 检查两个比特是否同时变化
            both_changed = bit_changes[:, :, i] & bit_changes[:, :, j]
            bic_values[i, j] = both_changed.float().mean(dim=0)

    # 计算整体满足率
    avg_bic = torch.mean(torch.abs(bic_values - 0.25))
    return 1.0 - avg_bic.item()


# 根据GPU可用性选择函数
if torch.cuda.is_available():
    nonlinearity = nonlinearity_gpu
    AC_f = AC_f_gpu
    compute_sbox_algebraic_degree = compute_sbox_algebraic_degree_gpu
    calculate_sac = calculate_sac_gpu
    calculate_bic = calculate_bic_gpu
else:
    # CPU版本的密码学指标计算
    def nonlinearity(f: List[int]) -> int:
        n = 8
        min_nl = 256
        for bit in range(n):
            truth_table = []
            for x in range(DIMENSION):
                truth_table.append((f[x] >> bit) & 1)
            walsh_spectrum = [0] * DIMENSION
            for w in range(DIMENSION):
                sum_val = 0
                for x in range(DIMENSION):
                    dot_product = bin(x & w).count("1") % 2
                    sum_val += 1 if dot_product == truth_table[x] else -1
                walsh_spectrum[w] = abs(sum_val)
            max_walsh = max(walsh_spectrum)
            nl = 128 - max_walsh // 2
            if nl < min_nl:
                min_nl = nl
        return min_nl


    def AC_f(f: List[int]) -> int:
        diff_table = np.zeros((DIMENSION, DIMENSION), dtype=int)
        s_arr = np.array(f)
        for a in range(1, DIMENSION):
            diff = s_arr ^ np.roll(s_arr, a)
            unique, counts = np.unique(diff, return_counts=True)
            for u, c in zip(unique, counts):
                diff_table[a, u] = c
        return np.max(diff_table[1:])


    def compute_sbox_algebraic_degree(s_box: List[int]) -> int:
        max_degree = 0
        for bit in range(8):
            truth_table = [0] * DIMENSION
            for x in range(DIMENSION):
                truth_table[x] = (s_box[x] >> bit) & 1
            # 执行Mobius变换
            for k in range(8):
                mask = 1 << k
                for i in range(DIMENSION):
                    if not (i & mask):
                        j = i | mask
                        truth_table[j] = truth_table[i] ^ truth_table[j]
            # 计算代数次数
            for i in range(DIMENSION):
                if truth_table[i] != 0:
                    degree = bin(i).count("1")
                    if degree > max_degree:
                        max_degree = degree
        return max_degree


    def calculate_sac(s_box: List[int]) -> float:
        n = len(s_box)
        sac_matrix = np.zeros((8, 8), dtype=float)
        for input_bit in range(8):
            mask = 1 << input_bit
            for output_bit in range(8):
                count = 0
                for x in range(n):
                    x_flip = x ^ mask
                    y_orig = s_box[x]
                    y_flip = s_box[x_flip]
                    if (y_orig ^ y_flip) & (1 << output_bit):
                        count += 1
                sac_matrix[input_bit][output_bit] = count / n
        avg_sac = np.mean(np.abs(sac_matrix - 0.5))
        return 1.0 - avg_sac


    def calculate_bic(s_box: List[int]) -> float:
        n = len(s_box)
        bic_matrix = np.zeros((8, 8, 8), dtype=float)
        for i in range(8):
            for j in range(i + 1, 8):
                for k in range(8):
                    mask = 1 << k
                    count = 0
                    for x in range(n):
                        x_flip = x ^ mask
                        y_orig = s_box[x]
                        y_flip = s_box[x_flip]
                        change_i = (y_orig ^ y_flip) & (1 << i)
                        change_j = (y_orig ^ y_flip) & (1 << j)
                        if change_i and change_j:
                            count += 1
                    bic_matrix[i][j][k] = count / n
        avg_bic = np.mean(np.abs(bic_matrix - 0.25))
        return 1.0 - avg_bic


# ======================== 深度学习模型 (增强版) ========================

class EnhancedSBoxPredictor(nn.Module):
    """增强版S盒质量预测模型"""

    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x)


class SBoxPredictorWrapper:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.input_size = 0
        self.trained = False

    def extract_features(self, s_box: List[int]) -> List[float]:
        """提取S盒的特征向量"""
        features = []
        features.append(nonlinearity(s_box))
        features.append(AC_f(s_box))
        features.append(compute_sbox_algebraic_degree(s_box))
        features.append(calculate_sac(s_box))
        features.append(calculate_bic(s_box))

        # 增加更多统计特征
        hist, _ = np.histogram(s_box, bins=32)
        features.extend(hist.tolist())

        # 差分特征
        max_diff = AC_f(s_box)
        features.extend([max_diff])

        # 统计特征
        features.append(np.mean(s_box))
        features.append(np.std(s_box))
        features.append(np.min(s_box))
        features.append(np.max(s_box))
        features.append(np.median(s_box))

        # 自相关特征
        autocorr = np.correlate(s_box, s_box, mode='full')
        features.append(np.max(autocorr))
        features.append(np.mean(autocorr))
        features.append(np.std(autocorr))

        # 非线性特征
        features.append(sum(1 for x in s_box if x & 1))

        self.input_size = len(features)
        return features

    def train(self, s_boxes: List[List[int]], scores: List[float]):
        """训练预测模型"""
        X = np.array([self.extract_features(s) for s in s_boxes])
        y = np.array(scores).reshape(-1, 1)

        # 数据标准化
        X = self.scaler.fit_transform(X)

        # 转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor, y_tensor, test_size=0.15, random_state=42
        )

        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # 初始化增强模型
        self.model = EnhancedSBoxPredictor(X_train.shape[1])
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            X_test = X_test.cuda()
            y_test = y_test.cuda()

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)

        # 训练模型
        self.model.train()
        best_loss = float('inf')
        for epoch in range(1000):  # 增加训练轮次
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪
                optimizer.step()
                epoch_loss += loss.item()

            # 评估验证集
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_test)
                val_loss = criterion(val_outputs, y_test)
                scheduler.step(val_loss)

                # 早停检查
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = self.model.state_dict()

            # 每50个epoch打印一次损失
            if (epoch + 1) % 50 == 0:
                print(
                    f'Epoch {epoch + 1}/1000, Loss: {epoch_loss / len(train_loader):.6f}, Val Loss: {val_loss.item():.6f}')

        # 加载最佳模型
        self.model.load_state_dict(best_model)

        # 最终评估
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test)
            test_loss = criterion(test_outputs, y_test)
            r2 = r2_score(y_test.cpu().numpy(), test_outputs.cpu().numpy())
            print(f"模型训练完成, 测试损失: {test_loss.item():.4f}, R^2分数: {r2:.4f}")

        self.trained = True

    def predict(self, s_box: List[int]) -> float:
        """预测S盒质量分数"""
        if not self.trained or self.model is None:
            return 0.0

        features = self.extract_features(s_box)
        features = self.scaler.transform([features])
        features_tensor = torch.tensor(features, dtype=torch.float32)

        if torch.cuda.is_available():
            features_tensor = features_tensor.cuda()

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(features_tensor)

        return prediction.item()


# 初始化深度学习预测器
quality_predictor = SBoxPredictorWrapper()


# ======================== PSO核心算法 (增强版) ========================

def fitness(s_box: List[int], use_dl: bool = False) -> float:
    """计算S盒的综合适应度"""
    # 使用深度学习模型预测质量
    if use_dl and quality_predictor.trained:
        dl_score = quality_predictor.predict(s_box)
        if dl_score < DL_THRESHOLD:
            return dl_score

    # 计算各项密码学指标
    N_s = nonlinearity(s_box)
    delta_s = AC_f(s_box)
    A_s = compute_sbox_algebraic_degree(s_box)
    sac_score = calculate_sac(s_box)
    bic_score = calculate_bic(s_box)

    # 应用更严格的约束条件
    if delta_s > 8:  # 更严格
        return 0.0
    if N_s < 100:  # 非线性度下限
        return 0.0
    if A_s < 6:  # 代数次数下限
        return 0.0

    # 综合适应度计算
    fitness_score = (
            a_s * N_s -
            a_d * delta_s +
            a_a * A_s +
            a_sac * sac_score +
            a_bic * bic_score
    )

    return max(fitness_score, 0.0)


def calculate_exchange_sequence(current: List[int], target: List[int]) -> Tuple[List[List[int]], int]:
    current = current.copy()
    exchange_sequence = []

    for i in range(DIMENSION):
        if current[i] != target[i]:
            for j in range(i + 1, DIMENSION):
                if current[j] == target[i]:
                    exchange_sequence.append([i, j])
                    current[i], current[j] = current[j], current[i]
                    break

    return exchange_sequence, len(exchange_sequence)


def sort_exchange_sequence(exchange_sequence: List[List[int]]) -> List[List[int]]:
    return sorted(exchange_sequence, key=lambda x: x[0])


def apply_exchange_sequence(particle: List[int], exchange_sequence: List[List[int]]) -> List[int]:
    particle = particle.copy()
    for swap in exchange_sequence:
        pos1, pos2 = swap
        particle[pos1], particle[pos2] = particle[pos2], particle[pos1]
    return particle


def apply_inertia_exchange_sequence(previous_retained_exchange: List[List[int]],
                                    w: float, z_rand: List[float], s: float) -> List[List[int]]:
    current_velocities = []

    for swap in previous_retained_exchange:
        if chaotic_rand(z_rand, s) < w:
            current_velocities.append(swap)
            if len(current_velocities) >= MAX_VELOCITY:
                break

    return current_velocities


def is_custom_S_box(s_box: List[int]) -> bool:
    return s_box == CUSTOM_S_BOX


def mutate_s_box(s_box: List[int], mutation_strength: int = 1) -> List[int]:
    s_box = s_box.copy()
    if random.random() < MUTATION_RATE:
        for _ in range(mutation_strength + random.randint(0, 2)):
            idx1, idx2 = random.sample(range(DIMENSION), 2)
            s_box[idx1], s_box[idx2] = s_box[idx2], s_box[idx1]
    return s_box


def local_search_gpu(best_s_box: List[int], num_searches: int = 300) -> List[int]:
    """使用GPU并行化局部搜索"""
    best_score = fitness(best_s_box)
    best_candidate = best_s_box.copy()

    # 批量生成候选解
    candidates = []
    for _ in range(num_searches):
        candidate = best_s_box.copy()
        num_swaps = random.randint(1, 3)
        for _ in range(num_swaps):
            idx1, idx2 = random.sample(range(DIMENSION), 2)
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
        candidates.append(candidate)

    # 批量计算适应度
    candidate_scores = parallel_fitness(candidates, use_dl=True)

    # 找到最佳候选
    best_idx = np.argmax(candidate_scores)
    if candidate_scores[best_idx] > best_score:
        best_candidate = candidates[best_idx]

    return best_candidate


def pso() -> Tuple[List[int], List[int], List[float], List[List[float]], List[float]]:
    # 初始化变量
    s = 3.999999
    iterations = 100
    z_rand = [random.random()]

    # 初始化粒子群
    print("生成初始S盒种群...")
    particles = generate_S_box_population(s, iterations)
    print(f"初始种群生成完成! 粒子数: {len(particles)}")

    # 训练深度学习预测模型
    print("训练深度学习质量预测模型...")
    training_set = particles[:]
    training_set.append(use_custom_S_box())

    # 生成训练分数
    print("并行计算训练分数...")
    training_scores = parallel_fitness(training_set, use_dl=False)

    # 训练模型
    quality_predictor.train(training_set, training_scores)
    print("深度学习模型训练完成!")

    # 初始化历史记录
    global_best_fitness_history = []
    particle_fitness_history = [[] for _ in range(NUM_PARTICLES)]
    inertia_weight_history = []

    # 初始化最优解
    print("初始化最优解...")
    p_best = [p.copy() for p in particles]
    p_best_fitness = parallel_fitness(particles, use_dl=True)

    g_best_idx = np.argmax(p_best_fitness)
    g_best = particles[g_best_idx].copy()
    g_best_fitness = p_best_fitness[g_best_idx]

    # 初始化速度
    velocities = [[] for _ in range(NUM_PARTICLES)]
    previous_retained_exchange = [[] for _ in range(NUM_PARTICLES)]

    print(f"初始全局最优适应度: {g_best_fitness:.2f}")

    # 创建精英粒子集合
    elite_indices = np.argsort(p_best_fitness)[-ELITE_SIZE:]
    elite_particles = [particles[i].copy() for i in elite_indices]

    # 迭代过程
    print(f"\n开始PSO优化 ({NUM_ITERATIONS} 次迭代, {NUM_PARTICLES} 个粒子)...")
    for t in tqdm(range(NUM_ITERATIONS), desc="PSO迭代进度"):
        w = W_MAX - (W_MAX - W_MIN) * t / NUM_ITERATIONS
        inertia_weight_history.append(w)

        # 记录当前适应度
        current_fitnesses = parallel_fitness(particles, use_dl=True)
        for i, fit in enumerate(current_fitnesses):
            particle_fitness_history[i].append(fit)
        global_best_fitness_history.append(g_best_fitness)

        # 更新每个粒子
        for i in range(NUM_PARTICLES):
            p_best_seq, _ = calculate_exchange_sequence(particles[i], p_best[i])
            g_best_seq, _ = calculate_exchange_sequence(particles[i], g_best)

            current_velocities = apply_inertia_exchange_sequence(
                previous_retained_exchange[i], w, z_rand, s
            )

            for swap in p_best_seq:
                if chaotic_rand(z_rand, s) < C1:
                    current_velocities.append(swap)
                    if len(current_velocities) >= MAX_VELOCITY:
                        break

            for swap in g_best_seq:
                if chaotic_rand(z_rand, s) < C2:
                    current_velocities.append(swap)
                    if len(current_velocities) >= MAX_VELOCITY:
                        break

            velocities[i] = current_velocities
            previous_retained_exchange[i] = current_velocities.copy()

            particles[i] = apply_exchange_sequence(particles[i], current_velocities)

            mutation_strength = 1 + t // (NUM_ITERATIONS // 3)
            particles[i] = mutate_s_box(particles[i], mutation_strength)

            current_fitness = fitness(particles[i], use_dl=True)
            if current_fitness > p_best_fitness[i]:
                p_best[i] = particles[i].copy()
                p_best_fitness[i] = current_fitness

        # 更新全局最优
        for i in range(NUM_PARTICLES):
            current_fitness = fitness(particles[i], use_dl=True)
            if current_fitness > g_best_fitness:
                g_best = particles[i].copy()
                g_best_fitness = current_fitness
                print(f"迭代 {t + 1}: 发现新的全局最优适应度 {g_best_fitness:.2f}")

        # 精英策略
        worst_idx = np.argmin(p_best_fitness)
        if p_best_fitness[worst_idx] < min(fitness(e) for e in elite_particles):
            elite_idx = random.randint(0, ELITE_SIZE - 1)
            particles[worst_idx] = elite_particles[elite_idx].copy()
            p_best_fitness[worst_idx] = fitness(particles[worst_idx])

        # 更新精英集合
        elite_indices = np.argsort(p_best_fitness)[-ELITE_SIZE:]
        elite_particles = [particles[i].copy() for i in elite_indices]

        # 增加混沌扰动
        if t % 5 == 0:  # 每5次迭代扰动一次
            for i in range(int(NUM_PARTICLES * 0.3)):
                idx = random.randint(0, NUM_PARTICLES - 1)
                particles[idx] = generate_initial_S_box(s, random.random(), 100)

        # 增强局部搜索
        if t % 3 == 0:  # 增加局部搜索频率
            if torch.cuda.is_available():
                improved_g_best = local_search_gpu(g_best, num_searches=500)
            else:
                improved_g_best = g_best
            improved_fitness = fitness(improved_g_best)
            if improved_fitness > g_best_fitness:
                g_best = improved_g_best
                g_best_fitness = improved_fitness
                print(f"迭代 {t + 1}: 局部搜索提升适应度到 {g_best_fitness:.2f}")

    # 最终局部搜索
    print("执行最终局部搜索...")
    if torch.cuda.is_available():
        improved_g_best = local_search_gpu(g_best, num_searches=500)
    else:
        improved_g_best = g_best
    improved_fitness = fitness(improved_g_best)
    if improved_fitness > g_best_fitness:
        g_best = improved_g_best
        g_best_fitness = improved_fitness
        print(f"最终局部搜索提升适应度到 {g_best_fitness:.2f}")

    # 添加最终适应度值
    global_best_fitness_history.append(g_best_fitness)
    final_fitnesses = parallel_fitness(particles, use_dl=True)
    for i, fit in enumerate(final_fitnesses):
        particle_fitness_history[i].append(fit)

    print(f"最终全局最优适应度: {g_best_fitness:.2f}")

    return g_best, particles, global_best_fitness_history, particle_fitness_history, inertia_weight_history


# ======================== 结果可视化 ========================

def visualize_results(global_best_history: List[float], particle_fitness_history: List[List[float]],
                      inertia_weight_history: List[float], best_s_box: List[int]) -> None:
    plt.figure(figsize=(18, 15))

    # 1. 全局最优适应度变化
    plt.subplot(3, 2, 1)
    plt.plot(global_best_history, 'o-', linewidth=2, color='darkblue')
    plt.title('Global Best Fitness Evolution', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(bottom=0)

    # 2. 所有粒子适应度变化
    plt.subplot(3, 2, 2)
    for i, history in enumerate(particle_fitness_history):
        plt.plot(history, alpha=0.5, linewidth=1)
    plt.title('Particle Fitness Evolution', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(bottom=0)

    # 3. 惯性权重变化
    plt.subplot(3, 2, 3)
    plt.plot(inertia_weight_history, 's-', color='purple', markersize=4)
    plt.title('Inertia Weight Evolution', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Inertia Weight', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 4. S盒可视化（只展示前32个值）
    plt.subplot(3, 2, 4)
    plt.bar(range(32), best_s_box[:32], color='skyblue')
    plt.title('First 32 Values of Best S-box', fontsize=14)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 5. 密码学特性指标
    nl = nonlinearity(best_s_box)
    du = AC_f(best_s_box)
    ad = compute_sbox_algebraic_degree(best_s_box)
    sac = calculate_sac(best_s_box)
    bic = calculate_bic(best_s_box)

    properties = ['Nonlinearity', 'Differential\nUniformity', 'Algebraic\nDegree', 'SAC', 'BIC']
    values = [nl, du, ad, sac, bic]
    ideal = [112, 4, 7, 0.5, 0.25]

    plt.subplot(3, 2, 5)
    x = np.arange(len(properties))
    width = 0.35
    plt.bar(x - width / 2, values, width, label='Actual', color='royalblue')
    plt.bar(x + width / 2, ideal, width, label='Ideal', color='limegreen', alpha=0.7)
    plt.xticks(x, properties, fontsize=10)
    plt.title('Cryptographic Properties Comparison', fontsize=14)
    plt.ylabel('Value', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate(values):
        plt.text(i - width / 2, v + 5, str(v), ha='center', fontsize=9)
        plt.text(i + width / 2, ideal[i] + 5, str(ideal[i]), ha='center', fontsize=9)

    # 6. 差分分布表
    plt.subplot(3, 2, 6)
    diff_table = np.zeros((DIMENSION, DIMENSION), dtype=int)
    for a in range(1, DIMENSION):
        for x in range(DIMENSION):
            b = best_s_box[x] ^ best_s_box[x ^ a]
            diff_table[a][b] += 1

    plt.imshow(np.log1p(diff_table[1:65, 1:65]), cmap='viridis', aspect='auto')
    plt.colorbar(label='Log(Count+1)')
    plt.title('Differential Distribution Table (64x64)', fontsize=14)
    plt.xlabel('Output Difference', fontsize=12)
    plt.ylabel('Input Difference', fontsize=12)

    plt.tight_layout(pad=3.0)
    plt.savefig('sbox_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nCryptographic Properties of Best S-box:")
    print(f"Nonlinearity: {nl} (理想值: 112)")
    print(f"Differential Uniformity: {du} (理想值: 4)")
    print(f"Algebraic Degree: {ad} (理想值: 7)")
    print(f"SAC Score: {sac:.4f} (理想值: 0.5)")
    print(f"BIC Score: {bic:.4f} (理想值: 0.25)")


# ======================== 主函数 ========================

def main():
    start_time = time.time()
    print("S盒优化程序开始运行...")
    print(f"配置参数: 粒子数={NUM_PARTICLES}, 迭代次数={NUM_ITERATIONS}, 维度={DIMENSION}")
    print(f"使用GPU加速: {torch.cuda.is_available()}")

    # 执行粒子群优化
    best_s_box, particles, global_best_history, particle_fitness_history, inertia_weight_history = pso()

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n优化完成! 总耗时: {int(hours):02d}h:{int(minutes):02d}m:{seconds:.2f}s")

    if is_custom_S_box(best_s_box):
        print("最优S盒是自定义S盒，输出次优S盒")
        second_best_fitness = -1
        second_best_s_box = None
        for s_box in particles:
            if not is_custom_S_box(s_box):
                current_fitness = fitness(s_box)
                if current_fitness > second_best_fitness:
                    second_best_fitness = current_fitness
                    second_best_s_box = s_box
        best_s_box = second_best_s_box
        print(f"次优S盒适应度: {second_best_fitness:.2f}")
    else:
        print("找到优于自定义S盒的优化结果")

    # 可视化结果
    print("\n生成可视化结果...")
    visualize_results(global_best_history, particle_fitness_history,
                      inertia_weight_history, best_s_box)

    # 输出最终S盒
    print("\n最佳S盒 (前32个值):")
    for i in range(0, 32, 8):
        print(" ".join(f"{x:3d}" for x in best_s_box[i:i + 8]))

    # 保存S盒到文件
    with open('optimized_sbox.txt', 'w') as f:
        f.write(f"// 优化S盒 - 生成时间: {time.ctime()}\n")
        f.write(f"// 非线性度: {nonlinearity(best_s_box)}")
        f.write(f" 差分均匀度: {AC_f(best_s_box)}")
        f.write(f" 代数次数: {compute_sbox_algebraic_degree(best_s_box)}\n")
        f.write("const uint8_t optimized_sbox[256] = {\n")
        for i in range(0, 256, 16):
            line = ", ".join(f"0x{x:02x}" for x in best_s_box[i:i + 16])
            f.write(f"    {line},\n")
        f.write("};\n")

    print("S盒已保存到 'optimized_sbox.txt'")


if __name__ == "__main__":
    main()