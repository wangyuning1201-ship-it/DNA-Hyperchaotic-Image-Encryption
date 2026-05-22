import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import chi2

# ------------------- 复用您提供的加密核心函数 -------------------
def generate_chaos(init, length, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = init
    X, Y, Z = np.zeros(length), np.zeros(length), np.zeros(length)
    for i in range(length):
        X[i], Y[i], Z[i] = x, y, z
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
    return X, Y, Z

def generate_dna_tables():
    rules = [
        [0,1,2,3], [0,2,1,3], [0,1,3,2], [0,3,1,2],
        [0,2,3,1], [0,3,2,1], [1,0,2,3], [1,0,3,2]
    ]
    enc_tables, dec_tables = [], []
    for rule in rules:
        enc = np.zeros(256, dtype=np.uint8)
        for b in range(256):
            g1 = (b>>6)&3; g2 = (b>>4)&3; g3 = (b>>2)&3; g4 = b&3
            nb = (rule[g1]<<6)|(rule[g2]<<4)|(rule[g3]<<2)|rule[g4]
            enc[b] = nb
        dec = np.zeros(256, dtype=np.uint8)
        for b in range(256):
            dec[enc[b]] = b
        enc_tables.append(enc)
        dec_tables.append(dec)
    return enc_tables, dec_tables

class CellularAutomaton:
    def __init__(self, rule=90):
        self.rule = rule
        if rule != 90:
            raise ValueError("Only rule 90 supported")
    def forward(self, data):
        n = len(data)
        if n==0: return data.copy()
        res = np.zeros(n, dtype=np.uint8)
        if n==1: return res
        res[0] = data[1]
        for i in range(1, n-1):
            res[i] = data[i-1] ^ data[i+1]
        res[n-1] = data[n-2]
        return res
    def backward(self, data):
        n = len(data)
        if n<=1: return data.copy()
        a = np.zeros(n, dtype=np.uint8)
        b = np.zeros(n, dtype=bool)
        a[0], b[0] = 0, True
        a[1], b[1] = data[0], False
        for i in range(2, n):
            a[i] = data[i-1] ^ a[i-2]
            b[i] = b[i-2]
        if b[n-2]:
            x0 = data[n-1] ^ a[n-2]
        else:
            if a[n-2] != data[n-1]:
                raise ValueError("Not invertible")
            x0 = 0
        res = np.zeros(n, dtype=np.uint8)
        res[0] = x0
        if n>1: res[1] = a[1] ^ (x0 if b[1] else 0)
        for i in range(2, n):
            res[i] = a[i] ^ (x0 if b[i] else 0)
        return res

def encrypt_image(image_path, key, s_box, output_path, rounds=3):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    shape = img_array.shape
    flat = img_array.flatten().astype(np.uint8)
    N = len(flat)

    chaos_len = N + 1000
    X, Y, Z = generate_chaos(key, chaos_len)
    X, Y, Z = X[1000:], Y[1000:], Z[1000:]

    rules = (np.abs(X[:N])*1e8).astype(np.uint64) % 8
    rules = rules.astype(np.uint8)
    perm = np.argsort(X[:N])
    chaos_xor = (np.abs(Y[:N])*1e8).astype(np.uint64) % 256
    chaos_xor = chaos_xor.astype(np.uint8)

    enc_tables, _ = generate_dna_tables()
    data = flat.copy()
    for _ in range(rounds):
        data = np.array([s_box[b] for b in data], dtype=np.uint8)
        data = data[perm]
        data = np.array([enc_tables[rules[i]][data[i]] for i in range(N)], dtype=np.uint8)
        data ^= chaos_xor
        ca = CellularAutomaton()
        data = ca.forward(data)
        for i in range(1, N):
            data[i] ^= data[i-1]
        for i in range(N-2, -1, -1):
            data[i] ^= data[i+1]

    enc_img = data.reshape(shape)
    Image.fromarray(enc_img, 'RGB').save(output_path, format='PNG')
    return enc_img

def calculate_entropy(image):
    """计算单通道或RGB图像的香农熵（8位图像理想值8）"""
    if len(image.shape) == 3:
        ents = []
        for c in range(3):
            hist = np.histogram(image[:,:,c].flatten(), bins=256, range=(0,256))[0]
            hist = hist[hist>0] / hist.sum()
            ents.append(-np.sum(hist * np.log2(hist)))
        return np.mean(ents), ents[0], ents[1], ents[2]  # 平均, R, G, B
    else:
        hist = np.histogram(image.flatten(), bins=256, range=(0,256))[0]
        hist = hist[hist>0] / hist.sum()
        return -np.sum(hist * np.log2(hist)), None, None, None

# ------------------- 信息熵实验主程序 -------------------
def entropy_experiment(image_path, key, s_box, rounds=3, save_plot="entropy_comparison.png"):
    """
    执行信息熵实验：
    1. 加密图像
    2. 计算原始图像和密文的熵（RGB通道及平均）
    3. 绘制柱状图对比，标注理想值8
    """
    # 加密
    print(f"加密图像: {image_path} ...")
    enc_img = encrypt_image(image_path, key, s_box, "temp_enc_entropy.png", rounds=rounds)
    original = np.array(Image.open(image_path).convert('RGB'))

    # 计算熵
    ent_orig_mean, ent_orig_r, ent_orig_g, ent_orig_b = calculate_entropy(original)
    ent_enc_mean, ent_enc_r, ent_enc_g, ent_enc_b = calculate_entropy(enc_img)

    # 打印数值结果
    print("\n========== 信息熵实验结果 ==========")
    print(f"原始图像     平均熵: {ent_orig_mean:.4f} (R:{ent_orig_r:.4f}, G:{ent_orig_g:.4f}, B:{ent_orig_b:.4f})")
    print(f"加密后图像   平均熵: {ent_enc_mean:.4f} (R:{ent_enc_r:.4f}, G:{ent_enc_g:.4f}, B:{ent_enc_b:.4f})")
    print(f"理想值: 8.0000")
    print(f"加密后熵提升: {ent_enc_mean - ent_orig_mean:.4f}")
    print("==================================\n")

    # 可视化
    categories = ['Red', 'Green', 'Blue', 'Average']
    orig_vals = [ent_orig_r, ent_orig_g, ent_orig_b, ent_orig_mean]
    enc_vals  = [ent_enc_r,  ent_enc_g,  ent_enc_b,  ent_enc_mean]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, orig_vals, width, label='Original', color='skyblue')
    bars2 = ax.bar(x + width/2, enc_vals, width, label='Encrypted', color='orange')
    ax.axhline(y=8, color='red', linestyle='--', linewidth=1.5, label='Ideal Entropy (8.0)')

    ax.set_ylabel('Entropy (bits)', fontsize=12)
    ax.set_title('Information Entropy Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.7)

    # 在柱子上方标注数值
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_plot, dpi=300)
    plt.close()
    print(f"可视化图表已保存至: {save_plot}")

    return {
        "original": {"mean": ent_orig_mean, "R": ent_orig_r, "G": ent_orig_g, "B": ent_orig_b},
        "encrypted": {"mean": ent_enc_mean, "R": ent_enc_r, "G": ent_enc_g, "B": ent_enc_b}
    }

# ------------------- 运行示例 -------------------
if __name__ == "__main__":
    # 准备测试图像（如果不存在则生成）
    try:
        from skimage import data
        test_img = data.camera()
        test_img_rgb = np.stack([test_img, test_img, test_img], axis=2)
        Image.fromarray(test_img_rgb).save("cameraman.png")
        image_path = "cameraman.png"
        print("使用 Cameraman 测试图像")
    except:
        test_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        Image.fromarray(test_img).save("test.png")
        image_path = "test.png"
        print("skimage 不可用，使用随机测试图像")

    # 随机 S-box 和密钥（与您原代码一致）
    rng = np.random.RandomState(42)
    s_box = rng.permutation(256).tolist()
    key = (1.0, 1.0, 1.0)

    # 执行实验
    entropy_results = entropy_experiment(image_path, key, s_box, rounds=3, save_plot="entropy_comparison.png")