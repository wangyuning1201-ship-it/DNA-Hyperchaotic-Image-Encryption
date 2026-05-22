import numpy as np
from PIL import Image
import time
import os
from scipy.stats import chi2
import matplotlib.pyplot as plt
import math

# ==================== 1. Hyperchaotic System (Lorenz) ====================
def generate_chaos(init, length, dt=0.01, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """
    Generate Lorenz chaotic sequences.
    :param init: tuple (x0, y0, z0)
    :param length: required sequence length
    :param dt: integration step
    :return: three numpy arrays X, Y, Z, each of length 'length'
    """
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


# ==================== 2. DNA Encoding Tables ====================
def generate_dna_tables():
    """
    Generate 8 DNA encoding and decoding tables.
    Each rule maps 2-bit values (0-3) to nucleotides (0-3).
    Returns:
        enc_tables: list of 8 arrays, each of length 256,
                    enc_tables[rule][original_byte] = encoded_byte
        dec_tables: list of 8 arrays, each of length 256,
                    dec_tables[rule][encoded_byte] = original_byte
    """
    # 8 rules, each is a length-4 list: index = 2-bit value, value = nucleotide
    rules = [
        [0, 1, 2, 3],  # Rule 0
        [0, 2, 1, 3],  # Rule 1
        [0, 1, 3, 2],  # Rule 2
        [0, 3, 1, 2],  # Rule 3
        [0, 2, 3, 1],  # Rule 4
        [0, 3, 2, 1],  # Rule 5
        [1, 0, 2, 3],  # Rule 6
        [1, 0, 3, 2],  # Rule 7
    ]
    enc_tables, dec_tables = [], []
    for rule in rules:
        enc = np.zeros(256, dtype=np.uint8)
        for b in range(256):
            # Split byte into four 2-bit groups (high to low)
            g1 = (b >> 6) & 0x03
            g2 = (b >> 4) & 0x03
            g3 = (b >> 2) & 0x03
            g4 = b & 0x03
            # Map
            ng1 = rule[g1]
            ng2 = rule[g2]
            ng3 = rule[g3]
            ng4 = rule[g4]
            # Combine into a new byte
            nb = (ng1 << 6) | (ng2 << 4) | (ng3 << 2) | ng4
            enc[b] = nb
        # Build decoding table as inverse of encoding table
        dec = np.zeros(256, dtype=np.uint8)
        for b in range(256):
            dec[enc[b]] = b
        enc_tables.append(enc)
        dec_tables.append(dec)
    return enc_tables, dec_tables


# ==================== 3. Reversible Cellular Automaton (Rule 90) ====================
class CellularAutomaton:
    def __init__(self, rule=90):
        self.rule = rule  # currently only rule 90 is supported
        if rule != 90:
            raise ValueError("Only rule 90 is supported currently")

    def forward(self, data):
        """
        Forward diffusion: new[i] = left[i] XOR right[i] (outside boundaries treated as 0)
        :param data: 1D numpy array, uint8 type
        :return: diffused array
        """
        n = len(data)
        if n == 0:
            return data.copy()
        result = np.zeros(n, dtype=np.uint8)
        if n == 1:
            return result  # for single element, result is always 0
        result[0] = 0 ^ data[1]  # left boundary: left neighbor = 0
        for i in range(1, n - 1):
            result[i] = data[i - 1] ^ data[i + 1]
        result[n - 1] = data[n - 2] ^ 0  # right boundary: right neighbor = 0
        return result

    def backward(self, data):
        """
        Inverse diffusion: solve tridiagonal XOR system to recover original data.
        :param data: diffused array
        :return: original array
        """
        n = len(data)
        if n == 0:
            return data.copy()
        if n == 1:
            # For a single element forward yields 0, so only data[0]==0 is invertible; here we return original (won't be used)
            return data.copy()
        # Thomas algorithm (XOR version) to solve linear system
        # a[i], b[i] satisfy: x[i] = a[i] ^ (b[i] & x0), where b[i] is bool
        a = np.zeros(n, dtype=np.uint8)
        b = np.zeros(n, dtype=bool)

        # i=0
        a[0] = 0
        b[0] = True
        # i=1 from equation i=0: x[1] = data[0]
        a[1] = data[0]
        b[1] = False
        # recurrence i=2..n-1
        for i in range(2, n):
            # equation i-1: x[i-2] ^ x[i] = data[i-1]  => x[i] = data[i-1] ^ x[i-2]
            a[i] = data[i - 1] ^ a[i - 2]
            b[i] = b[i - 2]

        # last equation i=n-1: x[n-2] = data[n-1]  => a[n-2] ^ (b[n-2] & x0) = data[n-1]
        if b[n - 2]:
            x0 = data[n - 1] ^ a[n - 2]
        else:
            # if b[n-2]==0, require a[n-2]==data[n-1], otherwise singular. Assume invertible, set x0=0.
            if a[n - 2] != data[n - 1]:
                raise ValueError("Cellular automaton is not invertible for this data (singular matrix)")
            x0 = 0

        # back substitution to compute all x
        result = np.zeros(n, dtype=np.uint8)
        result[0] = x0
        if n > 1:
            result[1] = a[1] ^ (x0 if b[1] else 0)
        for i in range(2, n):
            result[i] = a[i] ^ (x0 if b[i] else 0)
        return result


# ==================== 4. Enhanced Encryption and Decryption (with bidirectional feedback XOR) ====================
def encrypt_image(image_path, key, s_box, output_path, rounds=3):
    """
    Encrypt an image (enhanced version: chaotic XOR + bidirectional feedback XOR)
    :param image_path: input image path (RGB format)
    :param key: key as a triple (x0, y0, z0) for Lorenz system
    :param s_box: custom S-box, list of 256 unique integers 0-255
    :param output_path: encrypted image save path (PNG recommended)
    :param rounds: number of encryption rounds (default 3)
    """
    # Read image and flatten to byte array
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    original_shape = img_array.shape
    flat = img_array.flatten().astype(np.uint8)
    N = len(flat)

    # Generate chaotic sequences (discard first 1000 transient steps)
    chaos_len = N + 1000
    X, Y, Z = generate_chaos(key, chaos_len)
    X = X[1000:]
    Y = Y[1000:]
    Z = Z[1000:]

    # Rule selection sequence (one rule per byte)
    rules = (np.abs(X[:N]) * 1e8).astype(np.uint64) % 8
    rules = rules.astype(np.uint8)

    # Position permutation indices (based on sorting of X)
    perm_indices = np.argsort(X[:N])   # forward index
    inv_perm = np.argsort(perm_indices) # inverse index (for decryption)

    # Chaotic XOR sequence (using Y component)
    chaos_xor = (np.abs(Y[:N]) * 1e8).astype(np.uint64) % 256
    chaos_xor = chaos_xor.astype(np.uint8)

    # Precompute DNA tables
    enc_tables, dec_tables = generate_dna_tables()

    data = flat.copy()
    for _ in range(rounds):
        # 1. S-box substitution
        data = np.array([s_box[b] for b in data], dtype=np.uint8)
        # 2. Position permutation
        data = data[perm_indices]
        # 3. DNA encoding (dynamic rule)
        data = np.array([enc_tables[rules[i]][data[i]] for i in range(N)], dtype=np.uint8)
        # 4. Chaotic XOR
        data ^= chaos_xor
        # 5. CA diffusion
        ca = CellularAutomaton(rule=90)
        data = ca.forward(data)
        # 6. Forward feedback XOR (propagate differences backward)
        for i in range(1, N):
            data[i] ^= data[i-1]
        # 7. Backward feedback XOR (propagate differences forward)
        for i in range(N-2, -1, -1):
            data[i] ^= data[i+1]

    # Save encrypted image
    enc_img_array = data.reshape(original_shape)
    enc_img = Image.fromarray(enc_img_array, 'RGB')
    enc_img.save(output_path, format='PNG')


def decrypt_image(encrypted_path, key, s_box, output_path, rounds=3):
    """
    Decrypt an image (reverse of enhanced encryption)
    :param encrypted_path: encrypted image path
    :param key: same key as used in encryption
    :param s_box: same S-box as used in encryption
    :param output_path: decrypted image save path
    :param rounds: number of rounds (must match encryption)
    """
    img = Image.open(encrypted_path).convert('RGB')
    img_array = np.array(img)
    original_shape = img_array.shape
    flat = img_array.flatten().astype(np.uint8)
    N = len(flat)

    # Generate chaotic sequences (exactly the same as encryption)
    chaos_len = N + 1000
    X, Y, Z = generate_chaos(key, chaos_len)
    X = X[1000:]
    Y = Y[1000:]
    Z = Z[1000:]

    rules = (np.abs(X[:N]) * 1e8).astype(np.uint64) % 8
    rules = rules.astype(np.uint8)
    perm_indices = np.argsort(X[:N])
    inv_perm = np.argsort(perm_indices)

    # Chaotic XOR sequence (same as encryption)
    chaos_xor = (np.abs(Y[:N]) * 1e8).astype(np.uint64) % 256
    chaos_xor = chaos_xor.astype(np.uint8)

    # Compute inverse S-box
    inv_s_box = [0] * 256
    for i, val in enumerate(s_box):
        inv_s_box[val] = i

    enc_tables, dec_tables = generate_dna_tables()

    data = flat.copy()
    for _ in range(rounds):
        # Reverse order: undo the two feedback layers first (backward then forward)
        # 1. Inverse backward feedback XOR (forward order)
        for i in range(0, N-1):
            data[i] ^= data[i+1]
        # 2. Inverse forward feedback XOR (reverse order)
        for i in range(N-1, 0, -1):
            data[i] ^= data[i-1]
        # 3. Inverse CA diffusion
        ca = CellularAutomaton(rule=90)
        data = ca.backward(data)
        # 4. Inverse chaotic XOR
        data ^= chaos_xor
        # 5. DNA decoding
        data = np.array([dec_tables[rules[i]][data[i]] for i in range(N)], dtype=np.uint8)
        # 6. Inverse permutation
        data = data[inv_perm]
        # 7. Inverse S-box
        data = np.array([inv_s_box[b] for b in data], dtype=np.uint8)

    dec_img_array = data.reshape(original_shape)
    dec_img = Image.fromarray(dec_img_array, 'RGB')
    dec_img.save(output_path, format='PNG')


# ==================== Evaluation Metrics ====================
def calculate_entropy(image):
    """Calculate information entropy (ideal value for 8-bit image is 8)"""
    if len(image.shape) == 3:
        entropies = []
        for i in range(3):
            hist = np.histogram(image[:, :, i].flatten(), bins=256, range=(0, 256))[0]
            hist = hist[hist > 0] / hist.sum()
            entropies.append(-np.sum(hist * np.log2(hist)))
        return np.mean(entropies)
    else:
        hist = np.histogram(image.flatten(), bins=256, range=(0, 256))[0]
        hist = hist[hist > 0] / hist.sum()
        return -np.sum(hist * np.log2(hist))

def calculate_correlation(image, direction='horizontal'):
    """Calculate correlation coefficient of adjacent pixels (ideal ≈ 0)"""
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2).astype(np.uint8)
    else:
        gray = image
    h, w = gray.shape
    if direction == 'horizontal':
        pairs = [(gray[i, j], gray[i, j + 1]) for i in range(h) for j in range(w - 1)]
    elif direction == 'vertical':
        pairs = [(gray[i, j], gray[i + 1, j]) for i in range(h - 1) for j in range(w)]
    elif direction == 'diagonal':
        pairs = [(gray[i, j], gray[i + 1, j + 1]) for i in range(h - 1) for j in range(w - 1)]
    else:
        raise ValueError("direction must be 'horizontal', 'vertical', or 'diagonal'")
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    cov = np.cov(x, y)[0, 1]
    std_x = np.std(x)
    std_y = np.std(y)
    if std_x == 0 or std_y == 0:
        return 0
    return cov / (std_x * std_y)

def calculate_npcruaci(original, encrypted1, encrypted2):
    """Calculate NPCR and UACI (resistance to differential attacks)"""
    if len(original.shape) == 3:
        npcr_sum, uaci_sum = 0, 0
        for i in range(3):
            npcr, uaci = _calculate_npcruaci_channel(encrypted1[:, :, i], encrypted2[:, :, i])
            npcr_sum += npcr
            uaci_sum += uaci
        return npcr_sum / 3, uaci_sum / 3
    else:
        return _calculate_npcruaci_channel(encrypted1, encrypted2)

def _calculate_npcruaci_channel(img1, img2):
    """Single channel NPCR/UACI computation"""
    h, w = img1.shape
    d = (img1 != img2).astype(np.float32)
    npcr = np.sum(d) / (h * w) * 100
    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
    uaci = np.sum(diff) / (h * w * 255) * 100
    return npcr, uaci

def calculate_histogram_variance(image):
    """Calculate histogram variance (lower value indicates more uniform)"""
    if len(image.shape) == 3:
        variances = []
        for i in range(3):
            hist = np.histogram(image[:, :, i].flatten(), bins=256)[0]
            variances.append(np.var(hist))
        return np.mean(variances)
    else:
        hist = np.histogram(image.flatten(), bins=256)[0]
        return np.var(hist)

def calculate_chi_square(image):
    """Chi-square test for histogram uniformity (p > 0.05 indicates uniformity)"""
    if len(image.shape) == 3:
        chi2s = []
        for i in range(3):
            hist = np.histogram(image[:, :, i].flatten(), bins=256)[0]
            expected = np.mean(hist)
            chi2_stat = np.sum((hist - expected) ** 2 / expected)
            chi2s.append(chi2_stat)
        chi2_stat = np.mean(chi2s)
    else:
        hist = np.histogram(image.flatten(), bins=256)[0]
        expected = np.mean(hist)
        chi2_stat = np.sum((hist - expected) ** 2 / expected)
    p_value = 1 - chi2.cdf(chi2_stat, 255)
    return chi2_stat, p_value

def calculate_psnr_ssim(original, decrypted):
    """Calculate PSNR and SSIM (decryption quality)"""
    if len(original.shape) == 3:
        orig_gray = np.mean(original, axis=2).astype(np.uint8)
        dec_gray = np.mean(decrypted, axis=2).astype(np.uint8)
    else:
        orig_gray = original
        dec_gray = decrypted
    mse = np.mean((orig_gray - dec_gray) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10(255 ** 2 / mse)
    mu_x = np.mean(orig_gray)
    mu_y = np.mean(dec_gray)
    sigma_x = np.std(orig_gray)
    sigma_y = np.std(dec_gray)
    sigma_xy = np.mean((orig_gray - mu_x) * (dec_gray - mu_y))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim /= (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    return psnr, ssim

def plot_histograms(original, encrypted, save_path="histograms.png"):
    """Plot histograms of original and encrypted images for comparison"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    if len(original.shape) == 3:
        colors = ['red', 'green', 'blue']
        titles = ['Red Channel', 'Green Channel', 'Blue Channel']
        for i in range(3):
            axes[0, i].hist(original[:, :, i].flatten(), bins=256, color=colors[i], alpha=0.7)
            axes[0, i].set_title(f'Original - {titles[i]}')
            axes[0, i].set_xlim(0, 255)
            axes[1, i].hist(encrypted[:, :, i].flatten(), bins=256, color=colors[i], alpha=0.7)
            axes[1, i].set_title(f'Encrypted - {titles[i]}')
            axes[1, i].set_xlim(0, 255)
        axes[0, 3].imshow(original)
        axes[0, 3].set_title('Original Image')
        axes[0, 3].axis('off')
        axes[1, 3].imshow(encrypted)
        axes[1, 3].set_title('Encrypted Image')
        axes[1, 3].axis('off')
    else:
        axes[0, 0].hist(original.flatten(), bins=256, color='gray', alpha=0.7)
        axes[0, 0].set_title('Original Histogram')
        axes[0, 0].set_xlim(0, 255)
        axes[1, 0].hist(encrypted.flatten(), bins=256, color='gray', alpha=0.7)
        axes[1, 0].set_title('Encrypted Histogram')
        axes[1, 0].set_xlim(0, 255)
        axes[0, 1].imshow(original, cmap='gray')
        axes[0, 1].set_title('Original Image')
        axes[0, 1].axis('off')
        axes[1, 1].imshow(encrypted, cmap='gray')
        axes[1, 1].set_title('Encrypted Image')
        axes[1, 1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def evaluate_encryption_scheme(image_path, key, s_box, output_dir="results"):
    """Comprehensive evaluation of the encryption scheme, generating all metrics and plots for paper."""
    os.makedirs(output_dir, exist_ok=True)
    original = Image.open(image_path).convert('RGB')
    original_array = np.array(original)

    print("=" * 60)
    print("Performance Evaluation of Image Encryption Scheme")
    print("=" * 60)
    print(f"Test image: {image_path}")
    print(f"Image size: {original_array.shape}")

    # Encryption time
    start_time = time.time()
    encrypt_image(image_path, key, s_box, f"{output_dir}/encrypted.png", rounds=3)
    encrypt_time = time.time() - start_time

    # Decryption time
    start_time = time.time()
    decrypt_image(f"{output_dir}/encrypted.png", key, s_box, f"{output_dir}/decrypted.png", rounds=3)
    decrypt_time = time.time() - start_time

    encrypted = np.array(Image.open(f"{output_dir}/encrypted.png").convert('RGB'))
    decrypted = np.array(Image.open(f"{output_dir}/decrypted.png").convert('RGB'))

    # Basic security metrics
    print("\n[Basic Security Metrics]")
    ent_orig = calculate_entropy(original_array)
    ent_enc = calculate_entropy(encrypted)
    print(f"Entropy - Original: {ent_orig:.4f} | Encrypted: {ent_enc:.4f} (ideal ≈ 8)")

    print("\nAdjacent pixel correlation coefficients (ideal ≈ 0):")
    for direction in ['horizontal', 'vertical', 'diagonal']:
        corr_orig = calculate_correlation(original_array, direction)
        corr_enc = calculate_correlation(encrypted, direction)
        print(f"  {direction:10s}: Original {corr_orig:.4f} | Encrypted {corr_enc:.4f}")

    hist_var = calculate_histogram_variance(encrypted)
    chi2_stat, p_value = calculate_chi_square(encrypted)
    print(f"\nHistogram variance: {hist_var:.2f} (lower is better)")
    print(f"Chi-square test: χ²={chi2_stat:.2f}, p={p_value:.4f} (p>0.05 indicates uniformity)")

    # Differential attack analysis
    print("\n[Differential Attack Analysis]")
    modified = original_array.copy()
    modified[0, 0] = [255 - modified[0, 0, 0], 255 - modified[0, 0, 1], 255 - modified[0, 0, 2]]
    Image.fromarray(modified).save(f"{output_dir}/modified.png")
    encrypt_image(f"{output_dir}/modified.png", key, s_box, f"{output_dir}/encrypted_modified.png", rounds=3)
    encrypted_modified = np.array(Image.open(f"{output_dir}/encrypted_modified.png").convert('RGB'))
    npcr, uaci = calculate_npcruaci(original_array, encrypted, encrypted_modified)
    print(f"NPCR: {npcr:.4f}% (ideal ≥ 99.6%)")
    print(f"UACI: {uaci:.4f}% (ideal ≈ 33.46%)")

    # Key sensitivity
    print("\n[Key Sensitivity Analysis]")
    key2 = (key[0] + 1e-10, key[1], key[2])
    encrypt_image(image_path, key2, s_box, f"{output_dir}/encrypted_key2.png", rounds=3)
    encrypted_key2 = np.array(Image.open(f"{output_dir}/encrypted_key2.png").convert('RGB'))
    diff_ratio = np.sum(encrypted != encrypted_key2) / encrypted.size * 100
    print(f"Difference ratio due to slight key change: {diff_ratio:.4f}% (ideal ≈ 99.6%)")

    # Decryption quality
    print("\n[Decryption Quality]")
    psnr, ssim = calculate_psnr_ssim(original_array, decrypted)
    print(f"PSNR: {psnr:.2f} dB (∞ indicates perfect reconstruction)")
    print(f"SSIM: {ssim:.4f} (1 indicates perfect reconstruction)")

    # Key space
    print("\n[Key Space]")
    key_space = 10 ** 45
    key_space_bits = math.log10(key_space) / math.log10(2)
    print(f"Key space size: {key_space:.2e} (≈ 2^{key_space_bits:.1f})")
    print(f"Requirement: ≥ 2^100, {'Satisfied' if key_space_bits >= 100 else 'Not satisfied'}")

    # Efficiency
    print("\n[Efficiency]")
    print(f"Encryption time: {encrypt_time:.4f} seconds")
    print(f"Decryption time: {decrypt_time:.4f} seconds")
    img_size_mb = (original_array.nbytes) / (1024 * 1024)
    throughput = img_size_mb / encrypt_time
    print(f"Image size: {img_size_mb:.2f} MB")
    print(f"Encryption throughput: {throughput:.2f} MB/s")

    plot_histograms(original_array, encrypted, f"{output_dir}/histograms.png")
    print(f"\nHistograms saved to: {output_dir}/histograms.png")

    with open(f"{output_dir}/results_summary.txt", "w") as f:
        f.write("Performance Evaluation of Image Encryption Scheme\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test image: {image_path}\n")
        f.write(f"Image size: {original_array.shape}\n\n")
        f.write("[Basic Security Metrics]\n")
        f.write(f"Entropy: {ent_enc:.4f} (ideal ≈ 8)\n")
        f.write(f"Histogram variance: {hist_var:.2f}\n")
        f.write(f"Chi-square p-value: {p_value:.4f}\n\n")
        f.write("Correlation coefficients:\n")
        for direction in ['horizontal', 'vertical', 'diagonal']:
            corr = calculate_correlation(encrypted, direction)
            f.write(f"  {direction}: {corr:.4f}\n")
        f.write("\n")
        f.write("[Differential Attack]\n")
        f.write(f"NPCR: {npcr:.4f}%\n")
        f.write(f"UACI: {uaci:.4f}%\n\n")
        f.write("[Key Analysis]\n")
        f.write(f"Key space: 2^{key_space_bits:.1f}\n")
        f.write(f"Key sensitivity: {diff_ratio:.4f}%\n\n")
        f.write("[Efficiency]\n")
        f.write(f"Encryption time: {encrypt_time:.4f} s\n")
        f.write(f"Decryption time: {decrypt_time:.4f} s\n")
        f.write(f"Throughput: {throughput:.2f} MB/s\n")

    print(f"\nFull results saved to: {output_dir}/results_summary.txt")
    print("=" * 60)

    return {
        "entropy": ent_enc,
        "correlations": {
            "horizontal": calculate_correlation(encrypted, 'horizontal'),
            "vertical": calculate_correlation(encrypted, 'vertical'),
            "diagonal": calculate_correlation(encrypted, 'diagonal')
        },
        "npcr": npcr,
        "uaci": uaci,
        "histogram_variance": hist_var,
        "chi_square_p": p_value,
        "key_sensitivity": diff_ratio,
        "encrypt_time": encrypt_time,
        "decrypt_time": decrypt_time,
        "throughput": throughput
    }


# ==================== Usage Example ====================
if __name__ == "__main__":
    # Generate an example S-box (replace with your optimized S-box)
    rng = np.random.RandomState(42)
    s_box_example = rng.permutation(256).tolist()

    # Key
    key = (1.0, 1.0, 1.0)

    # If no test image exists in current directory, load from skimage or generate random
    try:
        from skimage import data
        test_img = data.camera()
        test_img_rgb = np.stack([test_img, test_img, test_img], axis=2)
        Image.fromarray(test_img_rgb).save("cameraman.png")
        image_path = "cameraman.png"
        print("Using 'cameraman' test image from skimage library")
    except ImportError:
        test_img = Image.fromarray(np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8))
        test_img.save("test.png")
        image_path = "test.png"
        print("skimage library not available, using random test image")
    except Exception as e:
        print(f"Error loading test image: {e}")
        test_img = Image.fromarray(np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8))
        test_img.save("test.png")
        image_path = "test.png"
        print("Using random test image")

    # Run full evaluation
    results = evaluate_encryption_scheme(image_path, key, s_box_example, "paper_results")