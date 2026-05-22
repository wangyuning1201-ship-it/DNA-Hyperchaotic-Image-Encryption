import numpy as np
from PIL import Image
import os
import random
import math
import matplotlib.pyplot as plt

# ==================== 1. Hyperchaotic System (Lorenz) ====================
def generate_chaos(init, length, dt=0.01, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
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
    rules = [
        [0, 1, 2, 3], [0, 2, 1, 3], [0, 1, 3, 2], [0, 3, 1, 2],
        [0, 2, 3, 1], [0, 3, 2, 1], [1, 0, 2, 3], [1, 0, 3, 2],
    ]
    enc_tables, dec_tables = [], []
    for rule in rules:
        enc = np.zeros(256, dtype=np.uint8)
        for b in range(256):
            g1 = (b >> 6) & 0x03
            g2 = (b >> 4) & 0x03
            g3 = (b >> 2) & 0x03
            g4 = b & 0x03
            nb = (rule[g1] << 6) | (rule[g2] << 4) | (rule[g3] << 2) | rule[g4]
            enc[b] = nb
        dec = np.zeros(256, dtype=np.uint8)
        for b in range(256):
            dec[enc[b]] = b
        enc_tables.append(enc)
        dec_tables.append(dec)
    return enc_tables, dec_tables

# ==================== 3. Reversible Cellular Automaton (Rule 90) ====================
class CellularAutomaton:
    def __init__(self, rule=90):
        self.rule = rule
        if rule != 90:
            raise ValueError("Only rule 90 is supported currently")

    def forward(self, data):
        n = len(data)
        if n == 0:
            return data.copy()
        result = np.zeros(n, dtype=np.uint8)
        if n == 1:
            return result
        result[0] = 0 ^ data[1]
        for i in range(1, n - 1):
            result[i] = data[i - 1] ^ data[i + 1]
        result[n - 1] = data[n - 2] ^ 0
        return result

    def backward(self, data):
        n = len(data)
        if n == 0:
            return data.copy()
        if n == 1:
            return data.copy()
        a = np.zeros(n, dtype=np.uint8)
        b = np.zeros(n, dtype=bool)
        a[0], b[0] = 0, True
        a[1], b[1] = data[0], False
        for i in range(2, n):
            a[i] = data[i - 1] ^ a[i - 2]
            b[i] = b[i - 2]
        if b[n - 2]:
            x0 = data[n - 1] ^ a[n - 2]
        else:
            if a[n - 2] != data[n - 1]:
                raise ValueError("Cellular automaton is not invertible for this data (singular matrix)")
            x0 = 0
        result = np.zeros(n, dtype=np.uint8)
        result[0] = x0
        if n > 1:
            result[1] = a[1] ^ (x0 if b[1] else 0)
        for i in range(2, n):
            result[i] = a[i] ^ (x0 if b[i] else 0)
        return result

# ==================== 4. Encryption and Decryption with IV ====================
def encrypt_image(image_path, key, s_box, output_path, rounds=3):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    original_shape = img_array.shape
    flat = img_array.flatten().astype(np.uint8)
    N = len(flat)

    iv = random.randint(0, 255)
    perturbed_key = (key[0] + iv * 1e-8, key[1], key[2])

    chaos_len = N + 1000
    X, Y, Z = generate_chaos(perturbed_key, chaos_len)
    X, Y, Z = X[1000:], Y[1000:], Z[1000:]

    rules = (np.abs(X[:N]) * 1e8).astype(np.uint64) % 8
    rules = rules.astype(np.uint8)
    perm_indices = np.argsort(X[:N])

    chaos_xor = (np.abs(Y[:N]) * 1e8).astype(np.uint64) % 256
    chaos_xor = chaos_xor.astype(np.uint8)

    enc_tables, _ = generate_dna_tables()

    data = flat.copy()
    for _ in range(rounds):
        data = np.array([s_box[b] for b in data], dtype=np.uint8)
        data = data[perm_indices]
        data = np.array([enc_tables[rules[i]][data[i]] for i in range(N)], dtype=np.uint8)
        data ^= chaos_xor
        ca = CellularAutomaton(rule=90)
        data = ca.forward(data)
        for i in range(1, N):
            data[i] ^= data[i - 1]
        for i in range(N - 2, -1, -1):
            data[i] ^= data[i + 1]

    enc_img_array = data.reshape(original_shape)
    enc_img_array[0, 0, 0] = iv
    enc_img = Image.fromarray(enc_img_array, 'RGB')
    enc_img.save(output_path, format='PNG')
    return enc_img_array

def decrypt_image(encrypted_path, key, s_box, output_path, rounds=3):
    img = Image.open(encrypted_path).convert('RGB')
    img_array = np.array(img)
    original_shape = img_array.shape
    flat = img_array.flatten().astype(np.uint8)
    N = len(flat)

    iv = flat[0]
    perturbed_key = (key[0] + iv * 1e-8, key[1], key[2])

    chaos_len = N + 1000
    X, Y, Z = generate_chaos(perturbed_key, chaos_len)
    X, Y, Z = X[1000:], Y[1000:], Z[1000:]

    rules = (np.abs(X[:N]) * 1e8).astype(np.uint64) % 8
    rules = rules.astype(np.uint8)
    perm_indices = np.argsort(X[:N])
    inv_perm = np.argsort(perm_indices)

    chaos_xor = (np.abs(Y[:N]) * 1e8).astype(np.uint64) % 256
    chaos_xor = chaos_xor.astype(np.uint8)

    inv_s_box = [0] * 256
    for i, val in enumerate(s_box):
        inv_s_box[val] = i

    _, dec_tables = generate_dna_tables()

    data = flat.copy()
    for _ in range(rounds):
        for i in range(0, N - 1):
            data[i] ^= data[i + 1]
        for i in range(N - 1, 0, -1):
            data[i] ^= data[i - 1]
        ca = CellularAutomaton(rule=90)
        data = ca.backward(data)
        data ^= chaos_xor
        data = np.array([dec_tables[rules[i]][data[i]] for i in range(N)], dtype=np.uint8)
        data = data[inv_perm]
        data = np.array([inv_s_box[b] for b in data], dtype=np.uint8)

    dec_img_array = data.reshape(original_shape)
    dec_img = Image.fromarray(dec_img_array, 'RGB')
    dec_img.save(output_path, format='PNG')
    return dec_img_array

# ==================== Helpers ====================
def extract_pairs(image, direction):
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
        raise ValueError("Invalid direction")
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    return x, y

def correlation_coef(x, y):
    cov = np.cov(x, y)[0, 1]
    std_x = np.std(x)
    std_y = np.std(y)
    if std_x == 0 or std_y == 0:
        return 0
    return cov / (std_x * std_y)

# ==================== Experiment 1: Correlation Scatter Plots ====================
def plot_correlation_scatter(original_image_path, key, s_box, output_path="correlation_scatter.png", sample_size=3000):
    plain_img = np.array(Image.open(original_image_path).convert('RGB'))
    enc_img = encrypt_image(original_image_path, key, s_box, "temp_enc.png", rounds=3)

    directions = ['horizontal', 'vertical', 'diagonal']
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for col, direction in enumerate(directions):
        x_plain, y_plain = extract_pairs(plain_img, direction)
        x_enc, y_enc = extract_pairs(enc_img, direction)

        if len(x_plain) > sample_size:
            idx = np.random.choice(len(x_plain), sample_size, replace=False)
            x_plain, y_plain = x_plain[idx], y_plain[idx]
        if len(x_enc) > sample_size:
            idx = np.random.choice(len(x_enc), sample_size, replace=False)
            x_enc, y_enc = x_enc[idx], y_enc[idx]

        corr_plain = correlation_coef(x_plain, y_plain)
        corr_enc = correlation_coef(x_enc, y_enc)

        axes[0, col].scatter(x_plain, y_plain, s=1, alpha=0.4, c='blue')
        axes[0, col].set_xlim(0, 255)
        axes[0, col].set_ylim(0, 255)
        axes[0, col].set_xlabel('Pixel value at (i,j)')
        axes[0, col].set_ylabel('Pixel value at neighbor')
        axes[0, col].set_title(f'Plain {direction}\nr={corr_plain:.4f}')
        axes[0, col].set_aspect('equal')

        axes[1, col].scatter(x_enc, y_enc, s=1, alpha=0.4, c='red')
        axes[1, col].set_xlim(0, 255)
        axes[1, col].set_ylim(0, 255)
        axes[1, col].set_xlabel('Pixel value at (i,j)')
        axes[1, col].set_ylabel('Pixel value at neighbor')
        axes[1, col].set_title(f'Encrypted {direction}\nr={corr_enc:.4f}')
        axes[1, col].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Experiment 1] Correlation scatter plot saved to {output_path}")

# ==================== Experiment 2: Differential Attack ====================
def plot_differential_attack(original_image_path, key, s_box, output_path="differential_attack.png"):
    plain = Image.open(original_image_path).convert('RGB')
    plain_array = np.array(plain)

    modified_array = plain_array.copy()
    modified_array[0, 0, 0] = (int(modified_array[0, 0, 0]) + 1) % 256
    Image.fromarray(modified_array).save("temp_modified.png")

    enc_orig = encrypt_image(original_image_path, key, s_box, "temp_enc_orig.png", rounds=3)
    enc_mod = encrypt_image("temp_modified.png", key, s_box, "temp_enc_mod.png", rounds=3)

    diff = np.abs(enc_orig.astype(np.int16) - enc_mod.astype(np.int16))
    diff_vis = (diff * 5).clip(0, 255).astype(np.uint8)  # Fixed clipping

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].imshow(plain_array)
    axes[0, 0].set_title('Original Plain Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(modified_array)
    axes[0, 1].set_title('Modified Plain (1 pixel changed)')
    axes[0, 1].axis('off')

    zoom = plain_array[:10, :10, :].copy()
    zoom[0, 0, :] = modified_array[0, 0, :]
    axes[0, 2].imshow(zoom)
    axes[0, 2].set_title('Top-left 10x10 (modified pixel)')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(enc_orig)
    axes[1, 0].set_title('Ciphertext of Original')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(enc_mod)
    axes[1, 1].set_title('Ciphertext of Modified')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(diff_vis, cmap='hot')
    axes[1, 2].set_title('Difference Image (5x)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    def npcr_uaci(img1, img2):
        d = (img1 != img2)
        npcr = np.sum(d) / img1.size * 100
        diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
        uaci = np.sum(diff) / (img1.size * 255) * 100
        return npcr, uaci

    npcr, uaci = npcr_uaci(enc_orig, enc_mod)
    print(f"[Experiment 2] Differential attack plot saved to {output_path}")
    print(f"   NPCR = {npcr:.4f}%, UACI = {uaci:.4f}%")
    return npcr, uaci

# ==================== Experiment 3: Key Sensitivity ====================
def plot_key_sensitivity(original_image_path, key, s_box, output_path="key_sensitivity.png", delta=1e-10):
    plain_img = np.array(Image.open(original_image_path).convert('RGB'))
    key2 = (key[0] + delta, key[1], key[2])

    enc1 = encrypt_image(original_image_path, key, s_box, "temp_enc_key1.png", rounds=3)
    enc2 = encrypt_image(original_image_path, key2, s_box, "temp_enc_key2.png", rounds=3)

    diff_cipher = np.abs(enc1.astype(np.int16) - enc2.astype(np.int16))
    diff_vis = (diff_cipher * 5).clip(0, 255).astype(np.uint8)  # Fixed clipping

    wrong_dec = decrypt_image("temp_enc_key1.png", key2, s_box, "temp_dec_wrong.png", rounds=3)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0, 0].imshow(plain_img)
    axes[0, 0].set_title('Original Plain Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(enc1)
    axes[0, 1].set_title('Ciphertext (Key1)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(enc2)
    axes[0, 2].set_title('Ciphertext (Key2: x0+1e-10)')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(diff_vis, cmap='hot')
    axes[0, 3].set_title('Ciphertext Difference (5x)')
    axes[0, 3].axis('off')

    correct_dec = decrypt_image("temp_enc_key1.png", key, s_box, "temp_dec_correct.png", rounds=3)
    axes[1, 0].imshow(correct_dec)
    axes[1, 0].set_title('Correct Decryption')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(wrong_dec)
    axes[1, 1].set_title('Wrong Key Decryption')
    axes[1, 1].axis('off')

    axes[1, 2].hist(wrong_dec.flatten(), bins=256, color='gray', alpha=0.7)
    axes[1, 2].set_title('Histogram of Wrong Decryption')
    axes[1, 2].set_xlim(0, 255)

    diff_percent = np.sum(enc1 != enc2) / enc1.size * 100
    axes[1, 3].text(0.1, 0.5, f'Ciphertext difference:\n{diff_percent:.2f}%', fontsize=14)
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Experiment 3] Key sensitivity plot saved to {output_path}")
    print(f"   Ciphertext difference with delta={delta}: {diff_percent:.4f}%")
    return diff_percent

# ==================== Experiment 4: Cropping Attack (multiple ratios) ====================
def plot_cropping_attack(original_image_path, key, s_box, output_path="cropping_attack.png", crop_fractions=[0.0625, 0.25]):
    """
    Test multiple cropping fractions. Show only the 25% case in paper (or combine).
    """
    original = np.array(Image.open(original_image_path).convert('RGB'))
    results = []

    # Process each crop fraction
    for frac in crop_fractions:
        enc_img = encrypt_image(original_image_path, key, s_box, f"temp_enc_{frac}.png", rounds=3)
        h, w, _ = enc_img.shape
        crop_size = int(min(h, w) * np.sqrt(frac))
        start_y = (h - crop_size) // 2
        start_x = (w - crop_size) // 2

        damaged_enc = enc_img.copy()
        damaged_enc[start_y:start_y+crop_size, start_x:start_x+crop_size, :] = 0
        Image.fromarray(damaged_enc).save(f"temp_damaged_{frac}.png")

        damaged_dec = decrypt_image(f"temp_damaged_{frac}.png", key, s_box, f"temp_dec_{frac}.png", rounds=3)

        # PSNR and SSIM
        mse = np.mean((original.astype(np.float32) - damaged_dec.astype(np.float32)) ** 2)
        psnr = 20 * math.log10(255.0 / math.sqrt(mse)) if mse != 0 else float('inf')
        # Simple SSIM
        mu_x, mu_y = np.mean(original), np.mean(damaged_dec)
        var_x, var_y = np.var(original), np.var(damaged_dec)
        cov = np.mean((original - mu_x) * (damaged_dec - mu_y))
        c1, c2 = (0.01*255)**2, (0.03*255)**2
        ssim = (2*mu_x*mu_y + c1) * (2*cov + c2)
        ssim /= (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)

        results.append((frac, enc_img, damaged_enc, damaged_dec, psnr, ssim))
        print(f"   Crop {frac*100:.0f}% -> PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")

    # Plot only the 25% case (or make a combined figure)
    # For simplicity, use the second fraction (25%) if exists
    target = results[-1]  # last one is 25%
    frac, enc_img, damaged_enc, damaged_dec, psnr_val, ssim_val = target

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(enc_img)
    axes[0, 0].set_title('Original Ciphertext')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(damaged_enc)
    axes[0, 1].set_title(f'Damaged Ciphertext ({frac*100:.0f}% cropped)')
    axes[0, 1].axis('off')

    cipher_diff = np.abs(enc_img.astype(np.int16) - damaged_enc.astype(np.int16))
    cipher_diff_vis = (cipher_diff * 2).clip(0, 255).astype(np.uint8)
    axes[0, 2].imshow(cipher_diff_vis, cmap='hot')
    axes[0, 2].set_title('Ciphertext Difference (2x)')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(original)
    axes[1, 0].set_title('Original Plain Image')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(damaged_dec)
    axes[1, 1].set_title('Decrypted from Damaged')
    axes[1, 1].axis('off')

    rec_diff = np.abs(original.astype(np.int16) - damaged_dec.astype(np.int16))
    rec_diff_vis = (rec_diff * 2).clip(0, 255).astype(np.uint8)
    axes[1, 2].imshow(rec_diff_vis, cmap='hot')
    axes[1, 2].set_title(f'Recovery Difference\nPSNR={psnr_val:.2f} dB, SSIM={ssim_val:.3f}')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Experiment 4] Cropping attack plot saved to {output_path}")

    # Also create a summary table plot
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    table_data = [["Cropping Fraction", "PSNR (dB)", "SSIM"]]
    for frac, _, _, _, p, s in results:
        table_data.append([f"{frac*100:.0f}%", f"{p:.2f}", f"{s:.4f}"])
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    plt.savefig("paper_plots/cropping_table.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   Summary table saved to paper_plots/cropping_table.png")
    return results

# ==================== Main ====================
if __name__ == "__main__":
    rng = np.random.RandomState(42)
    s_box_example = rng.permutation(256).tolist()
    key = (1.0, 1.0, 1.0)

    try:
        from skimage import data
        test_img = data.camera()
        test_img_rgb = np.stack([test_img, test_img, test_img], axis=2)
        Image.fromarray(test_img_rgb).save("cameraman.png")
        image_path = "cameraman.png"
        print("Using 'cameraman' test image from skimage.")
    except Exception:
        test_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        Image.fromarray(test_img).save("test.png")
        image_path = "test.png"
        print("skimage not available; using random test image.")

    os.makedirs("paper_plots", exist_ok=True)

    plot_correlation_scatter(image_path, key, s_box_example,
                             output_path="paper_plots/correlation_scatter.png")
    plot_differential_attack(image_path, key, s_box_example,
                             output_path="paper_plots/differential_attack.png")
    plot_key_sensitivity(image_path, key, s_box_example,
                         output_path="paper_plots/key_sensitivity.png")
    plot_cropping_attack(image_path, key, s_box_example,
                         output_path="paper_plots/cropping_attack.png",
                         crop_fractions=[0.0625, 0.25])

    print("\nAll experiments completed. Plots saved in './paper_plots/'")