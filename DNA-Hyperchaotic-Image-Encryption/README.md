```
# DNA Base‑Pairing Driven Dynamic Encoding and Neural‑Assisted Hyperchaotic Image Encryption

**Project repository for the paper:**  
*DNA Base‑Pairing Driven Dynamic Encoding and Neural‑Assisted Hyperchaotic Image Encryption*  
Authors: Yuning Wang, Meiyun Gui (Co‑first Authors)

---

## 1. Introduction

This repository contains the full implementation of the image encryption scheme proposed in our paper. The algorithm integrates a Lorenz‑like hyperchaotic system, byte‑level dynamic DNA encoding adaptively selected by chaotic sequences, a neural‑network‑assisted surrogate fitness model for S‑box optimization, chaotic discrete particle swarm optimization (PSO), bidirectional feedback XOR diffusion with reversible cellular automata (Rule 90), and multi‑purpose multiplexing of chaotic sequences. The code supports grayscale image encryption/decryption, GPU‑accelerated cryptographic metric evaluation, and comprehensive experimental analysis (NPCR, UACI, histogram, correlation, key sensitivity, cropping attacks, etc.).

---

## 2. Main Features

- Hyperchaotic keystream generation based on a 3D Lorenz‑like system  
- Byte‑level dynamic DNA encoding with 8 rule types selected adaptively by chaotic sequences  
- Neural‑network‑assisted S‑box optimization (surrogate fitness model + chaotic discrete PSO)  
- Bidirectional feedback XOR diffusion combined with reversible cellular automaton (Rule 90)  
- GPU acceleration (via PyTorch) for cryptographic metrics and batch processing (tested on NVIDIA RTX 4090D)  
- Graphical user interface (UI) for interactive encryption/decryption  
- Supporting experimental scripts to reproduce all figures and tables in the paper

---

## 3. File Structure
```



.
├── .idea/ # IDE configuration (ignorable)
├── .venv/ # Python virtual environment (optional)
├── paper_plots/ # Generated figures (histograms, scatter plots, etc.)
├── paper_results/ # Experimental metric results (NPCR, UACI, etc.)
├── temp_image/ # Temporary images (intermediate encryption/decryption)
├── temp_eval/ # Temporary evaluation files (e.g., difference distribution caches)
│
├── cameraman.png # Standard test image (256×256 grayscale)
├── cameraman_encrypted.png # Example encrypted image
├── main.py # Core encryption/decryption algorithm (command‑line interface)
├── ui.py # Graphical user interface (tkinter / PyQt based)
├── bbox_searching.py # S‑box search & optimization (PSO + neural surrogate)
├── optimized_bbox.txt # The final optimized S‑box (256 bytes)
├── Supporting experimental.py # Auxiliary experimental script (metrics, attacks, plotting)
└── README.md # This file

text

```
- `main.py` – Core encryption/decryption functions. Supports command‑line arguments.  
- `ui.py` – GUI for easy drag‑and‑drop or file‑based encryption/decryption.  
- `bbox_searching.py` – Implements chaotic discrete PSO and the neural surrogate model for S‑box optimization.  
- `optimized_bbox.txt` – The optimized S‑box (256 bytes) used by the encryption scheme.  
- `Supporting experimental.py` – Reproduces all experiments from the paper: NPCR, UACI, histograms, correlation, cropping attacks, key sensitivity, etc. It saves figures to `paper_plots/` and numerical results to `paper_results/`.

---

## 4. Installation & Dependencies

- **Python 3.8+** required.  
- It is recommended to create a virtual environment (e.g., `.venv`).  
- Install the required packages with pip:

```bash
pip install numpy pillow scipy matplotlib scikit-image torch torchvision
```



- For GPU acceleration (strongly recommended for training the surrogate model and for batch metric evaluation), install a CUDA‑compatible version of PyTorch. The code has been tested on an **NVIDIA RTX 4090D**.

------

## 5. Usage

### 5.1 Graphical User Interface

Launch the interactive encryption/decryption window:

bash

```
python ui.py
```



You can load an image (e.g., `cameraman.png`), set the hyperchaotic key (default `(1.0, 1.0, 1.0)`), and press **Encrypt** or **Decrypt**. The result is displayed and automatically saved in the working directory.

### 5.2 Command‑Line Interface

`main.py` supports direct command‑line usage:

bash

```
# Encrypt an image
python main.py --mode encrypt --input cameraman.png --output encrypted.png --key 1.0 1.0 1.0

# Decrypt an image
python main.py --mode decrypt --input encrypted.png --output decrypted.png --key 1.0 1.0 1.0
```



For a full list of parameters, run `python main.py --help`.

### 5.3 Reproducing Experimental Results

To generate all figures and tables from **Section IV** of the paper, run:

bash

```
python "Supporting experimental.py"
```



This script will:

- Load the optimized S‑box from `optimized_bbox.txt` (if missing, it automatically calls `bbox_searching.py` to generate a new one).
- Perform NPCR/UACI tests, histogram analysis, correlation scatter plots, cropping attack simulation, and key sensitivity tests.
- Save all plots into `paper_plots/` and numerical results into `paper_results/`.
- Print summary tables similar to Tables 4‑1, 4‑2, 4‑3, 4‑4 in the paper.

------

## 6. Key Parameters

| Parameter                   | Default value                    | Description                                                  |
| :-------------------------- | :------------------------------- | :----------------------------------------------------------- |
| Lorenz initial condition    | `(x0, y0, z0) = (1.0, 1.0, 1.0)` | Secret key for the hyperchaotic system                       |
| Integration step            | `0.01`                           | Step size for numerical integration                          |
| Transient iterations        | `1000`                           | Number of initial iterations discarded to enhance randomness |
| DNA encoding rules          | 8 types (chaotically selected)   | Rule index obtained from Eq. (14) in the paper               |
| Number of encryption rounds | `3`                              | Iterations of the confusion‑diffusion process                |
| S‑box                       | `optimized_bbox.txt`             | Nonlinear substitution table (256‑byte bijection)            |

------

## 7. Hardware & Performance Notes

The code is optimized for GPU acceleration using PyTorch. On an **NVIDIA RTX 4090D**, typical runtime for:

- S‑box optimization (200 iterations, 150 particles): **~12 minutes**
- Batch evaluation of 500 S‑box candidates: **<2 seconds**
- Full encryption of a 256×256 image (3 rounds): **~0.3 seconds**

CPU fallback is available but not recommended for large‑scale experiments.

------

## 8. Citation

If you find this work useful for your research, please cite our paper:

> Y. Wang and M. Gui, “DNA Base‑Pairing Driven Dynamic Encoding and Neural‑Assisted Hyperchaotic Image Encryption,” *IEEE Transactions on* [Journal Name], vol. X, no. Y, pp. 1‑9, 2026.

------

## 9. Contact

For questions, suggestions, or collaboration, please contact the corresponding author:

**Yuning Wang** – wyn@mails.guet.edu.cn

------

## 10. License

This project is provided for academic and research purposes. Please refer to the paper for detailed licensing information.