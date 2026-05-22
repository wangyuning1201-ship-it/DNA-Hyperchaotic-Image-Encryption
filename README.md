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
