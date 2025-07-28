# Dual Encoders for Diffusion-based Image Inpainting

## Paper

The paper describing this work is available at: [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10890576)

## Installation

```bash
conda create -n inpaint python=3.10
conda activate inpaint
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -e ./src/diffusers
pip install -r ./requirements.txt
```

## Evaluation

- Download BrushBench and EditBench datasets by following the instructions at: [TencentARC/BrushNet](https://github.com/TencentARC/BrushNet)
- To evaluate, run:

```bash
python evaluate.py
```

> **Note:**
> - The provided checkpoints have been adjusted for different mask ratios to improve performance on general inpainting tasks.
> - If you want to reproduce the results from the original paper, manually adjust the mask ratios in `train.py` (lines 745-780) and retrain the model.
> - In our reproduction, the masks generated in `evaluate.py` are consistent with the original paper, achieving PSNR 26.84 on BrushBench (normal setting) and PSNR 28.52 on BrushBench (pixel setting).

## Training

- Download the BrushData dataset for training by following the instructions at: [TencentARC/BrushNet](https://github.com/TencentARC/BrushNet)
- Run `accelerate config` before training.
- Edit `train.sh` as needed for your environment.
- Start training with:

```bash
./train.sh
```