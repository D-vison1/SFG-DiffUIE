# SFG-DiffUIE: Spatial-Frequency Fusion-Guided Diffusion Model for Underwater Image Enhancement
This project focuses on the research of underwater image restoration.

This is the repository for our paper. [Paper Link](https://doi.org/10.1016/j.patcog.2026.115064)

## Train
Run `train.py` to start training.

## Test
Set the checkpoint path in `config/config.yml`. Run `test.py`.

## Citation
If you use the code in this repository for your work, please cite our paper:
```bibtex
@article{WANG2027115064,
title = {SFG-DiffUIE: Spatial-frequency fusion-guided diffusion model for underwater image enhancement},
journal = {Pattern Recognition},
volume = {183},
pages = {115064},
year = {2027},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2026.115064},
url = {https://www.sciencedirect.com/science/article/pii/S0031320326020285},
author = {Yingbo Wang and Bofeng Duan and Yuhao Gong and Tongfei Liu and Xiaogang Du and Wan Li and Tao Lei and Asoke K. Nandi},
keywords = {Underwater image enhancement, Conditional diffusion model, Wavelet transform, Frequency-domain convolution, Attention mechanism},
abstract = {Underwater optical imaging suffers from light absorption and scattering, which cause low contrast, color distortion, blurred details, and uneven illumination. These degradations hinder deep-sea exploration and resource development. Although diffusion-based enhancement methods have advanced color correction and illumination recovery in the spatial domain, they often neglect high-frequency texture cues in the frequency domain, leading to incomplete restoration and visual artifacts. To address this limitation, we propose SFG-DiffUIE, a spatial-frequency fusion-guided diffusion model for underwater image enhancement. The framework integrates two core networks: a Wavelet Enhancement Network (WE-Net) and a Spatial-Frequency Guided Denoising Network (SFGD-Net). WE-Net employs a dual-branch architecture to enhance high- and low-frequency components separately in the wavelet domain using multi-scale convolutions and orientation-aware attention. SFGD-Net then refines the output by fusing spatial and frequency information, leveraging high-frequency guidance to recover fine details through iterative denoising. Experiments on three public benchmarks indicate that SFG-DiffUIE achieves competitive performance relative to existing methods, notably under challenging degradation conditions. Furthermore, cross-dataset evaluations on four additional datasets suggest promising generalization across diverse underwater scenarios and cross-domain image enhancement tasks. Our code will be released at: https://github.com/D-vison1/SFG-DiffUIE.}
}
