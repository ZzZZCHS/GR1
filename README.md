# Grounded Policy Network

A replicated version of [GR1](https://arxiv.org/abs/2312.13139) with added mask input. 

## Setup
Please follow the main [setup instruction](https://github.com/ZzZZCHS/RoboGround?tab=readme-ov-file#-environment-setup) to prepare RoboGround's Conda environment.

```bash
conda activate roboground

# install the additional dependency
pip install git+https://github.com/openai/CLIP.git
```

## Training

Training the model takes approximately 7 days using 8 NVIDIA RTX 4090 GPUs.

Simply start training with:
```bash
bash scripts/train.sh
```

## Inference

