## Instruction
### Install
```
conda create -n gr1 python=3.10
conda activate gr1
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/openai/CLIP.git
pip install omegaconf==2.3.0 timm numpy==1.23.1 einops-exts wandb transformers pyrender moviepy roboticstoolbox-python chardet scipy==1.10.1 setuptools==57.5.0 networkx==2.5 h5py
pip install opencv-python
# pytorch3d

add `from collections.abc import *` in collections/__init__.py

```

### Path setup
ln -s /mnt/hwfile/OpenRobotLab/chenyilun/pretrain/ ./

