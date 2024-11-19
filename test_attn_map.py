import matplotlib.pyplot as plt
import seaborn as sns  # Optional, but enhances the plot aesthetics
import torch
import numpy as np

# Assuming `attn` is your 14x14 tensor
# Convert it to a numpy array if it's a PyTorch tensor
# attn = attn.numpy()
attn = torch.randn(196)
attn = attn - attn.amax(dim=-1, keepdim=True)
attn = attn.softmax(dim=-1)
attn = attn.reshape(14, 14).numpy()

plt.figure(figsize=(8, 8))
sns.heatmap(attn, cmap="viridis", square=True, cbar=False,
            xticklabels=False, yticklabels=False)  # You can adjust `fmt` for decimal points
plt.title("14x14 Attention Heatmap")
plt.savefig("attention_heatmap.png", dpi=300, bbox_inches="tight")
# plt.show()
