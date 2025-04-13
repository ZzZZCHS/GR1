import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn
import matplotlib.pyplot as plt
import seaborn as sns


def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, mask=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale
        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        
        if mask is not None:
            d = mask.shape[-1]
            perceiver_num = sim.shape[-2] // 3
            mask1 = mask[:, 0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            mask2 = mask[:, 1].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            # sim[..., perceiver_num:perceiver_num*2, :d].masked_fill_(mask1, 0)
            # sim[..., perceiver_num*2:, :d].masked_fill_(mask2, 0)
            
            # sim[..., perceiver_num:perceiver_num*2, :d] += mask1
            # sim[..., perceiver_num*2:, :d] += mask2
            
        
        # tmp_sim = sim[0, :, 0, perceiver_num:perceiver_num*2, :196].mean(dim=0)
        # for i in range(tmp_sim.shape[0]):
        #     t_sim = tmp_sim[i].softmax(dim=-1)
        #     tmp_attn = t_sim.reshape(14, 14).detach().cpu().numpy()
        #     plt.figure(figsize=(8, 8))
        #     sns.heatmap(tmp_attn, cmap="viridis", square=True, cbar=False, xticklabels=False, yticklabels=False)
        #     plt.title("14x14 Attention Heatmap")
        #     plt.savefig(f"object_query_{i}.png", dpi=300, bbox_inches="tight")
        #     plt.clf()
        # tmp_sim = sim[0, :, 0, :perceiver_num, :196].mean(dim=0)
        # for i in range(tmp_sim.shape[0]):
        #     t_sim = tmp_sim[i].softmax(dim=-1)
        #     tmp_attn = t_sim.reshape(14, 14).detach().cpu().numpy()
        #     plt.figure(figsize=(8, 8))
        #     sns.heatmap(tmp_attn, cmap="viridis", square=True, cbar=False, xticklabels=False, yticklabels=False)
        #     plt.title("14x14 Attention Heatmap")
        #     plt.savefig(f"global_query_{i}.png", dpi=300, bbox_inches="tight")
        #     plt.clf()
        # tmp_sim = sim[0, :, 0, perceiver_num*2:, :196].mean(dim=0)
        # for i in range(tmp_sim.shape[0]):
        #     t_sim = tmp_sim[i].softmax(dim=-1)
        #     tmp_attn = t_sim.reshape(14, 14).detach().cpu().numpy()
        #     plt.figure(figsize=(8, 8))
        #     sns.heatmap(tmp_attn, cmap="viridis", square=True, cbar=False, xticklabels=False, yticklabels=False)
        #     plt.title("14x14 Attention Heatmap")
        #     plt.savefig(f"place_query_{i}.png", dpi=300, bbox_inches="tight")
        #     plt.clf()
        # breakpoint() # torch.Size([10, 8, 1, 27, 223])
        
        
        attn = sim.softmax(dim=-1)
        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=6,
        dim_head=64,
        heads=8,
        num_latents=64,
        max_num_media=None,
        max_num_frames=None,
        ff_mult=4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = (
            nn.Parameter(torch.randn(max_num_frames, dim))
            if exists(max_num_frames)
            else None
        )
        self.media_time_embs = (
            nn.Parameter(torch.randn(max_num_media, 1, dim))
            if exists(max_num_media)
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]
        # frame and media time embeddings
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:F], "F d -> b T F v d", b=b, T=T, v=v)
            x = x + frame_embs
        x = rearrange(
            x, "b T F v d -> b T (F v) d"
        )  # flatten the frame and spatial dimensions
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]

        # blocks
        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)
        for attn, ff in self.layers:
            latents = attn(x, latents, mask) + latents
            latents = ff(latents) + latents
        return self.norm(latents)