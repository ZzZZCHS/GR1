import random
from functools import partial
from copy import deepcopy
from timm.models.vision_transformer import Block
import torch
import time
from torch import nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import clip
import numpy as np
from models.vit_mae import MaskedAutoencoderViT
from models.perceiver_resampler import PerceiverResampler
from models.gpt2 import GPT2Model
from transformers import GPT2Config
import torch.utils.checkpoint as checkpoint


def generate_attention_mask(K, num_A, num_B):
    sequence_length = (num_A + num_B) * K
    attention_mask = torch.zeros((sequence_length, sequence_length))

    for i in range(K):
        start_index = i * (num_A + num_B)
        end_index = start_index + num_A + num_B

        # the i-th sub-sequence can not attend to the sub-sequences that after the i-th
        attention_mask[start_index:end_index, end_index:] = -float('inf')

        # the sub-sub-sequence B can not be attended to
        attention_mask[:, start_index + num_A:end_index] = -float('inf')

    return attention_mask

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


class GR1Agent(nn.Module):
    def __init__(
        self,
        transformer_layers,
        transformer_hidden_dim,
        transformer_heads,
        clip_device,
        checkpoint_path,
        sequence_length=10,
        num_resampler_query=9,
        num_obs_token_per_image=10,
        calvin_input_image_size=224,
        patch_size=16,
        robomimic_config=None
    ):
        super().__init__()
        self.HIDDEN_DIM = transformer_hidden_dim
        self.sequence_length = sequence_length
        
        self.text_projector = nn.Linear(512, self.HIDDEN_DIM)  # the output dim of ViT-L/14 text encoder is 768, the output dim of ViT-B/32 text encoder is 512

        # state encoder
        ARM_STATE_FEATURE_DIM =384
        GRIPPER_STATE_FEATURE_DIM = 384
        # self.arm_state_encoder = nn.Linear(6, ARM_STATE_FEATURE_DIM)
        # self.gripper_state_encoder = nn.Linear(2, GRIPPER_STATE_FEATURE_DIM)
        # self.state_projector = nn.Linear(ARM_STATE_FEATURE_DIM + GRIPPER_STATE_FEATURE_DIM, self.HIDDEN_DIM)
        self.state_projector = nn.Linear(9, self.HIDDEN_DIM)
        
        self.input_projector = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1)

        # vision encoder (frozen)
        self.vision_encoder = MaskedAutoencoderViT(
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        # resampler
        self.RESAMPLER_HIDDEN_DIM = 768
        self.NUM_RESAMPLER_QUERY = num_resampler_query
        # self.image_projector_for_resampler = nn.Linear(768, self.RESAMPLER_HIDDEN_DIM)
        self.perceiver_resampler = PerceiverResampler(dim=self.RESAMPLER_HIDDEN_DIM, num_latents=self.NUM_RESAMPLER_QUERY, depth=3)
        self.image_left_projector = nn.Linear(self.RESAMPLER_HIDDEN_DIM, self.HIDDEN_DIM)
        self.cls_token_left_projector = nn.Linear(self.RESAMPLER_HIDDEN_DIM, self.HIDDEN_DIM)
        self.image_right_projector = nn.Linear(self.RESAMPLER_HIDDEN_DIM, self.HIDDEN_DIM)
        self.cls_token_right_projector = nn.Linear(self.RESAMPLER_HIDDEN_DIM, self.HIDDEN_DIM)
        self.image_wrist_projector = nn.Linear(self.RESAMPLER_HIDDEN_DIM, self.HIDDEN_DIM)
        self.cls_token_wrist_projector = nn.Linear(self.RESAMPLER_HIDDEN_DIM, self.HIDDEN_DIM)

        # action_pred_token
        self.action_pred_token = nn.Parameter(torch.zeros(1, 1, 1, self.HIDDEN_DIM))
        # obs_token
        self.NUM_OBS_TOKEN_PER_IMAGE = num_obs_token_per_image
        self.NUM_OBS_TOKEN = self.NUM_OBS_TOKEN_PER_IMAGE * 3
        self.obs_tokens = nn.Parameter(torch.zeros(1, 1, self.NUM_OBS_TOKEN, self.HIDDEN_DIM))

        # causal transformer
        self.embedding_layer_norm = nn.LayerNorm(self.HIDDEN_DIM)
        num_non_learnable_token_per_timestep = 1+1+self.NUM_RESAMPLER_QUERY*3+1*3
        self.attention_mask = nn.Parameter(generate_attention_mask(K=self.sequence_length, num_A=num_non_learnable_token_per_timestep, num_B=self.NUM_OBS_TOKEN+1), requires_grad=False)
        self.transformer_backbone_position_embedding = nn.Parameter(torch.zeros(1, self.sequence_length, 1, self.HIDDEN_DIM), requires_grad=True)
        config = GPT2Config()
        config.n_layer = transformer_layers
        config.hidden_size = self.HIDDEN_DIM
        config.n_head = transformer_heads
        config.vocab_size = 1
        # !!!
        # config.resid_pdrop = 0
        # config.embd_pdrop = 0
        # config.attn_pdrop = 0
        # config.summary_first_dropout = 0
        self.transformer_backbone = GPT2Model(config)

        # action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(self.HIDDEN_DIM, 192),
            nn.ReLU(),
            nn.Linear(192, 192),
            nn.ReLU(),
        )
        if robomimic_config and 'real_actions' in robomimic_config.train.action_keys:
            self.arm_dim = 7
            self.arm_action_decoder = nn.Sequential(
                nn.Linear(192, 7)
            )
        else:
            self.arm_dim = 6
            self.arm_action_decoder = nn.Sequential(
                nn.Linear(192, 6),
                torch.nn.Tanh(),  # !!!
            )
        self.gripper_action_decoder = nn.Sequential(
            nn.Linear(192, 1),
            torch.nn.Sigmoid(),
        )
        
        # image decoder
        self.IMAGE_DECODER_HIDDEN_DIM = 384
        self.NUM_MASK_TOKEN = int(calvin_input_image_size**2 / patch_size / patch_size)  # i.e. num_patch
        self.PATCH_SIZE = patch_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.IMAGE_DECODER_HIDDEN_DIM))
        self.image_decoder_obs_pred_projector = nn.Linear(self.HIDDEN_DIM, self.IMAGE_DECODER_HIDDEN_DIM)
        self.image_decoder_position_embedding = nn.Parameter(torch.zeros(1, self.NUM_OBS_TOKEN_PER_IMAGE + self.NUM_MASK_TOKEN, self.IMAGE_DECODER_HIDDEN_DIM), requires_grad=False)  # fixed sin-cos embedding #   cls_token is alse passed to the decoder in mae
        self.image_decoder = nn.Sequential(
            Block(self.IMAGE_DECODER_HIDDEN_DIM, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
            Block(self.IMAGE_DECODER_HIDDEN_DIM, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
        )
        self.image_decoder_norm = nn.LayerNorm(self.IMAGE_DECODER_HIDDEN_DIM)
        self.image_decoder_pred = nn.Linear(self.IMAGE_DECODER_HIDDEN_DIM, self.PATCH_SIZE**2 * 3)

        self.num_non_learnable_token_per_timestep = num_non_learnable_token_per_timestep

        self.initialize_weights()

        # text encoder (frozen)
        self.clip_model, self.image_processor = clip.load("ViT-B/32", device=clip_device, download_root=f"{checkpoint_path}/checkpoints/clip")
        
        checkpoint = torch.load(f"{checkpoint_path}/checkpoints/vit_mae/mae_pretrain_vit_base.pth", map_location='cpu')
        msg = self.vision_encoder.load_state_dict(checkpoint['model'], strict=False)
    
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        image_decoder_position_embedding_obs = get_2d_sincos_pos_embed(self.IMAGE_DECODER_HIDDEN_DIM, int(self.NUM_OBS_TOKEN_PER_IMAGE**.5), cls_token=False)
        image_decoder_position_embedding_mask = get_2d_sincos_pos_embed(self.IMAGE_DECODER_HIDDEN_DIM, int(self.NUM_MASK_TOKEN**.5),cls_token=False)
        image_decoder_position_embedding = np.concatenate((image_decoder_position_embedding_obs, image_decoder_position_embedding_mask),axis=0)
        self.image_decoder_position_embedding.data.copy_(torch.from_numpy(image_decoder_position_embedding).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=.02)

        torch.nn.init.normal_(self.transformer_backbone_position_embedding, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _init_model_type(self):
        self.input_projector_type = next(self.input_projector.parameters()).type()
        self.vision_encoder_type = next(self.vision_encoder.parameters()).type()
        self.perceiver_resampler_type = next(self.perceiver_resampler.parameters()).type()
        self.transformer_backbone_type = next(self.transformer_backbone.parameters()).type()
        self.action_decoder_type = next(self.action_decoder.parameters()).type()
        self.image_decoder_type = next(self.image_decoder.parameters()).type()
        self.image_decoder_position_embedding = self.image_decoder_position_embedding.type(self.image_decoder_type)
        self.mask_token = self.mask_token.type(self.image_decoder_type)
        self.image_decoder_pred_type = next(self.image_decoder_pred.parameters()).type()

    def forward(self, image_left, image_right, image_wrist, state, text_token, epoch=0, masks=None):
        device = image_left.device
        B, S, _ = state.shape
        S_AND_FUTURE = image_left.shape[1]
        image_pred = None
        # encoder text
        with torch.no_grad():
            text_feature = self.clip_model.encode_text(text_token.flatten(0, 1))
            text_feature = text_feature.type(state.type())
        text_embedding = self.text_projector(text_feature)
        text_embedding = text_embedding.view(B, S, -1, self.HIDDEN_DIM) # (bs, sequence_length, 1, hidden_dim)

        # encode state
        state = state.flatten(0, 1)
        # arm_state_feature = self.arm_state_encoder(state[:, :6])
        # gripper_state_one_hot = torch.nn.functional.one_hot(torch.where(state[:, 6:].flatten() < 1, torch.tensor(0).to(device), torch.tensor(1).to(device)), num_classes=2)
        # gripper_state_feature = self.gripper_state_encoder(gripper_state_one_hot.type_as(state))
        # state_embedding = self.state_projector(torch.cat((arm_state_feature, gripper_state_feature), dim=1))
        state_embedding = self.state_projector(state)
        state_embedding = state_embedding.view(B, S, -1, self.HIDDEN_DIM) # (bs, sequence_length, 1, hidden_dim)

        image_left = image_left.flatten(0, 1)
        image_right = image_right.flatten(0, 1)
        image_wrist = image_wrist.flatten(0, 1)

        if image_left.type() != self.input_projector_type:
            image_left = image_left.type(self.input_projector_type)
            image_right = image_right.type(self.input_projector_type)
            image_wrist = image_wrist.type(self.input_projector_type)
        if image_left.shape[1] == 4:
            image_left = self.input_projector(image_left)
            image_right = self.input_projector(image_right)
            image_wrist = self.input_projector(image_wrist)
        
        # encode images
        if image_left.type() != self.vision_encoder_type:
            image_left = image_left.type(self.vision_encoder_type)
            image_right = image_right.type(self.vision_encoder_type)
            image_wrist = image_wrist.type(self.vision_encoder_type)
        
        if masks is not None:
            image_left_feature, _, _ = checkpoint.checkpoint(self.vision_encoder.forward_encoder, image_left)
            image_right_feature, _, _ = checkpoint.checkpoint(self.vision_encoder.forward_encoder, image_right)
            image_wrist_feature, _, _ = checkpoint.checkpoint(self.vision_encoder.forward_encoder, image_wrist)
        else:
            with torch.no_grad():
                image_left_feature, _, _ = self.vision_encoder.forward_encoder(image_left, mask_ratio=0.0)
                image_right_feature, _, _ = self.vision_encoder.forward_encoder(image_right, mask_ratio=0.0)
                image_wrist_feature, _, _ = self.vision_encoder.forward_encoder(image_wrist, mask_ratio=0.0)
        
        if image_left_feature.type() != self.perceiver_resampler_type:
            image_left_feature = image_left_feature.type(self.perceiver_resampler_type)
            image_right_feature = image_right_feature.type(self.perceiver_resampler_type)
            image_wrist_feature = image_wrist_feature.type(self.perceiver_resampler_type)
        image_left_feature = image_left_feature.view(B, S_AND_FUTURE, image_left_feature.shape[-2], image_left_feature.shape[-1])
        image_right_feature = image_right_feature.view(B, S_AND_FUTURE, image_right_feature.shape[-2], image_right_feature.shape[-1])
        image_wrist_feature = image_wrist_feature.view(B, S_AND_FUTURE, image_wrist_feature.shape[-2], image_wrist_feature.shape[-1])

        cls_token_idx = 0
        idx = cls_token_idx + 1

        image_left_cls_token = image_left_feature[:, :, :idx, :]
        image_right_cls_token = image_right_feature[:, :, :idx, :]
        image_wrist_cls_token = image_wrist_feature[:, :, :idx, :]

        image_left_feature = image_left_feature[:, :, idx:, :]
        image_right_feature = image_right_feature[:, :, idx:, :]
        image_wrist_feature = image_wrist_feature[:, :, idx:, :]

        # perceiver resampler
        image_left_feature = self.perceiver_resampler(
            image_left_feature.reshape(B*S, 196, self.RESAMPLER_HIDDEN_DIM).unsqueeze(1).unsqueeze(1),
            mask=masks[:, :2] if masks is not None else None
            
        )  # mae vit outputs 196 tokens
        image_right_feature = self.perceiver_resampler(
            image_right_feature.reshape(B*S, 196, self.RESAMPLER_HIDDEN_DIM).unsqueeze(1).unsqueeze(1),
            mask=masks[:, 2:4] if masks is not None else None
        ) 
        image_wrist_feature = self.perceiver_resampler(
            image_wrist_feature.reshape(B*S, 196, self.RESAMPLER_HIDDEN_DIM).unsqueeze(1).unsqueeze(1),
            mask=masks[:, 4:] if masks is not None else None
        )
        image_left_embedding = self.image_left_projector(image_left_feature.flatten(0, 2)).view(B, S, -1, self.HIDDEN_DIM)
        image_right_embedding = self.image_right_projector(image_right_feature.flatten(0, 2)).view(B, S, -1, self.HIDDEN_DIM)
        image_wrist_embedding = self.image_wrist_projector(image_wrist_feature.flatten(0, 2)).view(B, S, -1, self.HIDDEN_DIM)          
        image_embedding = torch.cat((image_left_embedding, image_right_embedding, image_wrist_embedding), dim=2)

        image_cls_token_left_embedding = self.cls_token_left_projector(image_left_cls_token.flatten(0, 2)).view(B, S, -1, self.HIDDEN_DIM)
        image_cls_token_right_embedding = self.cls_token_right_projector(image_right_cls_token.flatten(0, 2)).view(B, S, -1, self.HIDDEN_DIM)
        image_cls_token_wrist_embedding = self.cls_token_wrist_projector(image_wrist_cls_token.flatten(0, 2)).view(B, S, -1, self.HIDDEN_DIM)
        image_cls_token_embedding = torch.cat((image_cls_token_left_embedding, image_cls_token_right_embedding, image_cls_token_wrist_embedding), dim=2)

        # aggregate embeddings and add timestep position encoding

        embeddings = [text_embedding, state_embedding, image_embedding, image_cls_token_embedding]
        embeddings = torch.cat(embeddings, dim=2)
            
        pred_token_start_idx = embeddings.shape[2]

        transformer_input = torch.cat((embeddings, self.obs_tokens.repeat(B, S, 1, 1), self.action_pred_token.repeat(B, S, 1, 1)), dim=2)  # (B, S, 1+1+2*self.NUM_RESAMPLER_QUERY+2*1 + NUM_OBS_TOKEN + 1, HIDDEN_DIM)
        transformer_input = transformer_input + self.transformer_backbone_position_embedding.repeat(B, 1, transformer_input.shape[-2], 1)
        transformer_input = transformer_input.flatten(1, 2)

        # causal transformer forward
        transformer_input = self.embedding_layer_norm(transformer_input)
        transformer_output = self.transformer_backbone(inputs_embeds=transformer_input, attention_mask=self.attention_mask)
        transformer_output = transformer_output.view(B, S, -1, self.HIDDEN_DIM)
        obs_pred_feature = transformer_output[:, :, pred_token_start_idx : pred_token_start_idx+self.NUM_OBS_TOKEN, :]
        action_pred_feature = transformer_output[:, :, pred_token_start_idx+self.NUM_OBS_TOKEN : pred_token_start_idx+self.NUM_OBS_TOKEN+1, :]
        
        # decode action
        action_pred_feature = self.action_decoder(action_pred_feature.reshape(-1, self.HIDDEN_DIM))
        arm_action = self.arm_action_decoder(action_pred_feature).view(B, S, -1, self.arm_dim)
        gripper_action = self.gripper_action_decoder(action_pred_feature).view(B, S, -1, 1)


        # decode image
        obs_pred_embedding = self.image_decoder_obs_pred_projector(obs_pred_feature.reshape(-1, self.HIDDEN_DIM))
        obs_pred_embedding = obs_pred_embedding.view(B * S * (self.NUM_OBS_TOKEN // self.NUM_OBS_TOKEN_PER_IMAGE), self.NUM_OBS_TOKEN_PER_IMAGE, self.IMAGE_DECODER_HIDDEN_DIM)
        mask_tokens = self.mask_token.repeat(B * S * (self.NUM_OBS_TOKEN // self.NUM_OBS_TOKEN_PER_IMAGE), self.NUM_MASK_TOKEN, 1)
        image_decoder_input = torch.cat((obs_pred_embedding, mask_tokens), dim=1) 
        image_decoder_input = image_decoder_input + self.image_decoder_position_embedding

        image_decoder_output = self.image_decoder(image_decoder_input)

        image_pred_feature = image_decoder_output[:, -self.NUM_MASK_TOKEN:, :]
        image_pred_feature = self.image_decoder_norm(image_pred_feature.reshape(-1, self.IMAGE_DECODER_HIDDEN_DIM))
        image_pred = self.image_decoder_pred(image_pred_feature)  # need to be unpatchfied
        image_pred = image_pred.view(B * S, self.NUM_OBS_TOKEN // self.NUM_OBS_TOKEN_PER_IMAGE, self.NUM_MASK_TOKEN, -1)  # (B * S, self.NUM_OBS_TOKEN // self.NUM_OBS_TOKEN_PER_IMAGE, self.NUM_MASK_TOKEN, self.PATCH_SIZE**2 * 3 or image_latent_dim)

        return arm_action, gripper_action, image_pred
