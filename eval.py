
import os
import random
import numpy as np
import torch
import wandb
import clip
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.data_utils import get_calvin_dataset, get_calvin_val_dataset
from utils.distributed_utils import init_distributed_device, world_info_from_env
from utils.eval_utils import eval_one_epoch_calvin_ddp
from utils.train_utils import get_checkpoint_all_param, get_checkpoint
from torch.distributed.elastic.multiprocessing.errors import record
from utils.arguments_utils import get_args
from models.gr1 import GR1Agent


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

@record
def main():
    args = get_args(is_eval=True)

    if args.tcp_rel:
        args.clip_state = True
    if args.save_checkpoints_to_wandb and args.save_checkpoint and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    device_id = init_distributed_device(args)
    print("device_id: ", device_id)

    random_seed(args.seed)

    model = GR1Agent(
        transformer_layers=args.transformer_layers,
        transformer_hidden_dim=args.transformer_hidden_dim,
        transformer_heads=args.transformer_heads,
        clip_device=device_id,
        checkpoint_path=args.checkpoint_path,
        sequence_length=args.sequence_length,
        num_resampler_query=args.num_resampler_query,
        num_obs_token_per_image=args.num_obs_token_per_image,
        calvin_input_image_size=args.calvin_input_image_size,
        patch_size=args.patch_size,
    )

    random_seed(args.seed, args.rank)

    print(f"Start running training on rank {args.rank}.")

    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    device_id = args.rank % torch.cuda.device_count()
    if args.precision == "bf16" or args.precision == "amp_bfloat16" or args.precision == "amp_bf16":
        model = model.bfloat16()
    elif args.precision == "fp16":
        model = model.half()
    elif args.precision == "fp32":
        model = model.float()
        if 'vision_encoder' in args.bf16_module and not args.conv_vision_encoder:
            model.vision_encoder.bfloat16()
        if "causal_transformer" in args.bf16_module:
            model.transformer_backbone.bfloat16()
        if "image_decoder" in args.bf16_module:
            model.image_decoder.bfloat16()
            model.image_decoder_obs_pred_projector.bfloat16()

    model.clip_model.requires_grad_(False)
    if not args.conv_vision_encoder:
        model.vision_encoder.requires_grad_(False)
    model = model.to(device_id)
    model._init_model_type()
  
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    ## save all parameter for v100
    # for i in range(5,11):
    #     folder = '/mnt/petrelfs/yangsizhe/projects/pretrain_for_manipulation/Robogr1/exp/bs=512_lr1e-3_bf16ve_not_init'
    #     checkpoint = torch.load(f'{folder}/{i}.pth', map_location="cpu")
    #     ddp_model.load_state_dict(checkpoint["model_state_dict"], False)
    #     checkpoint_dict = {
    #                 "model_state_dict": get_checkpoint_all_param(ddp_model),
    #             }
    #     torch.save(checkpoint_dict, f'{folder}/{i}_not_init.pth')

    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)

    ddp_model.eval()
    # ckpt_name = args.resume_from_checkpoint.split('/')[-1].split('.')[0]
    # eval_log_dir = f"evaluate/{ckpt_name}"
    eval_log_dir = 'evaluate'

    eval_one_epoch_calvin_ddp(
        args=args,
        model=ddp_model,
        image_processor=model.image_processor,
        tokenizer=clip,
        dataset_path=args.calvin_dataset,
        future_act_len=args.future_act_len,
        eval_log_dir=eval_log_dir,
        debug=args.visualize,
        reset=args.reset,
        diverse_inst=args.diverse_inst
    )

if __name__ == "__main__":
    os.environ['NCCL_BLOCKING_WAIT'] = '0'
    main()
