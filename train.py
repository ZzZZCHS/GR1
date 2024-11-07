import glob
import os
import random
from collections import OrderedDict
import numpy as np
import torch
import wandb
import clip
import json
import functools
from torch.nn.parallel import DistributedDataParallel as DDP
# from utils.robomimic_data_utils import SequenceDataset, MetaDataset
from utils.distributed_utils import init_distributed_device, world_info_from_env
from utils.train_utils import get_checkpoint, get_checkpoint_all_param, train_one_epoch_calvin, get_ckpt_name  #, validate_one_epoch_calvin
from utils.data_utils import preprocess_image, get_robomimic_dataset
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from utils.arguments_utils import get_args
from models.gr1 import GR1Agent

from robomimic.config import config_factory

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


@record
def main():
    args = get_args()
    
    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.unlocked():
            config.update(ext_cfg)
    else:
        config = None

    if args.tcp_rel:
        args.clip_state = True
    if args.save_checkpoints_to_wandb and args.save_checkpoint and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_DIR"] = f"{os.path.abspath(args.checkpoint_path)}"

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
        robomimic_config=config
    )

    if args.data_type == "calvin":
        calvin_dataset = get_calvin_dataset(args, model.image_processor, clip, epoch=0)
    elif args.data_type == "calvin_except_lang":
        calvin_dataset = get_calvin_dataset(args, model.image_processor, clip, epoch=0, key='all')  # except_lang
    elif args.data_type == "robomimic":
        calvin_dataset = get_robomimic_dataset(args, model.image_processor, clip, epoch=0, config=config)
    else:
        raise NotImplementedError
    # if args.validation:
    #     calvin_val_dataset = get_calvin_val_dataset(args, image_processor, clip, epoch=0)

    random_seed(args.seed, args.rank)

    print(f"Start running training on rank {args.rank}.")

    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
            dir=f"{os.path.abspath(args.checkpoint_path)}"
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
        # model._init_model_type()

    model.clip_model.requires_grad_(False)
    model.vision_encoder.requires_grad_(False)
    model = model.to(device_id)
    model._init_model_type()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    optimizer = torch.optim.AdamW([p for p in ddp_model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)  # TODO make sure the parameters which need to be optimized are passing

    total_training_steps = calvin_dataset.dataloader.num_batches * args.num_epochs
    args.warmup_steps = calvin_dataset.dataloader.num_batches 

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    if args.lr_scheduler == "linear":
        if args.gradient_accumulation_steps > 1:
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps + 1,
                num_training_steps=total_training_steps // args.gradient_accumulation_steps + 1,
            )
        else:
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=total_training_steps,
            )
    elif args.lr_scheduler == "cosine":
        if args.gradient_accumulation_steps > 1:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps + 1,
                num_training_steps=total_training_steps // args.gradient_accumulation_steps + 1,
            )
        else:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=total_training_steps,
            )
    elif args.lr_scheduler == 'cosine_restart':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    resume_from_epoch = 0

    if args.finetune_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Starting finetuning from pretrained checkpoint {args.finetune_from_checkpoint}")    
        checkpoint = torch.load(args.finetune_from_checkpoint, map_location="cpu")
        if args.reset_action_token:
            del checkpoint["model_state_dict"]["module.action_pred_token"] 
        if args.reset_obs_token:
            del checkpoint["model_state_dict"]["module.obs_tokens"]
        if args.reset_mask_token:
            del checkpoint["model_state_dict"]["module.mask_token"] 
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)
    
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        resume_from_epoch = checkpoint["epoch"] + 1
    
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=model.image_processor
    )

    ddp_model.train()
    if args.real_data:
        resume_from_epoch = 0 
    for epoch in range(resume_from_epoch, args.num_epochs):
        calvin_dataset.set_epoch(epoch)
        calvin_loader = calvin_dataset.dataloader

        train_one_epoch_calvin(
            args=args,
            model=ddp_model,
            epoch=epoch,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            calvin_loader=calvin_loader,
            device_id=device_id,
            wandb=wandb,
            image_processor=preprocess_image_fn
        )

        if args.rank == 0 and args.save_checkpoint:
            if not os.path.exists(f"{args.checkpoint_path}/exp/{args.run_name}"):
                os.makedirs(f"{args.checkpoint_path}/exp/{args.run_name}")

            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": get_checkpoint(ddp_model),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            }
            if args.delete_previous_checkpoint:
                if epoch > 0:
                    os.remove(ckpt_path)
            ckpt_name = get_ckpt_name(args, epoch)
            ckpt_path = os.path.join(f"{args.checkpoint_path}/exp", args.run_name, ckpt_name)

            print(f"Saving checkpoint to {ckpt_path}")
            torch.save(checkpoint_dict, ckpt_path)
            
    
    if args.rank == 0 and args.report_to_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
