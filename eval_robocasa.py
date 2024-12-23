import os
import random
import numpy as np
import torch
import wandb
import clip
import json

from torch.nn.parallel import DistributedDataParallel as DDP
from utils.data_utils import get_calvin_dataset, get_calvin_val_dataset
from utils.distributed_utils import init_distributed_device, world_info_from_env
from utils.eval_utils_robocasa import eval_one_epoch_calvin_ddp
from utils.train_utils import get_checkpoint_all_param, get_checkpoint
from torch.distributed.elastic.multiprocessing.errors import record
from utils.arguments_utils import get_args
from models.gr1 import GR1Agent


from robomimic.config import config_factory
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.ground_utils import GroundUtils


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

@record
def main():
    args = get_args(is_eval=True)
    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.unlocked():
            config.update(ext_cfg)
    else:
        config = None
        
    if args.val_domain == "val":
        TrainUtils.VAL_ENV_INFOS = torch.load("/ailab/user/huanghaifeng/work/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_data_old/val_env_infos.pt", map_location="cpu")
    elif args.val_domain == "val_indomain":
        TrainUtils.VAL_ENV_INFOS = torch.load("/ailab/user/huanghaifeng/work/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_data_old/val_env_infos_indomain.pt", map_location="cpu")
    elif args.val_domain == "train":
        TrainUtils.VAL_ENV_INFOS = torch.load("/ailab/user/huanghaifeng/work/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_data/train_env_infos.pt", map_location="cpu")
    elif args.val_domain == "origin":
        TrainUtils.VAL_ENV_INFOS = torch.load("/ailab/user/huanghaifeng/work/robocasa_exps_haifeng/robocasa/datasets/v0.1/raw_data/val_env_infos_origin.pt", map_location="cpu")
    elif args.val_domain == "tmp":
        TrainUtils.VAL_ENV_INFOS = torch.load("/ailab/user/huanghaifeng/work/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_data/tmp_env_infos.pt", map_location="cpu")
    else:
        raise NotImplementedError

    ObsUtils.initialize_obs_utils_with_config(config)
    
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

    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)

    ddp_model.eval()
    # ckpt_name = args.resume_from_checkpoint.split('/')[-1].split('.')[0]
    # eval_log_dir = f"evaluate/{ckpt_name}"
    
    exp_dir = os.path.dirname(args.resume_from_checkpoint)
    eval_log_dir = os.path.join(exp_dir, args.run_name)
    
    if args.use_glamm:
        grounding_model = GroundUtils(device=device_id)
    else:
        grounding_model = None

    all_logs = eval_one_epoch_calvin_ddp(
        args=args,
        model=ddp_model,
        image_processor=model.image_processor,
        tokenizer=clip,
        dataset_path=args.calvin_dataset,
        future_act_len=args.future_act_len,
        eval_log_dir=eval_log_dir,
        debug=args.visualize,
        reset=args.reset,
        diverse_inst=args.diverse_inst,
        robocasa_config=config,
        grounding_model=grounding_model
    )
    
    if device_id == 0:
        with open(os.path.join(eval_log_dir, 'log.json'), 'w') as f:
            json.dump(all_logs, f)
        print(all_logs)

if __name__ == "__main__":
    os.environ['NCCL_BLOCKING_WAIT'] = '0'
    main()
