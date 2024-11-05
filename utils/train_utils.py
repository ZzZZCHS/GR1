import time
from contextlib import suppress

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from utils.data_utils import world_to_tcp_frame, tcp_to_world_frame, RandomShiftsAug
from einops import rearrange

from PIL import Image


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16" or precision == "amp_bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    else:
        cast_dtype = torch.float32
    return cast_dtype

def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress

def get_ckpt_name(args, epoch=-1):
    return f'{epoch}.pth'

def patchify(imgs, patch_size):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0

    h = w = imgs.shape[2] // patch_size
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, patch_size, w, patch_size))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, patch_size**2 * 3))

    return x

def normalize_patchfied_image(patchfied_imgs):
    mean = patchfied_imgs.mean(dim=-1, keepdim=True)
    var = patchfied_imgs.var(dim=-1, keepdim=True)
    patchfied_imgs = (patchfied_imgs - mean) / (var + 1.e-6)**.5

    return patchfied_imgs

def train_one_epoch_calvin(
    args,
    model,
    epoch,
    calvin_loader,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
    image_processor=None
):
    num_batches_per_epoch_calvin = calvin_loader.num_batches

    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []
    camera_names = ['robot0_agentview_left_image', 'robot0_agentview_right_image', 'robot0_eye_in_hand_image']
    state_names = ['robot0_base_to_eef_pos', 'robot0_base_to_eef_quat', 'robot0_gripper_qpos']

    # torch.cuda.synchronize()
    # t0 = time.time()
    
    for num_steps, batch_calvin in t:
        # if num_steps == 0:
        #     batch_calvin = batch_calvin_tmp
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch
        
        images_dict = {}
        for cam_name in camera_names:
            batch_seq_images = batch_calvin['obs'][cam_name][:, :args.window_size, ...]
            image_list = batch_seq_images.flatten(0, 1).unbind(dim=0)
            image_list = [Image.fromarray(image.numpy().astype(np.uint8)) for image in image_list]
            processed_images = image_processor(image_list)
            images_dict[cam_name] = processed_images.reshape(args.batch_size_calvin, args.window_size, 3, 224, 224)
        images_left = images_dict['robot0_agentview_left_image'].to(device_id, dtype=cast_dtype, non_blocking=True)
        images_right = images_dict['robot0_agentview_right_image'].to(device_id, dtype=cast_dtype, non_blocking=True)
        images_wrist = images_dict['robot0_eye_in_hand_image'].to(device_id, dtype=cast_dtype, non_blocking=True)

        text_tokens = batch_calvin['obs']['lang_emb'][:, :args.window_size, :].to(device_id, non_blocking=True)
        
        states = []
        for state_name in state_names:
            states.append(batch_calvin['obs'][state_name][:, :args.window_size, :].to(device_id, non_blocking=True))
        states = torch.cat(states, dim=-1).to(torch.float32)
        # states[:, :, :7] += torch.randn(states[:, :, :7].shape).to(states.device) * 0.001 # !!!
        # states *= 0 # !!!

        actions = batch_calvin['actions'][:, :args.window_size, :7].to(device_id, non_blocking=True).to(torch.float32)
        
        # if 0 in batch_calvin['index']:
        #     breakpoint()

        actions[..., 6:] = (actions[..., 6:] + 1) // 2
        # actions[..., :6] /= 2

        # prepare input and label
        input_image_left = images_left[:, :args.sequence_length, :]
        input_image_right = images_right[:, :args.sequence_length, :]
        input_image_wrist = images_wrist[:, :args.sequence_length, :]
        input_text_token = text_tokens[:, :args.sequence_length, :]
        input_state = states[:, :args.sequence_length, :]
        label_action = actions[:, :args.sequence_length, :].unsqueeze(-2)

        with autocast():  # image_left, image_wrist, state, language_instruction
            arm_action, gripper_action, image_pred = model(
                image_left=input_image_left,
                image_right=input_image_right,
                image_wrist=input_image_wrist,
                state=input_state,
                text_token=input_text_token,
                epoch=epoch
            )
        
        loss_arm_action = torch.nn.functional.smooth_l1_loss(arm_action, label_action[:, :, :, :6])
        loss_gripper_action = torch.nn.functional.binary_cross_entropy(gripper_action, label_action[:, :, :, 6:])

        label_image_left = images_left[:, args.future_steps:, :].flatten(0, 1)
        label_image_right = images_right[:, args.future_steps:, :].flatten(0, 1)
        label_image_wrist = images_wrist[:, args.future_steps:, :].flatten(0, 1)
        label_image_left = patchify(label_image_left, patch_size=args.patch_size)
        label_image_right = patchify(label_image_right, patch_size=args.patch_size)
        label_image_wrist = patchify(label_image_wrist, patch_size=args.patch_size)
        label_image_left = normalize_patchfied_image(label_image_left)
        label_image_right = normalize_patchfied_image(label_image_right)
        label_image_wrist = normalize_patchfied_image(label_image_wrist)
        loss_image = 0.5 * (torch.nn.functional.mse_loss(image_pred[:, 0, :, :], label_image_left) + torch.nn.functional.mse_loss(image_pred[:, 1, :, :], label_image_right) + torch.nn.functional.mse_loss(image_pred[:, 2, :, :], label_image_wrist))

        loss_calvin = loss_arm_action + 0.01 * loss_gripper_action + 0.1 * loss_image
        loss = loss_calvin / args.gradient_accumulation_steps
        loss_arm_action = loss_arm_action / args.gradient_accumulation_steps
        loss_gripper_action = loss_gripper_action / args.gradient_accumulation_steps
        loss_image = loss_image / args.gradient_accumulation_steps
        mv_avg_loss.append(loss.item())

        #### BACKWARD PASS ####
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                calvin_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    * args.world_size
                    / step_time_m.val
                )
                calvin_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    / step_time_m.val
                )

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "calvin_samples_per_second": calvin_samples_per_second,
                        "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_calvin": loss.item() * args.gradient_accumulation_steps,
                        "loss_arm_action": loss_arm_action.item() * args.gradient_accumulation_steps,
                        "loss_gripper_action": loss_gripper_action.item() * args.gradient_accumulation_steps,
                        "loss_image": loss_image.item() * args.gradient_accumulation_steps,
                        "global_step": global_step,
                    },
                )

        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss": loss_calvin.item(), "loss_arm_action": loss_arm_action.item(), "loss_gripper_action": loss_gripper_action.item(), "loss_image": loss_image.item()})

        if args.save_every_iter != -1 and args.save_checkpoint and global_step % args.save_every_iter == 0 and global_step > 0:
                
            if args.rank == 0:
                import os
                if not os.path.exists(f"{args.checkpoint_path}/exp/{args.run_name}"):
                    os.makedirs(f"{args.checkpoint_path}/exp/{args.run_name}")

                checkpoint_dict = {
                    "epoch": epoch,
                    "model_state_dict": get_checkpoint(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                }

                ckpt_name = get_ckpt_name(args, global_step)
                ckpt_path = os.path.join(f"{args.checkpoint_path}/exp", args.run_name, ckpt_name)
                print(f"Saving checkpoint to {ckpt_path}")
                torch.save(checkpoint_dict, ckpt_path)
                if args.delete_previous_checkpoint:
                    if epoch > 0:
                        os.remove(ckpt_path)

def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            del state_dict[name]

    return state_dict

def get_checkpoint_all_param(model):
    state_dict = model.state_dict()

    return state_dict

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

