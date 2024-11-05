from collections import defaultdict, namedtuple
import logging
import os, json, random
from pathlib import Path
import sys
import time
import PIL.Image as Image
import copy
from collections import deque
from moviepy.editor import ImageSequenceClip
# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel
import time
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_env_state_for_initial_condition,
    get_log_dir,
    print_and_save,
)
import hydra
import numpy as np
from omegaconf import OmegaConf
from termcolor import colored
import torch
from tqdm.auto import tqdm

from calvin_env.envs.play_table_env import get_env
from utils.data_utils import preprocess_image, preprocess_text_calvin, world_to_tcp_frame, tcp_to_world_frame
import functools
from utils.train_utils import get_cast_dtype
os.environ['PYOPENGL_PLATFORM'] = 'egl'
logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    return env


class ModelWrapper(CalvinBaseModel):
    def __init__(self, model, tokenizer, image_processor, cast_dtype, history_len=10, tcp_rel=False):
        super().__init__()
        self.model = model

        self.cast_type = cast_dtype
        self.use_diff = False
        self.text_process_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
        self.image_process_fn = functools.partial(preprocess_image, image_processor=image_processor)
        self.action_hist_queue = []
        self.tcp_rel = tcp_rel
        self.history_len = history_len
        self.img_queue = deque(maxlen=history_len)
        self.gripper_queue = deque(maxlen=history_len)
        self.state_queue = deque(maxlen=history_len)
        self.mask_queue = deque(maxlen=history_len)
        self.text_queue = deque(maxlen=history_len)

    def reset(self):
        self.img_queue = deque(maxlen=self.history_len)
        self.gripper_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.mask_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)

    def step(self, obs, goal, timestep):
        # preprocess image
        image = obs["rgb_obs"]['rgb_static']
        image = Image.fromarray(image)
        image_x = self.image_process_fn([image])
        # expand image dimension
        image_x = image_x.unsqueeze(1).to(dtype=self.cast_type)

        gripper = obs["rgb_obs"]['rgb_gripper']
        gripper = Image.fromarray(gripper)
        gripper = self.image_process_fn([gripper])
        # expand image dimension
        gripper = gripper.unsqueeze(1).to(dtype=self.cast_type)

        # expand text dimension
        text_x = self.text_process_fn([goal])
        text_x = text_x.unsqueeze(1)

        state = obs['robot_obs']
        state = torch.from_numpy(np.stack([state]))
        state = state.unsqueeze(1).to(dtype=self.cast_type)
        state = torch.cat([state[..., :6], state[..., [-1]]], dim=-1)

        with torch.no_grad():
            device = 'cuda'
            image_x = image_x.to(device)
            text_x = text_x.to(device)
            gripper = gripper.to(device)
            state = state.to(device)

            self.img_queue.append(image_x)  # TODO find out how the policy completes the 5 sub-tasks. the obs of the later task will be appended after the former?
            self.gripper_queue.append(gripper)
            self.state_queue.append(state)
            if len(self.text_queue) == 0 and text_x is not None:  # the instruction does not change
                self.text_queue.append(text_x)
                for _ in range(self.model.module.sequence_length - 1):
                    self.text_queue.append(text_x)
            
            image_primary = torch.cat(list(self.img_queue), dim=1)
            image_wrist = torch.cat(list(self.gripper_queue), dim=1)
            state = torch.cat(list(self.state_queue), dim=1)
            input_text_token = torch.cat(list(self.text_queue), dim=1)

            num_step = image_primary.shape[1]
            if num_step < 10:  # padding
                input_image_primary = torch.cat([image_primary, image_primary[:, -1].repeat(1, 10-num_step, 1, 1, 1)], dim=1)
                input_image_wrist = torch.cat([image_wrist, image_wrist[:, -1].repeat(1, 10-num_step, 1, 1, 1)], dim=1)
                input_state = torch.cat([state, state[:, -1].repeat(1, 10-num_step, 1)], dim=1)
                # input_text_token = torch.cat([input_text_token, input_text_token.repeat(1, 10-num_step, 1)], dim=1)
            else:
                input_image_primary = image_primary
                input_image_wrist = image_wrist
                input_state = state
            arm_action, gripper_action, image_pred = self.model(
                image_primary=input_image_primary,
                image_wrist=input_image_wrist,
                state=input_state,
                text_token=input_text_token,
            )
            action = torch.concat((arm_action[0, :, 0, :], gripper_action[0, :, 0, :] > 0.5), dim=-1)
            action[:, -1] = (action[:, -1] - 0.5) * 2  # scale to -1 or 1
            action = action.cpu().detach().to(dtype=torch.float16).numpy()
            if num_step < 10:
                action = action[num_step - 1]
            else:
                action = action[-1]  # which action is needed? depend on the timestep  
        if self.tcp_rel:
            state = obs['robot_obs']
            state = torch.from_numpy(np.stack([state])).unsqueeze(0).float().cpu().detach()
            action = torch.from_numpy(np.stack([action])).unsqueeze(0).float().cpu().detach()
            action = tcp_to_world_frame(action, state)
            action=action.squeeze().to(dtype=torch.float16).numpy()
        return action
    
    def step_robo_flamingo(self, obs, goal):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """

        # preprocess image
        image = obs["rgb_obs"]['rgb_static']
        image = Image.fromarray(image)
        image_x = self.image_process_fn([image])
        # expand image dimension
        image_x = image_x.unsqueeze(1).unsqueeze(1).to(dtype=self.cast_type)
        # expand text dimension
        text_x = self.text_process_fn([goal])

        # fix window_size : ddp_model -> ... -> window_size
        if self.model.module.sep_lm_head:
            window_size = self.model.module.lm_head.window_size
            self.model.module.lm_head.window_size = 1
            if self.model.module.pad_length != -1 and self.feature_cache is None:
                self.model.module.lm_head.window_size = self.model.module.pad_length
        else:
            window_size = self.model.module.lang_encoder.lm_head.window_size
            self.model.module.lang_encoder.lm_head.window_size = 1
            if self.model.module.pad_length != -1 and self.feature_cache is None:
                self.model.module.lang_encoder.lm_head.window_size = self.model.module.pad_length
        gripper = None
        state = None

        if self.model.module.use_gripper:
            gripper = obs["rgb_obs"]['rgb_gripper']
            gripper = Image.fromarray(gripper)
            gripper = self.image_process_fn([gripper])
            # expand image dimension
            gripper = gripper.unsqueeze(1).unsqueeze(1).to(dtype=self.cast_type)
        
        # if self.model.module.use_state or self.model.module.sep_lm_head:
        if self.model.module.use_state or self.model.module.sep_lm_head:
            state = obs['robot_obs']
            state = torch.from_numpy(np.stack([state]))
            # if self.model.module.sep_lm_head:
            #     state = torch.cat([state[...,:6], state[...,[-1]]], dim=-1)
            if self.fusion_mode == 'two_way':
                state = state.repeat(2, 1)
            state = state.unsqueeze(1).unsqueeze(1).to(dtype=self.cast_type)
            state = state.to(torch.float32)
        with torch.no_grad():
            device = 'cuda'
            image_x = image_x.to(device)
            text_x = text_x.to(device)
            if gripper is not None:
                gripper = gripper.to(device)
            if state is not None:
                state = state.to(device)
            
            # if self.model.module.pad_length != -1:
            if len(self.img_queue) == 0:
                self.img_queue.append(image_x)
                for _ in range(self.model.module.pad_length - 1):
                    self.img_queue.append(image_x)
            else:
                self.img_queue.append(image_x)
            if len(self.gripper_queue) == 0 and gripper is not None:
                self.gripper_queue.append(gripper)
                for _ in range(self.model.module.pad_length - 1):
                    self.gripper_queue.append(gripper)
            else:
                self.gripper_queue.append(gripper)
            if len(self.state_queue) == 0 and state is not None:
                self.state_queue.append(state)
                for _ in range(self.model.module.pad_length - 1):
                    self.state_queue.append(state)
            else:
                self.state_queue.append(state)
            if len(self.mask_queue) == 0 and mask is not None:
                self.mask_queue.append(mask)
                for _ in range(self.model.module.pad_length - 1):
                    self.mask_queue.append(mask)
            if len(self.text_queue) == 0 and text_x is not None:
                self.text_queue.append(text_x)
                for _ in range(self.model.module.pad_length - 1):
                    self.text_queue.append(text_x)
            
            if self.model.module.pad_length != -1 and self.feature_cache is None:
                image_x = torch.cat(list(self.img_queue), dim=0)
                if gripper is not None:
                    gripper = torch.cat(list(self.gripper_queue), dim=0)
                if state is not None:
                    state = torch.cat(list(self.state_queue), dim=0)
                mask = torch.cat(list(self.mask_queue), dim=0)
                text_x = torch.cat(list(self.text_queue), dim=0)
            
            if self.fusion_mode == 'vit_concat':
                image_x = torch.cat(list(self.img_queue), dim=0)
                if gripper is not None:
                    gripper = torch.cat(list(self.gripper_queue), dim=0)
                if state is not None:
                    state = torch.cat(list(self.state_queue), dim=0)
                pass

            if self.fusion_mode == 'two_way':
                vision_x = torch.cat([image_x, gripper], dim=0)
                text_x = text_x.repeat(2, 1)
                mask = mask.repeat(2, 1)
                action = self.model(vision_x=vision_x, lang_x=text_x, attention_mask=mask, state_tensor = state, return_feature=True)
            else:
                action = self.model(vision_x=image_x, lang_x=text_x, attention_mask=mask, vision_gripper = gripper, state_tensor = state, return_feature=True)
            
            if self.model.module.pad_length != -1:
                if self.feature_cache is None:
                    self.feature_cache = action.logits[-1]
                else:
                    new_feat = torch.cat([self.feature_cache[1:], action.logits[-1]], dim=0)
                    self.feature_cache = new_feat
                    if not self.model.module.sep_lm_head:
                        self.model.module.lang_encoder.lm_head.window_size = window_size
                        lm_out = self.model.module.lang_encoder.lm_head(new_feat)
                    else:
                        self.model.module.lm_head.window_size = window_size
                        lm_out = self.model.module.lm_head(new_feat)
                    Output = namedtuple('Output', ['logits'])
                    action = Output(lm_out)
            
            if self.model.module.act_step == 1:
                action = torch.concat((action.logits[0], action.logits[1] > 0.5), dim=2).squeeze(0)[-1] # support multi step history
            else:
                pose = action.logits[0]
                gripper = action.logits[1] > 0.5
                pose = pose.squeeze(0)[-1].view(self.model.module.act_step, -1)
                gripper = gripper.squeeze(0)[-1].view(self.model.module.act_step, -1)
                action = torch.cat([pose, gripper], dim=-1)
                action = action[0] # select first step action
                
            action[-1] = (action[-1] - 0.5) * 2  # scale to -1 or 1
            action = action.cpu().detach().to(dtype=torch.float16).numpy()
        
        if self.model.module.tcp_rel:
            state = obs['robot_obs']
            state = torch.from_numpy(np.stack([state])).unsqueeze(0).float().cpu().detach()
            action = torch.from_numpy(np.stack([action])).unsqueeze(0).float().cpu().detach()
            action = tcp_to_world_frame(action, state)
            action=action.squeeze().to(dtype=torch.float16).numpy()
        return action

def evaluate_policy_ddp(model, env, epoch, calvin_conf_path, eval_log_dir=None, debug=False, create_plan_tsne=False, reset=False, diverse_inst=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    
    # val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    print("diverse_inst: ", diverse_inst)
    # set_trace()
    if diverse_inst:
        with open('./utils/lang_annotation_cache.json', 'r') as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)
    with open('./utils/eval_sequences.json', 'r') as f:
        eval_sequences = json.load(f)
    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    assert NUM_SEQUENCES % device_num == 0
    interval_len = int(NUM_SEQUENCES // device_num)
    eval_sequences = eval_sequences[device_id*interval_len:min((device_id+1)*interval_len, NUM_SEQUENCES)]
    results = []
    plans = defaultdict(list)
    local_sequence_i = 0
    base_sequence_i = device_id * interval_len

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug, eval_log_dir, base_sequence_i+local_sequence_i, reset=reset, diverse_inst=diverse_inst)
        results.append(result)
        eval_sequences.set_description(
            " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
        )
        local_sequence_i += 1
    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    def extract_iter_from_tqdm(tqdm_iter):
        return [_ for _ in tqdm_iter]
    
    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)

    eval_sequences = extract_iter_from_tqdm(eval_sequences)

    res_tup = [(res, eval_seq) for res, eval_seq in zip(results, eval_sequences)]
    all_res_tup = [copy.deepcopy(res_tup) for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(res_tup, all_res_tup, dst=0)

    if torch.distributed.get_rank() == 0:
        res_tup_list = merge_multi_list(all_res_tup)
        res_list = [_[0] for _ in res_tup_list]
        eval_seq_list = [_[1] for _ in res_tup_list]
        print_and_save(res_list, eval_seq_list, eval_log_dir, epoch)

    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug, eval_log_dir='', sequence_i=-1, reset=False, diverse_inst=False):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    # set_trace()
    success_counter = 0

    for subtask_i, subtask in enumerate(eval_sequence):
        # set_trace()
        if reset:
            success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug, eval_log_dir, subtask_i, sequence_i, diverse_inst=diverse_inst, robot_obs=robot_obs, scene_obs=scene_obs)
        else:
            success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug, eval_log_dir, subtask_i, sequence_i, diverse_inst=diverse_inst)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug, eval_log_dir='', subtask_i=-1, sequence_i=-1, robot_obs=None, scene_obs=None, diverse_inst=False):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    planned_actions = []
    if robot_obs is not None and scene_obs is not None:
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    obs = env.get_obs()
    # get lang annotation for subtask
    if diverse_inst:
        lang_annotation = val_annotations[sequence_i][subtask_i]
    else:
        lang_annotation = val_annotations[subtask][0]
    lang_annotation = lang_annotation.split('\n')[0]
    if '\u2019' in lang_annotation:
        lang_annotation.replace('\u2019', '\'')
    model.reset()
    start_info = env.get_info()

    for step in range(EP_LEN):
        breakpoint()
        action = model.step(obs, lang_annotation, step)
        if len(planned_actions) == 0:
            if action.shape == (7,):
                planned_actions.append(action)
            else:
                planned_actions.extend([action[i] for i in range(action.shape[0])])
        action = planned_actions.pop(0)

        obs, _, _, current_info = env.step(action)

        if step == 0:
            # for tsne plot, only if available
            collect_plan(model, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})

        if len(current_task_info) > 0:
            return True

    return False


def eval_one_epoch_calvin_ddp(args, model, dataset_path, image_processor, tokenizer, eval_log_dir=None, debug=False, future_act_len=-1, reset=False, diverse_inst=False):
    env = make_env(dataset_path)
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = 10
    wrapped_model = ModelWrapper(model, tokenizer, image_processor, cast_dtype, history_len=hist_len, tcp_rel=args.tcp_rel)
    evaluate_policy_ddp(wrapped_model, env, 0, args.calvin_conf_path, eval_log_dir=eval_log_dir, debug=debug, reset=reset, diverse_inst=diverse_inst)
