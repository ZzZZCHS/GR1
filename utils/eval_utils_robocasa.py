from collections import defaultdict, namedtuple
import logging
import os, json, random
from pathlib import Path
import sys
import time
import PIL.Image as Image
import copy
from collections import deque, OrderedDict
from moviepy.editor import ImageSequenceClip
from copy import deepcopy
import cv2
# This is for using the locally installed repo clone when using slurm
# from calvin_agent.models.calvin_base_model import CalvinBaseModel
import time
# sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

# from calvin_agent.evaluation.multistep_sequences import get_sequences
# from calvin_agent.evaluation.utils import (
#     collect_plan,
#     count_success,
#     create_tsne,
#     get_env_state_for_initial_condition,
#     get_log_dir,
#     print_and_save,
# )
# import hydra
import numpy as np
from omegaconf import OmegaConf
from termcolor import colored
import torch
from tqdm.auto import tqdm
import imageio
import traceback

# from calvin_env.envs.play_table_env import get_env
from utils.data_utils import preprocess_image, preprocess_text_calvin, world_to_tcp_frame, tcp_to_world_frame
import functools
from utils.train_utils import get_cast_dtype
os.environ['PYOPENGL_PLATFORM'] = 'egl'
logger = logging.getLogger(__name__)

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.train_utils as TrainUtils

EP_LEN = 360
NUM_SEQUENCES = 50 # !!!

import h5py
f = h5py.File('/ailab/user/huanghaifeng/work/robocasa_exps_haifeng/robocasa/datasets/v0.1/generated_data/PnPCounterToCab.hdf5', 'r')


def make_env(dataset_path, robocasa_config):
    # val_folder = Path(dataset_path) / "validation"
    # env = get_env(val_folder, show_gui=False)
    env_meta_list = []
    shape_meta_list = []
    for dataset_cfg in robocasa_config.train.data:
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        ds_format = robocasa_config.train.data_format
        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

        # load basic metadata from training file
        print("\n============= Loaded Environment Metadata =============")
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path, ds_format=ds_format)
        # populate language instruction for env in env_meta
        env_meta["env_lang"] = dataset_cfg.get("lang", None)

        # update env meta if applicable
        from robomimic.utils.script_utils import deep_update
        deep_update(env_meta, dataset_cfg.get("env_meta_update_dict", {}))
        deep_update(env_meta, robocasa_config.experiment.env_meta_update_dict)
        env_meta_list.append(env_meta)

        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=dataset_path,
            action_keys=robocasa_config.train.action_keys,
            all_obs_keys=robocasa_config.all_obs_keys,
            ds_format=ds_format,
            verbose=True
        )
        shape_meta_list.append(shape_meta)
    
    eval_env_meta_list = []
    eval_shape_meta_list = []
    eval_env_name_list = []
    eval_env_horizon_list = []
    
    for (dataset_i, dataset_cfg) in enumerate(robocasa_config.train.data):
        if env_meta_list[dataset_i]["env_name"] not in TrainUtils.VAL_ENV_INFOS:
            continue
        do_eval = dataset_cfg.get("do_eval", True)
        if do_eval is not True:
            continue
        eval_env_meta_list.append(env_meta_list[dataset_i])
        eval_shape_meta_list.append(shape_meta_list[dataset_i])
        eval_env_name_list.append(env_meta_list[dataset_i]["env_name"])
        horizon = dataset_cfg.get("horizon", robocasa_config.experiment.rollout.horizon)
        eval_env_horizon_list.append(horizon)
    
    return env_iterator(eval_env_meta_list, eval_shape_meta_list, eval_env_name_list, eval_env_horizon_list, robocasa_config)


class ModelWrapper:
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
        self.img_left_queue = deque(maxlen=history_len)
        self.img_right_queue = deque(maxlen=history_len)
        self.gripper_queue = deque(maxlen=history_len)
        self.state_queue = deque(maxlen=history_len)
        self.mask_queue = deque(maxlen=history_len)
        self.text_queue = deque(maxlen=history_len)
        
        self.state_names = ['robot0_base_to_eef_pos', 'robot0_base_to_eef_quat', 'robot0_gripper_qpos']

    def reset(self):
        self.img_left_queue = deque(maxlen=self.history_len)
        self.img_right_queue = deque(maxlen=self.history_len)
        self.gripper_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.mask_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)

    def step(self, obs, goal, timestep):
        # preprocess image
        # image = obs["rgb_obs"]['rgb_static']
        # image = Image.fromarray(image)
        # image_x = self.image_process_fn([image])
        
        image_left = obs['robot0_agentview_left_image'][-1].transpose(1, 2, 0)
        image_left = Image.fromarray((image_left * 255).astype(np.uint8))
        # breakpoint()
        image_left_x = self.image_process_fn([image_left])
        # expand image dimension
        image_left_x = image_left_x.unsqueeze(1).to(dtype=self.cast_type)
        
        image_right = obs['robot0_agentview_right_image'][-1].transpose(1, 2, 0)
        image_right = Image.fromarray((image_right * 255).astype(np.uint8))
        image_right_x = self.image_process_fn([image_right])
        # expand image dimension
        image_right_x = image_right_x.unsqueeze(1).to(dtype=self.cast_type)
            

        gripper = obs['robot0_eye_in_hand_image'][-1].transpose(1, 2, 0)
        gripper = Image.fromarray((gripper * 255).astype(np.uint8))
        gripper = self.image_process_fn([gripper])
        # expand image dimension
        gripper = gripper.unsqueeze(1).to(dtype=self.cast_type)
        
        
        mask_names = ['robot0_agentview_left_mask', 'robot0_agentview_right_mask', 'robot0_eye_in_hand_mask']
        masks = None
        if mask_names[0] in obs:
            masks = []
            for mask_name in mask_names:
                tmp_mask = obs[mask_name]
                target_obj_mask = torch.tensor((tmp_mask == 1)).to(torch.float32)
                target_place_mask = torch.tensor((tmp_mask == 2)).to(torch.float32)
                target_obj_mask = torch.nn.functional.interpolate(target_obj_mask, size=(14, 14), mode='bicubic', align_corners=True)
                target_place_mask = torch.nn.functional.interpolate(target_place_mask, size=(14, 14), mode='bicubic', align_corners=True)
                masks.extend([target_obj_mask, target_place_mask])
                
                tmp_mask = torch.tensor(tmp_mask).squeeze(-1).to(torch.uint8)
                tmp_mask = torch.nn.functional.interpolate(tmp_mask, size=(224, 224), mode='nearest').unsqueeze(2)
                if 'left' in mask_name:
                    image_left_x = torch.cat([image_left_x, tmp_mask[:1]], dim=2)
                elif 'right' in mask_name:
                    image_right_x = torch.cat([image_right_x, tmp_mask[:1]], dim=2)
                else:
                    gripper = torch.cat([gripper, tmp_mask[:1]], dim=2)
            masks = torch.stack(masks, dim=1).flatten(-2, -1) > 0
            masks = masks.squeeze(2)

        # expand text dimension
        text_x = self.text_process_fn([goal])
        text_x = text_x.unsqueeze(1)

        states = [obs[state_name][-1] for state_name in self.state_names]
        state = np.concatenate(states, axis=0)
        state = torch.from_numpy(np.stack([state]))
        state = state.unsqueeze(1).to(dtype=self.cast_type)
        # state = torch.cat([state[..., :6], state[..., [-1]]], dim=-1)
        # if timestep == 0:
        #     print(state)
        # breakpoint()
        # state *= 0 # !!!

        with torch.no_grad():
            device = 'cuda'
            image_left_x = image_left_x.to(device)
            image_right_x = image_right_x.to(device)
            text_x = text_x.to(device)
            gripper = gripper.to(device)
            state = state.to(device)

            self.img_left_queue.append(image_left_x)
            self.img_right_queue.append(image_right_x)
            self.gripper_queue.append(gripper)
            self.state_queue.append(state)
            if len(self.text_queue) == 0 and text_x is not None:  # the instruction does not change
                self.text_queue.append(text_x)
                for _ in range(self.model.module.sequence_length - 1):
                    self.text_queue.append(text_x)
            
            image_left = torch.cat(list(self.img_left_queue), dim=1)
            image_right = torch.cat(list(self.img_right_queue), dim=1)
            image_wrist = torch.cat(list(self.gripper_queue), dim=1)
            state = torch.cat(list(self.state_queue), dim=1)
            input_text_token = torch.cat(list(self.text_queue), dim=1)

            num_step = image_left.shape[1]
            if num_step < 10:  # padding
                input_image_left = torch.cat([image_left, image_left[:, -1].repeat(1, 10-num_step, 1, 1, 1)], dim=1)
                input_image_right = torch.cat([image_right, image_right[:, -1].repeat(1, 10-num_step, 1, 1, 1)], dim=1)
                input_image_wrist = torch.cat([image_wrist, image_wrist[:, -1].repeat(1, 10-num_step, 1, 1, 1)], dim=1)
                input_state = torch.cat([state, state[:, -1].repeat(1, 10-num_step, 1)], dim=1)
                # input_text_token = torch.cat([input_text_token, input_text_token.repeat(1, 10-num_step, 1)], dim=1)
            else:
                input_image_left = image_left
                input_image_right = image_right
                input_image_wrist = image_wrist
                input_state = state
            arm_action, gripper_action, image_pred = self.model(
                image_left=input_image_left,
                image_right=input_image_right,
                image_wrist=input_image_wrist,
                state=input_state,
                text_token=input_text_token,
                masks=masks
            )
            action = torch.concat((arm_action[0, :, 0, :].clamp(min=-1., max=1.), gripper_action[0, :, 0, :] > 0.5), dim=-1)
            action[:, -1] = (action[:, -1] - 0.5) * 2  # scale to -1 or 1
            
            action = action.cpu().detach().to(dtype=torch.float16).numpy()
            if num_step < 10:
                action = action[num_step - 1]
            else:
                action = action[-1]  # which action is needed? depend on the timestep  
            action = np.concatenate([action, np.array([0., 0., 0., 0., -1.])], axis=0)
        # if self.tcp_rel:
        #     state = obs['robot_obs']
        #     state = torch.from_numpy(np.stack([state])).unsqueeze(0).float().cpu().detach()
        #     action = torch.from_numpy(np.stack([action])).unsqueeze(0).float().cpu().detach()
        #     action = tcp_to_world_frame(action, state)
        #     action=action.squeeze().to(dtype=torch.float16).numpy()
        return action


def evaluate_policy_ddp(model, env, epoch, calvin_conf_path, eval_log_dir=None, debug=False, create_plan_tsne=False, reset=False, diverse_inst=False, horizon=500, args=None):
    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    # assert NUM_SEQUENCES % device_num == 0
    interval_len = int((NUM_SEQUENCES + device_num - 1) // device_num)
    eval_ep_ids = range(device_id*interval_len, min((device_id+1)*interval_len, NUM_SEQUENCES))
    env_name = env.name
    video_writer = None
    if device_id == 0:
        print(f'start evaluating {env_name}...')
    video_dir = os.path.join(eval_log_dir, 'videos')
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"{env_name}_{device_id}.mp4")
    video_writer = imageio.get_writer(video_path, fps=20)
    if device_id == 0:
        eval_ep_ids = tqdm(eval_ep_ids, position=0, leave=True)

    rollout_logs = []
    num_success = 0
    num_contact = 0
    for ep_i in eval_ep_ids:
        # if ep_i != 0: # !!!
        #     continue
        initial_state = TrainUtils.VAL_ENV_INFOS[env_name][ep_i]
        rollout_timestamp = time.time()
        try:
            rollout_info = run_rollout(
                model=model,
                env=env,
                horizon=horizon,
                initial_state=initial_state,
                video_writer=video_writer,
                ep_i=ep_i,
                args=args
            )
        except Exception as e:
            print(traceback.format_exc())
            print(env_name, "Rollout exception at episode number {}!".format(ep_i))
            # break
            continue
        rollout_info["time"] = time.time() - rollout_timestamp
        rollout_logs.append(rollout_info)
        num_success += rollout_info["Success_Rate"]
        num_contact += rollout_info["Contact_Rate"]
        if device_id == 0:
            print(" horizon={}, num_success={}, num_contact={}".format(horizon, num_success, num_contact))

    

    all_rollout_logs = [None for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(rollout_logs, all_rollout_logs, dst=0)

    if torch.distributed.get_rank() == 0 and len(all_rollout_logs) > 0:
        tmp_all_rollout_logs = []
        for logs in all_rollout_logs:
            tmp_all_rollout_logs.extend(logs)
        all_rollout_logs = tmp_all_rollout_logs
        all_rollout_logs = dict((k, [all_rollout_logs[i][k] for i in range(len(all_rollout_logs))]) for k in all_rollout_logs[0])
        rollout_logs_mean = dict((k, np.mean(v)) for k, v in all_rollout_logs.items())
        rollout_logs_mean["Time_Episode"] = np.sum(all_rollout_logs["time"]) / 60. # total time taken for rollouts in minutes
        return rollout_logs_mean
    
    return {"Time_Episode": -1, "Return": -1, "Success_Rate": -1, "time": -1}


def run_rollout(
        model, 
        env, 
        horizon,
        initial_state=None,
        video_writer=None,
        video_skip=5,
        terminate_on_success=False,
        ep_i=0,
        args=None
    ):
    env.env.env.add_object_num = 0
    ob_dict = env.reset_to(initial_state)
    assert env.env.env.unique_attr == json.loads(initial_state["ep_meta"])["unique_attr"]   
    
    # policy.start_episode(lang=env._ep_lang_str)
    # print(env._ep_lang_str)
    lang_annotation = env._ep_lang_str
    model.reset()
    
    results = {}
    video_count = 0  # video frame counter

    rews = []
    success = None #{ k: False for k in env.is_success() } # success metrics

    end_step = None
    video_frames = []
    camera_names = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
    
    masked_dict = {}
    if args.addmask:
        # print('getting mask...')
        image_size = 256
        target_obj_str = env.env.env.target_obj_str
        if target_obj_str == "obj":
            target_obj_str += "_main"
        target_place_str = env.env.env.target_place_str
        
        geom2body_id_mapping = {geom_id: body_id for geom_id, body_id in enumerate(env.env.env.sim.model.geom_bodyid)}
        name2id = env.env.env.sim.model._body_name2id
        for cam_name in camera_names:
            seg = env.env.env.sim.render(
                camera_name=cam_name,
                width=image_size,
                height=image_size,
                depth=False,
                segmentation=True
            )
            seg = seg[::-1, :, 1]
            tmp_seg = (
                np.fromiter(
                    map(
                        lambda x: geom2body_id_mapping.get(x, -1),
                        seg.flatten()
                    ),
                    dtype=np.int32
                ).reshape(image_size, image_size)
            )
            tmp_mask = np.zeros(tmp_seg.shape, dtype=np.uint8)
            for tmp_target_obj_str in target_obj_str.split('/'):
                tmp_mask[tmp_seg == name2id[tmp_target_obj_str]] = 1
            if target_place_str:
                tmp_mask[tmp_seg == name2id[target_place_str]] = 2
                if (tmp_seg == name2id[target_place_str]).sum() == 0 and target_place_str == "container_main" and name2id[target_place_str] == name2id[None] - 1:
                    tmp_mask[tmp_seg == name2id[None]] = 2
            # tmp_mask = tmp_mask.astype(np.float32) / 2.
            # obj_mask = np.zeros((512, 512, 3), dtype=np.uint8)
            # obj_mask[tmp_mask == 1, 0] = 255
            # place_mask = np.zeros((512, 512, 3), dtype=np.uint8)
            # place_mask[tmp_mask == 2, 2] = 255
            # Image.fromarray(obj_mask).save('obj_mask.jpg')
            # Image.fromarray(place_mask).save('place_mask.jpg')
            # breakpoint()
            tmp_mask = np.expand_dims(tmp_mask, axis=0)
            tmp_mask = np.expand_dims(tmp_mask, axis=0).repeat(ob_dict[f"{cam_name}_image"].shape[0], axis=0)
            masked_dict[f"{cam_name}_mask"] = tmp_mask
    
    for step_i in range(horizon): #LogUtils.tqdm(range(horizon)):
        # for cam_name in camera_names:
        #     depth_name = f"{cam_name}_depth"
        #     _, depth = env.env.env.sim.render(
        #         camera_name=cam_name,
        #         width=args.image_size,
        #         height=args.image_size,
        #         depth=True
        #     )
        #     depth = np.expand_dims(depth[::-1], axis=0)
        #     if depth_name not in env.obs_history:
        #         env.obs_history[depth_name] = deque(
        #             [depth[None]] * env.num_frames,
        #             maxlen=env.num_frames
        #         )
        #     else:
        #         env.obs_history[depth_name].append(depth[None])
        #     ob_dict = env._get_stacked_obs_from_history()
        
        ob_dict.update(masked_dict)
        
        
        # get action from policy
        # ac = model(ob=ob_dict) #, return_ob=True)
        action = model.step(ob_dict, lang_annotation, step_i)
        # action[:6] *= 2 # !!!
        action[:6] = np.clip(action[:6], a_min=-1., a_max=1.)
        # try:
        #     action = f['data/demo_0']['actions'][step_i]
        # except:
        #     action = np.zeros(12)

        # play action
        ob_dict, r, done, info = env.step(action)

        # compute reward
        rews.append(r)

        cur_success_metrics = info["is_success"]

        if success is None:
            success = deepcopy(cur_success_metrics)
        else:
            for k in success:
                success[k] = success[k] | cur_success_metrics[k]

        # visualization
        if video_writer is not None:
            if video_count % video_skip == 0:
                frame = env.render(mode="rgb_array", height=512, width=512)
                frame = frame.copy()
                text1 = env._ep_lang_str
                position1 = (10, 50)
                color = (255, 0, 0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                thickness = 1
                font_scale = 0.5
                cv2.putText(frame, text1, position1, font, font_scale, color, thickness)
                text2 = f"Success: {success['task']}"
                position2 = (10, 100)
                cv2.putText(frame, text2, position2, font, font_scale, color, thickness)
                video_frames.append(frame)
            video_count += 1

        if done or (terminate_on_success and success["task"]):
            end_step = step_i
            break


    if video_writer is not None:
        for frame in video_frames:
            video_writer.append_data(frame)

    end_step = end_step or step_i
    total_reward = np.sum(rews[:end_step + 1])
    
    results["Return"] = total_reward
    results["Contact_Rate"] = max(rews)
    results["Horizon"] = end_step + 1
    results["Success_Rate"] = float(success["task"])

    return results


def eval_one_epoch_calvin_ddp(args, model, dataset_path, image_processor, tokenizer, eval_log_dir=None, debug=False, future_act_len=-1, reset=False, diverse_inst=False, robocasa_config=None):
    envs = make_env(dataset_path, robocasa_config)
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = 10
    wrapped_model = ModelWrapper(model, tokenizer, image_processor, cast_dtype, history_len=hist_len, tcp_rel=args.tcp_rel)
    
    all_env_logs = OrderedDict()
    for env, horizon in envs:
        all_env_logs[env.name] = evaluate_policy_ddp(wrapped_model, env, 0, args.calvin_conf_path, eval_log_dir=eval_log_dir, debug=debug, reset=reset, diverse_inst=diverse_inst, horizon=horizon, args=args)
    return all_env_logs


def env_iterator(eval_env_meta_list, eval_shape_meta_list, eval_env_name_list, eval_env_horizon_list, config):
    for (env_meta, shape_meta, env_name, env_horizon) in zip(eval_env_meta_list, eval_shape_meta_list, eval_env_name_list, eval_env_horizon_list):
        def create_env_helper(env_i=0):
            env_kwargs = dict(
                env_meta=env_meta,
                env_name=env_name,
                render=False,
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"],
                seed=config.train.seed * 1000 + env_i,
            )
            env = EnvUtils.create_env_from_metadata(**env_kwargs)
            # handle environment wrappers
            env = EnvUtils.wrap_env_from_config(env, config=config)  # apply environment warpper, if applicable

            return env

        if config.experiment.rollout.batched:
            from tianshou.env import SubprocVectorEnv
            env_fns = [lambda env_i=i: create_env_helper(env_i) for i in range(config.experiment.rollout.num_batch_envs)]
            env = SubprocVectorEnv(env_fns)
            # env_name = env.get_env_attr(key="name", id=0)[0]
        else:
            env = create_env_helper()
            # env_name = env.name
        # print(env)
        yield env, env_horizon
        
        
        
# [ 0.2556,  0.0116,  0.6040, -0.9878, -0.0376, -0.1487,  0.0255,
        #    0.0201, -0.0200]]]