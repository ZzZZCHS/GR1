import logging

import numpy as np
try:
    from pytorch3d.transforms import (
        euler_angles_to_matrix,
        matrix_to_euler_angles,
        matrix_to_quaternion,
        quaternion_to_matrix,
    )
except:
    print('no pytorch3d')
import torch
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)

import functools
import math
import io
import os
import random
import re
import pickle
from multiprocessing import Value
from functools import partial
import json
from itertools import chain
from dataclasses import dataclass
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info, Dataset
from torch.utils.data.distributed import DistributedSampler
try:
    from petrel_client.client import Client
except:
    print("no petrel_client")
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import bisect
from itertools import accumulate
import copy
from typing import List
from torchvision import transforms as torchtransforms
from PIL import Image
import clip
from scipy.spatial.transform import Rotation as R
import cv2
import h5py

Image.MAX_IMAGE_PIXELS = 1000000000
MAX_NUM_TOKENS = 256
MAX_NUM_IMAGES = 5
TINY_IMAGE_SIZE_THRESHOLD = 1
N_CHANNELS = 3
INTERLEAVED_IMAGE_SIZE = 224

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

MIN_KB = 10
MAX_NUM_IMAGES = 5
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Tuple, Callable, Union
import roboticstoolbox as rtb

import robomimic.utils.train_utils as TrainUtils

robot = rtb.models.Panda()
# print(robot)
# print(robot.link_dict.keys())
links = [robot.link_dict['panda_link1'], robot.link_dict['panda_link2'], robot.link_dict['panda_link3'], robot.link_dict['panda_link4'], robot.link_dict['panda_link5'], robot.link_dict['panda_link6'], robot.link_dict['panda_link7']]

obs_config = DictConfig(
    {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
)

prop_state = DictConfig(
    {
        "n_scene_obs": 24,
        "n_state_obs": 15,
        "keep_indices": [[0, 15]],
        "robot_orientation_idx": [3, 6],
        "normalize": True,
        "normalize_robot_orientation": True,
    }
)

def transformation_matrix_to_trans_and_euler(transformation_matrix):
    translation = transformation_matrix[:3, 3]
    rotation_matrix = transformation_matrix[:3, :3]

    euler = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)

    return translation, euler

def get_state_info_dict(episode: Dict[str, np.ndarray]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Create a dictionary with raw state observations for environment resets.

    Args:
        episode: Sequence dictionary.

    Returns:
         Info dict of full robot and scene state (for env resets).
    """
    return {
        "state_info": {
            "robot_obs": torch.from_numpy(episode["robot_obs"]),
            "scene_obs": torch.from_numpy(episode["scene_obs"]),
        }
    }

def process_state(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    proprio_state: DictConfig,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, torch.Tensor]:
    state_obs_keys = observation_space["state_obs"]
    state_obs_list_normalized = []
    state_obs_list_unnormalized = []
    for state_ob in state_obs_keys:
        if window_size == 0 and seq_idx == 0:  # single file loader
            state_tensor = torch.from_numpy(episode[state_ob]).float()
        else:  # episode loader
            state_tensor = torch.from_numpy(episode[state_ob][seq_idx : seq_idx + window_size]).float()
        # expand dims for single environment obs
        if len(state_tensor.shape) != 2:
            state_tensor = state_tensor.unsqueeze(0)
        # shape: (BxN_state_obs)
        assert len(state_tensor.shape) == 2
        if state_ob in transforms:
            state_tensor_normalized = transforms[state_ob](state_tensor)
            state_obs_list_normalized.append(state_tensor_normalized)
        else:
            state_obs_list_normalized.append(state_tensor)
        state_obs_list_unnormalized.append(state_tensor)
    seq_state_obs = torch.cat(state_obs_list_normalized, dim=1)
    seq_state_obs_unnormalized = torch.cat(state_obs_list_unnormalized, dim=1)

    if not proprio_state.normalize_robot_orientation and "robot_orientation_idx" in proprio_state:
        seq_state_obs[:, slice(*proprio_state.robot_orientation_idx)] = seq_state_obs_unnormalized[
            :, slice(*proprio_state.robot_orientation_idx)
        ]

    if not proprio_state.normalize:
        seq_state_obs = seq_state_obs_unnormalized

    # slice the specified parts of the proprioception state
    state_obs_sliced = []
    for slice_ids in proprio_state.keep_indices:
        seq_state_obs_ = seq_state_obs[:, slice(*slice_ids)]
        state_obs_sliced.append(seq_state_obs_)
    seq_state_obs = torch.cat(state_obs_sliced, dim=1)

    return {"robot_obs": seq_state_obs}

def preprocess_image(sample, image_processor):
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    # apply random horizontal flip and color jitter
    return image

def preprocess_text_calvin(sample, tokenizer):
    text = tokenizer.tokenize(sample, truncate=True)
    return text

def process_rgb(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    rgb_obs_keys = observation_space["rgb_obs"]

    seq_rgb_obs_dict = {}
    for _, rgb_obs_key in enumerate(rgb_obs_keys):
        rgb_obs = episode[rgb_obs_key]
        # expand dims for single environment obs
        if len(rgb_obs.shape) != 4:
            rgb_obs = np.expand_dims(rgb_obs, axis=0)
        assert len(rgb_obs.shape) == 4
        if window_size == 0 and seq_idx == 0:  # single file loader
            # To Square image
            seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte().permute(0, 3, 1, 2)
        else:  # episode loader
            seq_rgb_obs_ = torch.from_numpy(rgb_obs[seq_idx : seq_idx + window_size]).byte().permute(0, 3, 1, 2)
        # we might have different transformations for the different cameras
        if rgb_obs_key in transforms:
            seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
        seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
    # shape: N_rgb_obs x (BxCxHxW)
    return {"rgb_obs": seq_rgb_obs_dict}


def process_depth(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    # expand dims for single environment obs
    def exp_dim(depth_img):
        if len(depth_img.shape) != 3:
            depth_img = np.expand_dims(depth_img, axis=0)
        return depth_img

    depth_obs_keys = observation_space["depth_obs"]
    seq_depth_obs_dict = {}
    for _, depth_obs_key in enumerate(depth_obs_keys):
        depth_ob = exp_dim(episode[depth_obs_key])
        assert len(depth_ob.shape) == 3
        if window_size == 0 and seq_idx == 0:  # single file loader
            depth_ob_ = torch.from_numpy(depth_ob).float()
        else:  # episode loader
            depth_ob_ = torch.from_numpy(depth_ob[seq_idx : seq_idx + window_size]).float()
        # we might have different transformations for the different cameras
        if depth_obs_key in transforms:
            depth_ob_ = transforms[depth_obs_key](depth_ob_)
        seq_depth_obs_dict[depth_obs_key] = depth_ob_
    # shape: N_depth_obs x(BxHxW)
    return {"depth_obs": seq_depth_obs_dict}


def process_actions(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, torch.Tensor]:
    # shape: (N_actions)
    action_keys = observation_space["actions"]
    if len(action_keys) != 1:
        raise NotImplementedError
    action_key = action_keys[0]
    if window_size == 0 and seq_idx == 0:  # single file loader
        action = episode[action_key]
        if "actions" in transforms:
            action = transforms["actions"]((action, episode["robot_obs"]))
        seq_acts = torch.from_numpy(action).float()
    else:  # episode loader
        seq_acts = torch.from_numpy(episode[action_key][seq_idx : seq_idx + window_size]).float()
    return {"actions": seq_acts}


def process_language(episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool) -> Dict[str, torch.Tensor]:
    seq_lang = {"lang": torch.empty(0)}
    if with_lang:
        lang = torch.from_numpy(episode["language"]).float()
        if "language" in transforms:
            lang = transforms["language"](lang)
        seq_lang["lang"] = lang
    return seq_lang

def lookup_naming_pattern(dataset_dir: Path, save_format: str) -> Tuple[Tuple[Path, str], int]:
    """
    Check naming pattern of dataset files.

    Args:
        dataset_dir: Path to dataset.
        save_format: File format (CALVIN default is npz).

    Returns:
        naming_pattern: 'file_0000001.npz' -> ('file_', '.npz')
        n_digits: Zero padding of file enumeration.
    """
    it = os.scandir(dataset_dir)
    while True:
        filename = Path(next(it))
        if save_format in filename.suffix:
            break
    aux_naming_pattern = re.split(r"\d+", filename.stem)
    naming_pattern = (filename.parent / aux_naming_pattern[0], filename.suffix)
    n_digits = len(re.findall(r"\d+", filename.stem)[0])
    assert len(naming_pattern) == 2
    assert n_digits > 0
    return naming_pattern, n_digits

def load_partial_traj_data():
    with open('utils/partial_task_data.json', 'r') as f:
        data = json.load(f)
    return data

def subtract_ranges(rangeA, rangeB):
    def subtract_single_range(a, b):
        result = []
        a_start, a_end = a
        b_start, b_end = b

        if b_start > a_end or b_end < a_start:
            # No overlap
            return [a]
        if b_start > a_start:
            result.append((a_start, min(a_end, b_start - 1)))
        if b_end < a_end:
            result.append((max(a_start, b_end + 1), a_end))

        return [r for r in result if r[0] <= r[1]]

    result = rangeA
    for b in rangeB:
        new_result = []
        for a in result:
            new_result.extend(subtract_single_range(a, b))
        result = new_result

    return result

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

    def forward_traj(self, x):
        n, t, c, h, w = x.size()
        x = x.view(n*t, *x.shape[2:])
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        base_grid = base_grid.unsqueeze(1).repeat(1, t, 1, 1, 1)
        base_grid = base_grid.view(n*t, *base_grid.shape[2:])
        shift = torch.randint(1,
                              2 * self.pad + 1,
                              size=(n*t, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        x = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        x = x.view(n, t, *x.shape[1:])
        return x

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


class BaseCalvinDataset(Dataset):
    """
    Abstract dataset base class.

    Args:
        datasets_dir: Path of folder containing episode files (string must contain 'validation' or 'training').
        obs_space: DictConfig of observation space.
        proprio_state: DictConfig with shape of prioprioceptive state.
        key: 'vis' or 'lang'.
        lang_folder: Name of the subdirectory of the dataset containing the language annotations.
        num_workers: Number of dataloading workers for this dataset.
        transforms: Dict with pytorch data transforms.
        batch_size: Batch size.
        min_window_size: Minimum window length of loaded sequences.
        max_window_size: Maximum window length of loaded sequences.
        pad: If True, repeat last frame such that all sequences have length 'max_window_size'.
        aux_lang_loss_window: How many sliding windows to consider for auxiliary language losses, counted from the end
            of an annotated language episode.
    """

    def __init__(
        self,
        datasets_dir: Path,
        *args: Any,
        proprio_state: DictConfig = prop_state,
        lang_folder: str = "lang_annotations",
        num_workers: int = 0,
        key: str = "lang",
        obs_space: DictConfig = obs_config,
        transforms: Dict = {},
        batch_size: int = 32,
        window_size: int = 16,
        min_window_size: int = 16,
        max_window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        rgb_pad=-1,
        gripper_pad=-1,
        traj_cons=False,
        text_aug=False,
        dif_ws=False,
        act_step=1,
        data_in_ceph=False,
        **kwargs: Any,
    ):
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.except_lang = key == "except_lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]

        self.pad = pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size
        if not dif_ws:
            self.min_window_size = window_size + act_step - 1
            self.max_window_size = window_size + act_step - 1
        else:
            self.min_window_size = min_window_size
            self.max_window_size = max_window_size
        self.act_step = act_step
        # print('ws {}, min_ws {}, max_ws {}'.format(self.window_size, self.max_window_size, self.min_window_size))
        self.abs_datasets_dir = datasets_dir
        self.lang_folder = lang_folder  # if self.with_lang else None
        self.aux_lang_loss_window = aux_lang_loss_window
        self.traj_cons = traj_cons
        self.data_in_ceph = data_in_ceph
        if self.data_in_ceph:
            self.conf_path = '~/petreloss.conf'
            self.client = Client(self.conf_path)
       
        with open('./utils/enrich_lang_annotations.json', 'r') as f:
            self.enrich_lang = json.load(f)
        self.text_aug = text_aug

        self.rgb_pad = rgb_pad
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)

        if self.data_in_ceph:
            assert (
                "validation" in self.abs_datasets_dir
                or "training" in self.abs_datasets_dir
            )
            self.validation = "validation" in self.abs_datasets_dir
        else:
            assert (
                "validation" in self.abs_datasets_dir.as_posix()
                or "training" in self.abs_datasets_dir.as_posix()
            )
            self.validation = "validation" in self.abs_datasets_dir.as_posix()
       #  print("data_in_ceph", data_in_ceph)
        if self.data_in_ceph:
            # print("data_in_ceph")
            assert self.client.isdir(self.abs_datasets_dir)
        else:
            assert self.abs_datasets_dir.is_dir()
        print(f"loading dataset at {self.abs_datasets_dir}")
        print("finished loading dataset")

    def process_rgb(
        self,
        episode: Dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: Dict,
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        rgb_obs_keys = observation_space["rgb_obs"]
        seq_rgb_obs_dict = {}
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            # expand dims for single environment obs
            if len(rgb_obs.shape) != 4:
                rgb_obs = np.expand_dims(rgb_obs, axis=0)
            assert len(rgb_obs.shape) == 4
            if window_size == 0 and seq_idx == 0:  # single file loader
                # To Square image
                seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
            else:  # episode loader
                seq_rgb_obs_ = torch.from_numpy(
                    rgb_obs[seq_idx : seq_idx + window_size]
                ).byte()
            
            if rgb_obs_key in transforms:
                seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
        # shape: N_rgb_obs x (BxHxWxC)
        return {"rgb_obs": seq_rgb_obs_dict}

    def process_language(
        self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
    ):
        return {"lang": episode["language"]} if with_lang else {"lang": 'no lang'}

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        # if dist.get_rank() == 0:
        # set_trace()
        if isinstance(idx, int):
            # When max_ws_size and min_ws_size are equal, avoid unnecessary padding
            # acts like Constant dataset. Currently, used for language data

            # 默认中min_window_size, max_window_size为12
            # 
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            elif self.min_window_size < self.max_window_size:
                window_size = self._get_window_size(idx)
            else:
                print(
                    f"min_window_size {self.min_window_size} > max_window_size {self.max_window_size}"
                )
                raise ValueError
        else:
            idx, window_size = idx
        
        head = False
        sequence = self._get_sequences(idx, window_size, head=head)

        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size, head=head)
        
        import copy
        new_list = []
        np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
        for i in range(np_rgb.shape[0]):
            new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_static"] = new_list
        new_list = []
        np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
        for i in range(np_gripper.shape[0]):
            new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_gripper"] = new_list

        sequence["self_key_points"] = []
        for i in range(window_size):
            self_key_points = []
            for link in links:
                Te = robot.fkine(sequence['robot_obs'][i][7:14], end=link)  # forward kinematics
                translation, euler = transformation_matrix_to_trans_and_euler(np.array(Te))
                self_key_points.append(np.concatenate([translation, euler]))
            sequence["self_key_points"].append(self_key_points)
        return sequence

    def _get_sequences(self, idx: int, window_size: int, head: bool=False) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.
            window_size: Length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """

        episode = self._load_episode(idx, window_size)

        # here episode is a dict with keys "rgb_static", "rgb_gripper", "robot_obs", "rel_actions", "scene_obs", "language"
        # rbg_static : [window_size, 200, 200, 3]
        # rgb_gripper : [window_size, 84, 84, 3]
        # robot_obs : [window_size, 15]
        # rel_actions : [window_size, 7]
        # scene_obs : [window_size, 24]
        # language : str

        ### default中没有任何调整
        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )

        ### default中process_rgb将rgb转成tensor ###
        seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)

        ### default中depth_obs为空 ###
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)

        ### default中process_actions将rel_actions转换成tensor ###
        seq_acts = process_actions(episode, self.observation_space, self.transforms)

        ### 获得了robot_obs与scene_obs信息 ###
        info = get_state_info_dict(episode)

        ### default中process_language对language进行恒同操作 ###
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)


        info = self._add_language_info(info, idx)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
        }  # type:ignore
        seq_dict["idx"] = idx  # type:ignore

        return seq_dict

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def _get_window_size(self, idx: int) -> int:
        """
        Sample a window size taking into account the episode limits.

        Args:
            idx: Index of the sequence to load.

        Returns:
            Window size.
        """
        window_diff = self.max_window_size - self.min_window_size
        if len(self.episode_lookup) <= idx + window_diff:
            # last episode
            max_window = self.min_window_size + len(self.episode_lookup) - idx - 1
        elif (
            self.episode_lookup[idx + window_diff]
            != self.episode_lookup[idx] + window_diff
        ):
            # less than max_episode steps until next episode
            steps_to_next_episode = int(
                np.nonzero(
                    self.episode_lookup[idx : idx + window_diff + 1]
                    - (self.episode_lookup[idx] + np.arange(window_diff + 1))
                )[0][0]
            )
            max_window = min(
                self.max_window_size, (self.min_window_size + steps_to_next_episode - 1)
            )
        else:
            max_window = self.max_window_size

        return np.random.randint(self.min_window_size, max_window + 1)

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return len(self.episode_lookup)

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.max_window_size - len(sequence["actions"])

    def _pad_sequence(self, seq: Dict, pad_size: int, head: bool=False) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update(
            {
                "rgb_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["rgb_obs"].items()
                }
            }
        )
        seq.update(
            {
                "depth_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["depth_obs"].items()
                }
            }
        )

        if not self.relative_actions:
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                # repeat action for world coordinates action space
                seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size, head)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                seq_acts = torch.cat(
                    [
                        self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head),
                        self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
                    ],
                    dim=-1,
                )
            seq.update({"actions": seq_acts})
        seq.update(
            {
                "state_info": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["state_info"].items()
                }
            }
        )
        return seq

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        if head:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((last_repeated, input_tensor))
        else:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        if head:
            padded = torch.vstack((zeros_repeated, input_tensor))
        else:
            padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def _add_language_info(self, info: Dict, idx: int) -> Dict:
        """
        If dataset contains language, add info to determine if this sequence will be used for the auxiliary losses.

        Args:
            info: Info dictionary.
            idx: Sequence index.

        Returns:
            Info dictionary with updated information.
        """
        if not self.with_lang:
            return info
        use_for_aux_lang_loss = (
            idx + self.aux_lang_loss_window >= len(self.lang_lookup)
            or self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window]
        )
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None
    dataset: Dataset = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

class DiskCalvinDataset(BaseCalvinDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        image_fn: Callable,
        text_fn: Callable,
        *args: Any,
        skip_frames: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        partial_data=False,
        # data_in_ceph=False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_format = save_format
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.partial_data = partial_data
        if self.save_format == "pkl":
            self.load_file = self.load_pkl
        elif self.save_format == "npz":
            self.load_file = partial(self.load_npz, data_in_ceph=self.data_in_ceph)
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.skip_frames = skip_frames

        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_lookup,
                self.lang_ann,
                self.lang_task
            ) = self._build_file_indices_lang()
        elif self.except_lang:
            self.episode_lookup = self._build_file_indices_except_lang()
        else:
            self.episode_lookup = self._build_file_indices()

        if self.data_in_ceph:
            self.naming_pattern, self.n_digits = self.ceph_lookup_naming_pattern()
        else:
            self.naming_pattern, self.n_digits = lookup_naming_pattern(
                self.abs_datasets_dir, self.save_format
            )

        print("self.naming_pattern", self.naming_pattern)
        print("self.n_digits, ", self.n_digits)
    
    def ceph_lookup_naming_pattern(self):
        filenames = self.client.list(self.abs_datasets_dir)
        for filename in filenames:
            if self.save_format in filename:
                break
        filename = self.abs_datasets_dir + f"/{filename}"
        suffix = "." + self.save_format
        stem_suffix = filename.split('/')[-1]
        stem = stem_suffix.replace(suffix, "")
        aux_naming_pattern = re.split(r"\d+", stem)
        # print("ceph filename", filename)
        naming_pattern = (filename.replace(stem_suffix, aux_naming_pattern[0]), suffix)
        n_digits = len(re.findall(r"\d+", stem)[0])
        assert len(naming_pattern) == 2
        assert n_digits > 0
        return naming_pattern, n_digits

    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.
        Args:
            file_idx: index of starting frame.
        Returns:
            Path to file.
        """
        if self.data_in_ceph:
            return f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        else:
            return Path(
                f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
            )

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        # set_trace()
        start_idx = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episodes = [
            self.load_file(self._get_episode_name(file_idx))
            for file_idx in range(start_idx, end_idx)
        ]

        # # print(episodes[0].keys())
        # print("actions: ", episodes[0]["actions"])
        # print("rel_actions: ", episodes[0]["rel_actions"])
        # print("robot_obs: ", episodes[0]["robot_obs"])
        # print("keys: ", keys)

        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
                episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, # abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        abs_datasets_dir = self.abs_datasets_dir
        if self.data_in_ceph:
            assert self.client.isdir(abs_datasets_dir)
        else:
            assert abs_datasets_dir.is_dir()

        episode_lookup = []

        try:
            if self.data_in_ceph:
                print(
                "trying to load lang data from: ",
                abs_datasets_dir +f"/{self.lang_folder}/auto_lang_ann.npy",
                )
                lang_data_bytes = self.client.get(abs_datasets_dir+f"/{self.lang_folder}/auto_lang_ann.npy", enable_cache=True)
                lang_data = io.BytesIO(lang_data_bytes)
                lang_data = np.load(lang_data, allow_pickle=True).item()
            else:
                print(
                "trying to load lang data from: ",
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                )
                lang_data = np.load(
                    abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                    allow_pickle=True,
                ).item()
        except Exception:
            if self.data_in_ceph:
                print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir + "/auto_lang_ann.npy",
                )
                lang_data_bytes = self.client.get(abs_datasets_dir+f"/auto_lang_ann.npy", enable_cache=True)
                lang_data = io.BytesIO(lang_data_bytes)
                lang_data = np.load(lang_data, allow_pickle=True).item()
            else:
                print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir / "auto_lang_ann.npy",
                )
                lang_data = np.load(
                    abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True
                ).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        lang_ann = lang_data["language"]["ann"]  # length total number of annotations
        lang_task = lang_data["language"]["task"]
        lang_lookup = []
        partial_st_ed_list = load_partial_traj_data()
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            if self.partial_data:
                if [start_idx, end_idx] not in partial_st_ed_list:
                    continue
            if self.pretrain:
                start_idx = max(
                    start_idx,
                    end_idx + 1 - self.min_window_size - self.aux_lang_loss_window,
                )
            assert end_idx >= self.max_window_size
            cnt = 0
            
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                cnt += 1

        length = 0
        for idx in ep_start_end_ids:
            length += idx[1] - idx[0]
        print("num data frames:", length)
        
        return np.array(episode_lookup), lang_lookup, lang_ann, lang_task

    def _build_file_indices(self) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        abs_datasets_dir = self.abs_datasets_dir
        if self.data_in_ceph:
            self.client.isdir(abs_datasets_dir)
        else:
            assert abs_datasets_dir.is_dir()

        episode_lookup = []

        if self.data_in_ceph:
            lang_data_bytes = self.client.get(abs_datasets_dir+f"ep_start_end_ids.npy", enable_cache=True)
            lang_data = io.BytesIO(lang_data_bytes)
            ep_start_end_ids = np.load(lang_data)
        else:
            ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        print(
            f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.'
        )

        length = 0
        for idx in ep_start_end_ids:
            length += idx[1] - idx[0]
        print("num data frames:", length)

        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)
    
    def _build_file_indices_except_lang(self) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        abs_datasets_dir = self.abs_datasets_dir
        if self.data_in_ceph:
            self.client.isdir(abs_datasets_dir)
        else:
            assert abs_datasets_dir.is_dir()
        
        lang_data = np.load(
            abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
            allow_pickle=True,
        ).item()
        lang_ep_start_end_ids = lang_data["info"]["indx"]

        episode_lookup = []

        if self.data_in_ceph:
            lang_data_bytes = self.client.get(abs_datasets_dir+f"ep_start_end_ids.npy", enable_cache=True)
            lang_data = io.BytesIO(lang_data_bytes)
            ep_start_end_ids = np.load(lang_data)
        else:
            ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        print(
            f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.'
        )

        # ep_start_end_ids = subtract_ranges(np.array(ep_start_end_ids).tolist(), np.array(lang_ep_start_end_ids).tolist())
        # np.save(abs_datasets_dir / "except_lang_idx" / "except_lang_idx.npy", np.array(ep_start_end_ids))
        ep_start_end_ids = np.load(abs_datasets_dir / "except_lang_idx" / "except_lang_idx.npy").tolist()
        length = 0
        for idx in ep_start_end_ids:
            length += idx[1] - idx[0]
        print("num data frames:", length)

        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)

    def collator(self, sample):
        action_tensors = torch.from_numpy(np.array([np.stack(s["actions"]) for s in sample]))
        state_tensors = torch.from_numpy(np.array([np.stack(s["robot_obs"]) for s in sample]))
        self_key_point_tensors = torch.from_numpy(np.array([np.stack(s["self_key_points"]) for s in sample]))
        # for i in range(13):
        #     sample[1]["rgb_obs"]["rgb_static"][i].save(f'cprimary_{i}.png')
        #     sample[1]["rgb_obs"]["rgb_gripper"][i].save(f'cgripper_{i}.png')
        image_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
        gripper_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
        stacked_language = [s["lang"] for s in sample]
        text_tensors = self.text_fn(stacked_language)

        if self.rgb_pad != -1:
            bs, seq_len = image_tensors.shape[:2]
            if self.traj_cons:
                image_tensors = self.rgb_shift.forward_traj(image_tensors)
            else:
                image_tensors = image_tensors.view(bs*seq_len, *image_tensors.shape[2:])
                image_tensors = self.rgb_shift(image_tensors)
                image_tensors = image_tensors.view(bs, seq_len, *image_tensors.shape[1:])
        if self.gripper_pad != -1:
            bs, seq_len = gripper_tensors.shape[:2]
            if self.traj_cons:
                gripper_tensors = self.gripper_shift.forward_traj(gripper_tensors)
            else:
                gripper_tensors = gripper_tensors.view(bs * seq_len, *gripper_tensors.shape[2:])
                gripper_tensors = self.gripper_shift(gripper_tensors)
                gripper_tensors = gripper_tensors.view(bs, seq_len, *gripper_tensors.shape[1:])
        
        robot_obs = torch.zeros(1)
        
        if self.act_step != 1:
        
            actions = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, action_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    actions[b, ix] = action_tensors[b, ix:ix+self.act_step]

            robot_obs = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, state_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    robot_obs[b, ix] = state_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.cat([robot_obs[..., :6], robot_obs[..., [-1]]], dim=-1)

            action_tensors = actions
            image_tensors = image_tensors[:, :-(self.act_step-1)]
            gripper_tensors = gripper_tensors[:, :-(self.act_step-1)]
            state_tensors = state_tensors[:, :-(self.act_step-1)]
        
        return image_tensors, text_tensors, action_tensors, gripper_tensors, state_tensors, self_key_point_tensors

    def load_pkl(self, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def load_npz(self, filename, data_in_ceph=False):
        if not data_in_ceph:
            return np.load(filename.as_posix())
        else:
            # print("filename, ", filename)
            data_bytes = self.client.get(filename, enable_cache=True)
            data = io.BytesIO(data_bytes)
            # print("data, ", data)

            # lang_data_bytes = self.client.get(abs_datasets_dir+f"/{self.lang_folder}/auto_lang_ann.npy", enable_cache=True)
            # lang_data = io.BytesIO(lang_data_bytes)
            # lang_data = np.load(lang_data, allow_pickle=True).item()
            try:
                data = np.load(data, allow_pickle=True)
            except:
                data = np.load(data)
            return data

def get_calvin_dataset(args, image_processor, tokenizer, epoch=0, floor=False, key='lang'):
    dataset_path = args.calvin_dataset

    # ann is dict including language and info
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    if args.data_in_ceph:
        datasets_dir = dataset_path + "/training"
    else:
        datasets_dir = Path(dataset_path) / "training"
    calvin_dataset = DiskCalvinDataset(
        datasets_dir=datasets_dir,
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        window_size=args.window_size,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        act_step=args.multi_step_action,
        partial_data=args.partial_data,
        data_in_ceph=args.data_in_ceph,
        key=key,
    )

    round_fn = math.floor if floor else math.ceil

    num_samples = len(calvin_dataset)
    global_batch_size = args.batch_size_calvin * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    
    num_workers = max(1, args.workers)

    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size

    sampler = DistributedSampler(
        calvin_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    # the batch_size and num_workers are per-GPU !
    # num_workers = 0
    dataloader = DataLoader(
        calvin_dataset,
        batch_size=args.batch_size_calvin,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=calvin_dataset.collator,
        drop_last=True
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=calvin_dataset)


def get_robomimic_dataset(args, image_processor, tokenizer, epoch=0, floor=False, key='lang', config=None):
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    
    trainset, _ = TrainUtils.load_data_for_training(
        config, obs_keys=config.all_obs_keys, lang_encoder=preprocess_text_fn)
    
    round_fn = math.floor if floor else math.ceil
    num_samples = len(trainset)
    global_batch_size = args.batch_size_calvin * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    
    num_workers = max(1, args.workers)
    
    num_worker_batches = round_fn(num_batches / num_workers)
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    
    sampler = DistributedSampler(
        trainset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True
    )
    
    collator_fn = functools.partial(robomimic_collator, image_processor=preprocess_image_fn, args=args)
    
    dataloader = DataLoader(
        trainset,
        batch_size=args.batch_size_calvin,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=collator_fn,
        drop_last=True
    )
    
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=trainset)


def get_calvin_val_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    dataset_path = args.calvin_dataset

    # ann is dict including language and info
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    if args.data_in_ceph:
        datasets_dir = dataset_path + "/validation"
    else:
        datasets_dir = Path(dataset_path) / "validation"

    calvin_dataset = DiskCalvinDataset(
        datasets_dir=datasets_dir,
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        window_size=args.window_size,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        act_step=args.multi_step_action,
        partial_data=args.partial_data,
        data_in_ceph=args.data_in_ceph
    )

    round_fn = math.floor if floor else math.ceil

    num_samples = len(calvin_dataset)
    global_batch_size = args.batch_size_calvin * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    
    num_workers = max(1, args.workers)

    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size

    sampler = DistributedSampler(
        calvin_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=False,
        seed=args.seed,
        drop_last=True,
    )
    # the batch_size and num_workers are per-GPU !
    dataloader = DataLoader(
        calvin_dataset,
        batch_size=args.batch_size_calvin,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        # sampler=sampler,
        shuffle=False,
        persistent_workers=True,
        collate_fn=calvin_dataset.collator,
        drop_last=True
    )
    # dataloader = DataLoader(calvin_dataset, batch_size=args.batch_size_calvin)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=calvin_dataset)


def world_to_tcp_frame(action, robot_obs):
    with autocast(dtype=torch.float32):
        flag = False
        if len(action.shape) == 4:
            flag = True
            b, s, f, _ = action.shape
            action = action.view(b, s*f, -1)
            robot_obs = robot_obs.view(b, s*f, -1)
        b, s, _ = action.shape
        world_T_tcp = euler_angles_to_matrix(robot_obs[..., 3:6], convention="XYZ").float().view(-1, 3, 3)
        tcp_T_world = torch.inverse(world_T_tcp)
        pos_w_rel = action[..., :3].view(-1, 3, 1)
        pos_tcp_rel = tcp_T_world @ pos_w_rel
        # downscaling is necessary here to get pseudo infinitesimal rotation
        orn_w_rel = action[..., 3:6] * 0.01
        world_T_tcp_new = (
            euler_angles_to_matrix(robot_obs[..., 3:6] + orn_w_rel, convention="XYZ").float().view(-1, 3, 3)
        )
        tcp_new_T_tcp_old = torch.inverse(world_T_tcp_new) @ world_T_tcp
        orn_tcp_rel = matrix_to_euler_angles(tcp_new_T_tcp_old, convention="XYZ").float()
        orn_tcp_rel = torch.where(orn_tcp_rel < -np.pi, orn_tcp_rel + 2 * np.pi, orn_tcp_rel)
        orn_tcp_rel = torch.where(orn_tcp_rel > np.pi, orn_tcp_rel - 2 * np.pi, orn_tcp_rel)
        # upscaling again
        orn_tcp_rel *= 100
        action_tcp = torch.cat([pos_tcp_rel.view(b, s, -1), orn_tcp_rel.view(b, s, -1), action[..., -1:]], dim=-1)
        if flag:
            action_tcp = action_tcp.view(b, s, -1, action_tcp.shape[-1])
        assert not torch.any(action_tcp.isnan())
    return action_tcp


def tcp_to_world_frame(action, robot_obs):
    with autocast(dtype=torch.float32):
        flag = False
        if len(action.shape) == 4:
            flag = True
            b, s, f, _ = action.shape
            action = action.view(b, s*f, -1)
            robot_obs = robot_obs.view(b, s*f, -1)
        b, s, _ = action.shape
        world_T_tcp = euler_angles_to_matrix(robot_obs[..., 3:6], convention="XYZ").float().view(-1, 3, 3)
        pos_tcp_rel = action[..., :3].view(-1, 3, 1)
        pos_w_rel = world_T_tcp @ pos_tcp_rel
        # downscaling is necessary here to get pseudo infinitesimal rotation
        orn_tcp_rel = action[..., 3:6] * 0.01
        tcp_new_T_tcp_old = euler_angles_to_matrix(orn_tcp_rel, convention="XYZ").float().view(-1, 3, 3)
        world_T_tcp_new = world_T_tcp @ torch.inverse(tcp_new_T_tcp_old)

        orn_w_new = matrix_to_euler_angles(world_T_tcp_new, convention="XYZ").float()
        if torch.any(orn_w_new.isnan()):
            logger.warning("NaN value in euler angles.")
            orn_w_new = matrix_to_euler_angles(
                quaternion_to_matrix(matrix_to_quaternion(world_T_tcp_new)), convention="XYZ"
            ).float()
        orn_w_rel = orn_w_new - robot_obs[..., 3:6].view(-1, 3)
        orn_w_rel = torch.where(orn_w_rel < -np.pi, orn_w_rel + 2 * np.pi, orn_w_rel)
        orn_w_rel = torch.where(orn_w_rel > np.pi, orn_w_rel - 2 * np.pi, orn_w_rel)
        # upscaling again
        orn_w_rel *= 100
        action_w = torch.cat([pos_w_rel.view(b, s, -1), orn_w_rel.view(b, s, -1), action[..., -1:]], dim=-1)
        if flag:
            action_w = action_w.view(b, s, -1, action_w.shape[-1])
        assert not torch.any(action_w.isnan())
    return action_w


def robomimic_collator(samples, image_processor, args):
    image_names = ['robot0_agentview_left_image', 'robot0_agentview_right_image', 'robot0_eye_in_hand_image']
    mask_names = ['robot0_agentview_left_mask', 'robot0_agentview_right_mask', 'robot0_eye_in_hand_mask']
    state_names = ['robot0_base_to_eef_pos', 'robot0_base_to_eef_quat', 'robot0_gripper_qpos']
    images_dict = {}
    for image_name in image_names:
        if image_name not in samples[0]['obs']:
            tmp_image_name = 'robot0_agentview_left_image'
        else:
            tmp_image_name = image_name
        batch_seq_images = torch.tensor(np.array([sample['obs'][tmp_image_name][:args.window_size, ...] for sample in samples]))
        # batch_seq_images = batch_seq_images.view(-1, 256, 256, 3).permute(0, 3, 1, 2)
        # batch_seq_images = torch.nn.functional.interpolate(batch_seq_images, size=(224, 224), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).view(-1, args.window_size, 224, 224, 3)
        # exit()
        image_list = batch_seq_images.flatten(0, 1).unbind(dim=0)
        image_list = [Image.fromarray(image.numpy().astype(np.uint8)) for image in image_list]
        processed_images = image_processor(image_list)
        images_dict[image_name] = processed_images.reshape(args.batch_size_calvin, args.window_size, 3, 224, 224)
    images_left = images_dict['robot0_agentview_left_image']
    images_right = images_dict['robot0_agentview_right_image']
    images_wrist = images_dict['robot0_eye_in_hand_image']
    
    # Image.fromarray((images_left[0][0].permute(1, 2, 0).numpy()[:,:,::-1]*255).astype(np.uint8)).save('tmp_image.jpg')
    # exit()
    
    masks = None
    if args.addmask:
        masks = []
        for mask_name in mask_names:
            tmp_mask = np.array([sample['obs'][mask_name][:args.window_size, ...] for sample in samples])
            target_obj_mask = torch.tensor((tmp_mask == 1)).squeeze(-1).to(torch.float32)
            target_place_mask = torch.tensor((tmp_mask == 2)).squeeze(-1).to(torch.float32)
            target_obj_mask = torch.nn.functional.interpolate(target_obj_mask, size=(14, 14), mode='bicubic', align_corners=True)
            target_place_mask = torch.nn.functional.interpolate(target_place_mask, size=(14, 14), mode='bicubic', align_corners=True)
            masks.extend([target_obj_mask, target_place_mask])
        masks = torch.stack(masks, dim=2).flatten(-2, -1) > 0
    
    text_tokens = torch.tensor(np.array([sample['obs']['lang_emb'][:args.window_size, :] for sample in samples]))
    
    states = []
    for state_name in state_names:
        tmp_state = torch.tensor(np.array([sample['obs'][state_name][:args.window_size, :] for sample in samples]))
        states.append(tmp_state)
    states = torch.cat(states, dim=-1).to(torch.float32)

    if 'real_actions' in samples[0]:
        actions = torch.tensor(np.array([sample['actions'][:args.window_size, :8] for sample in samples])).to(torch.float32)
        actions[..., 7:] = (actions[..., 7:] > 0)
    else:
        actions = torch.tensor(np.array([sample['actions'][:args.window_size, :7] for sample in samples])).to(torch.float32)
        actions[..., 6:] = (actions[..., 6:] + 1) // 2
    
    
    return {
        "images_left": images_left,
        "images_right": images_right,
        "images_wrist": images_wrist,
        "text_tokens": text_tokens,
        "states": states,
        "actions": actions,
        "masks": masks
    }
    
    
if __name__ == "__main__":
    from utils.arguments_utils import get_args
    from tqdm import tqdm

    args = get_args()

    device='cuda'
    model, image_processor = clip.load("ViT-B/32", device=device)
    calvin_dataset = get_calvin_dataset(args, image_processor, clip, epoch=0)
    calvin_dataset.set_epoch(epoch=0)
    calvin_loader = calvin_dataset.dataloader
    num_batches_per_epoch = calvin_loader.num_batches

    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=num_batches_per_epoch,
    )
    mv_avg_loss = []
    for num_steps, batch in t:
        print(11)