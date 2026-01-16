# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################################################################################
# Utilities
########################################################################################

import pickle
import logging
import time
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache
import h5py
import cv2
import torch
from deepdiff import DeepDiff
from termcolor import colored

from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_features_from_robot
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, has_method
from lerobot.common.datasets.my_utils import get_point_cloud


def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1 / dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    # TODO(aliberts): move robot-specific logs logic in robot.print_logs()
    if not robot.robot_type.startswith("stretch"):
        for name in robot.leader_arms:
            key = f"read_leader_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRlead", robot.logs[key])

        for name in robot.follower_arms:
            key = f"write_follower_{name}_goal_pos_dt_s"
            if key in robot.logs:
                log_dt("dtWfoll", robot.logs[key])

            key = f"read_follower_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRfoll", robot.logs[key])

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    logging.info(info_str)


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def predict_action(observation, policy, device, use_amp, robot, hdf5_path=None, demo_idx=0, current_step=0):
    if hdf5_path is not None:
        try:
            with h5py.File(hdf5_path, 'r') as f:
                demo_key = f'data/demo_{demo_idx+1}'  # 你的HDF5结构是 data/demo_1, data/demo_2, ...
                
                if demo_key in f:
                    demo_data = f[demo_key]
                    total_frames = demo_data['obs']['cam_right'].shape[0]
                    frame_idx = current_step % total_frames
                    # 读取图像数据 - 根据你的HDF5结构
                    cam_right = demo_data['obs']['cam_right'][frame_idx:frame_idx+2]  # 形状应该是 (84, 84, 3)
                    cam_left = demo_data['obs']['cam_left'][frame_idx:frame_idx+2]
                    cam_arm = demo_data['obs']['cam_arm'][frame_idx:frame_idx+2]
                    state = demo_data['obs']['state'][frame_idx:frame_idx+2]
                
                    # 转换为torch tensor并调整格式
                    observation = {
                        'cam_right': torch.from_numpy(cam_right),  # 添加批次维度 (1, 84, 84, 3)
                        'cam_left': torch.from_numpy(cam_left),
                        'cam_arm': torch.from_numpy(cam_arm),
                        'state': torch.from_numpy(state)
                    }
                    print("state in the observation",state)
                    print("dataset:\n", demo_data['action'][frame_idx:frame_idx+8])
                    
                else:
                    print(f"警告: {demo_key} 不存在于HDF5文件中")
                    # 回退到原始observation
                    
        except Exception as e:
            print(f"从HDF5读取数据时出错: {e}")
            import traceback
            traceback.print_exc()
    observation = copy(observation)
    # hardcode
   
    use_pcd = True
    if use_pcd:
        for name in observation:
            if "depths.cam_front" in name:
                cam_name = name.split(".")[-1]
                observation[name.replace("depths","pointclouds")] = torch.Tensor(get_point_cloud(observation[name.replace("depths","images")].numpy(),observation[name].numpy(),robot.cameras[cam_name].intrinsics,robot.inv_extrinsics[cam_name],robot.cameras[cam_name].depth_scale,robot.config.crop_bounds))
                break
    
        observation = {k:observation[k] for k in observation if 'images' not in k and 'depths' not in k}
            
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        # for act_tensor in previous_actions:

        for name in observation:
            # observation[name].squeeze(0)
            if name in ['cam_right', 'cam_left', 'cam_arm']:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(0, 3, 1, 2).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)
        # Compute the next action with the policy
        # based on the current observation
        action = policy.predict_action(observation)

    ### for policy with error detector ###
    return torch.from_numpy(action)

        ### for policy yuan ###
    #     pred_action=action['action_pred']
    #     action=action['action']

    #     # Remove batch dimension
    #     action = action.squeeze()
    #     # Move to cpu, if not already the case
    #     action = action.to("cpu")

    #     # print("policy:\n", action)
    # return action, pred_action


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def warmup_record(
    robot,
    events,
    enable_teleoperation,
    warmup_time_s,
    display_cameras,
    fps,
):
    control_loop(
        robot=robot,
        control_time_s=warmup_time_s,
        display_cameras=display_cameras,
        events=events,
        fps=fps,
        teleoperate=enable_teleoperation,
        # eval=True
    )
    # import pdb;pdb.set_trace()
    # print(robot.follower_arms.keys())
    # robot.follower_arms['main'].robot.set_servo_angle(angle=[3.1,-11.1,-48.7,-176.4,-78.7,188.3])
    # time.sleep(1)
def record_episode(
    robot,
    dataset,
    events,
    episode_time_s,
    display_cameras,
    policy,
    fps,
    single_task,
    eval,
):
    control_loop(
        robot=robot,
        control_time_s=episode_time_s,
        display_cameras=display_cameras,
        dataset=dataset,
        events=events,
        policy=policy,
        fps=fps,
        teleoperate=policy is None,
        single_task=single_task,
        eval=eval
    )
# import matplotlib.pyplot as plt

def filter_obs(observation, data_flag):
    included_keys = []
    if data_flag == "rgb_front":
        included_keys=["observation.images.cam_front"]
    if data_flag == "rgb":
        included_keys=["observation.images.cam_front","observation.images.cam_arm","observation.images.cam_back"]
    if data_flag is None:
        included_keys = [k for k in observation]
    return {k:observation[k] for k in observation if 'cam' not in k or k in included_keys}
    
@safe_stop_image_writer
def control_loop(
    robot,
    control_time_s=None,
    teleoperate=False,
    display_cameras=False,
    dataset: LeRobotDataset | None = None,
    events=None,
    policy: PreTrainedPolicy = None,
    fps: int | None = None,
    single_task: str | None = None,
    eval: bool = False
):
    # hardcode
    data_flag = None
    action_list=[]
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if teleoperate and policy is not None:
        raise ValueError("When `teleoperate` is True, `policy` should be None.")

    if dataset is not None and single_task is None:
        raise ValueError("You need to provide a task as argument in `single_task`.")

    if dataset is not None and fps is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    timestamp = 0
    start_episode_t = time.perf_counter()
    step_counter = 0
    
    # cv2.namedWindow('Image',cv2.WINDOW_AUTOSIZE)
    # with open('action', 'rb') as file:
    #     previous_actions = pickle.load(file)
    i=0
    while timestamp < control_time_s:
        i=i+1
        start_loop_t = time.perf_counter()
        if teleoperate:
            observation, action = robot.teleop_step(record_data=True)
        else:
            observation = robot.capture_observation()
            observation = filter_obs(observation,data_flag)
            expected_keys = ['cam_right', 'cam_left', 'cam_arm']
            new_obs = {}
            for key in expected_keys:
                full_key = f'observation.images.{key}'
                if full_key in observation:
                    img_tensor = observation[full_key]  # (B, H, W, C)
                    resized_imgs = []
                    for i in range(img_tensor.shape[0]):
                        img_np = img_tensor[i].numpy()
                        resized_np = cv2.resize(img_np, (84, 84))
                        resized_tensor = torch.from_numpy(resized_np)
                        resized_imgs.append(resized_tensor)
                    new_obs[key] = torch.stack(resized_imgs)
            new_obs['state']=observation['observation.state']
            if policy is not None:
                ### for policy with error detector ###
                pred_action = predict_action(
                    new_obs, policy, get_safe_torch_device("cuda:0"), use_amp=False, robot=robot)
                pred_action = pred_action.squeeze(0)
                for act in pred_action:
                    action = robot.send_action(act)
                    action = {"action": action}

                ### for policy yuan ###
                # pred_action, action_pred = predict_action(
                #     new_obs, policy, get_safe_torch_device("cuda:0"), use_amp=False, robot=robot, hdf5_path=None, demo_idx=20, current_step=step_counter)
                # action_list.append(action_pred)
                # for act in pred_action:
                #     action = robot.send_action(act)
                #     action = {"action": action}
        if dataset is not None and not eval:
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)

        # if display_cameras and not is_headless():
        #     image_keys = [key for key in observation if "image" in key]
        #     for key in image_keys:
        #         # path = f"./test_images/{timestamp}_{key}.jpg"
        #         # import os
        #         # os.makedirs(os.path.dirname(path),exist_yuan/ok=True)
        #         # print(path)
                
        #         # res = cv2.imwrite(path,cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
                
        #         # plt.figure(key)
        #         # plt.imshow(observation[key].numpy())
        #         # plt.show()
        #         # cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
        #         pass
        #     cv2.waitKey(1)

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            print(dt_s)
            busy_wait(1 / fps - dt_s)
        
        dt_s = time.perf_counter() - start_loop_t
        
        log_info = False
        if log_info:
            log_control_info(robot, dt_s, fps=fps)

        step_counter += 1
        timestamp = time.perf_counter() - start_episode_t
        if events["exit_early"]:
            events["exit_early"] = False
            break
    import numpy as np
    if hasattr(policy, 'errors_'):
        # Save the errors from the policy
        if isinstance(policy.errors_, list):
            errors = np.array(policy.errors_)
        elif isinstance(policy.errors_, np.ndarray):
            errors = policy.errors_
        np.save('error_array',errors)
    # with open('action','wb') as file:
    #     pickle.dump(action_list,file)


def record_episode_dp3(
    robot,
    dataset,
    events,
    episode_time_s,
    display_cameras,
    policy,
    fps,
    single_task,
    eval,
):
    control_loop_dp3(
        robot=robot,
        control_time_s=episode_time_s,
        display_cameras=display_cameras,
        dataset=dataset,
        events=events,
        policy=policy,
        fps=fps,
        teleoperate=policy is None,
        single_task=single_task,
        eval=eval
    )
# import matplotlib.pyplot as plt

def filter_obs(observation, data_flag):
    included_keys = []
    if data_flag == "rgb_front":
        included_keys=["observation.images.cam_front"]
    if data_flag == "rgb":
        included_keys=["observation.images.cam_front","observation.images.cam_arm","observation.images.cam_back"]
    if data_flag is None:
        included_keys = [k for k in observation]
    return {k:observation[k] for k in observation if 'cam' not in k or k in included_keys}
    
@safe_stop_image_writer
def control_loop_dp3(
    robot,
    control_time_s=None,
    teleoperate=False,
    display_cameras=False,
    dataset: LeRobotDataset | None = None,
    events=None,
    policy: PreTrainedPolicy = None,
    fps: int | None = None,
    single_task: str | None = None,
    eval: bool = False
):
    # hardcode
    data_flag = None
    action_list=[]
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if teleoperate and policy is not None:
        raise ValueError("When `teleoperate` is True, `policy` should be None.")

    if dataset is not None and single_task is None:
        raise ValueError("You need to provide a task as argument in `single_task`.")

    if dataset is not None and fps is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    timestamp = 0
    start_episode_t = time.perf_counter()
    step_counter = 0
    
    # cv2.namedWindow('Image',cv2.WINDOW_AUTOSIZE)
    # with open('action', 'rb') as file:
    #     previous_actions = pickle.load(file)
    i=0
    while timestamp < control_time_s:
        i=i+1
        start_loop_t = time.perf_counter()
        if teleoperate:
            observation, action = robot.teleop_step(record_data=True)
        else:
            observation = robot.capture_observation()

            observation = filter_obs(observation,data_flag)
            expected_keys = ['cam_right', 'cam_left', 'cam_arm']
            new_obs = {}
            for key in expected_keys:
                full_key = f'observation.images.{key}'
                if full_key in observation:
                    img_tensor = observation[full_key]  # (B, H, W, C)
                    resized_imgs = []
                    for i in range(img_tensor.shape[0]):
                        img_np = img_tensor[i].numpy()
                        resized_np = cv2.resize(img_np, (84, 84))
                        resized_tensor = torch.from_numpy(resized_np)
                        resized_imgs.append(resized_tensor)
                    new_obs[key] = torch.stack(resized_imgs)
            new_obs['state']=observation['observation.state']
            from lerobot.realtime_pointcloud import RealtimePointCloudGenerator
            generator = create_realtime_generator(
                calib_path=args.calib,
                inst_path=args.inst,
                device='cuda:0'
            )
            pointcloud = generator.generate(rgb_dict, depth_dict, ee_pos, num_points=8192)
            if policy is not None:
                ### for policy with error detector ###
                # pred_action = predict_action(
                #     new_obs, policy, get_safe_torch_device("cuda:0"), use_amp=False, robot=robot)
                # time_step = min(i-1, pred_action.shape[1]-1) 
                # current_action = pred_action[0, time_step, :]
                # if hasattr(current_action, 'cpu'):
                #     current_action = current_action.cpu()
                # action_list.append(current_action)
                # actual_action = robot.send_action(current_action)
                # action = {"action": actual_action}

                ### for policy yuan ###
                pred_action, action_pred = predict_action(
                    new_obs, policy, get_safe_torch_device("cuda:0"), use_amp=False, robot=robot, hdf5_path=None, demo_idx=20, current_step=step_counter)
                action_list.append(action_pred)
                for act in pred_action:
                    action = robot.send_action(act)
                    action = {"action": action}
                    
        if dataset is not None and not eval:
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)

        # if display_cameras and not is_headless():
        #     image_keys = [key for key in observation if "image" in key]
        #     for key in image_keys:
        #         # path = f"./test_images/{timestamp}_{key}.jpg"
        #         # import os
        #         # os.makedirs(os.path.dirname(path),exist_yuan/ok=True)
        #         # print(path)
                
        #         # res = cv2.imwrite(path,cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
                
        #         # plt.figure(key)
        #         # plt.imshow(observation[key].numpy())
        #         # plt.show()
        #         # cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
        #         pass
        #     cv2.waitKey(1)

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            print(dt_s)
            busy_wait(1 / fps - dt_s)
        
        dt_s = time.perf_counter() - start_loop_t
        
        log_info = False
        if log_info:
            log_control_info(robot, dt_s, fps=fps)

        step_counter += 1
        timestamp = time.perf_counter() - start_episode_t
        if events["exit_early"]:
            events["exit_early"] = False
            break
    # with open('action','wb') as file:
    #     pickle.dump(action_list,file)


def reset_environment(robot, events, reset_time_s, fps):
    # TODO(rcadene): refactor warmup_record and reset_environment
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    control_loop(
        robot=robot,
        control_time_s=reset_time_s,
        events=events,
        fps=fps,
        teleoperate=True,
    )


def stop_recording(robot, listener, display_cameras):
    robot.disconnect()

    if not is_headless():
        if listener is not None:
            listener.stop()

        if display_cameras:
            cv2.destroyAllWindows()


def sanity_check_dataset_name(repo_id, policy_cfg):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided ({policy_cfg.type})."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, use_videos: bool
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, get_features_from_robot(robot, use_videos)),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
