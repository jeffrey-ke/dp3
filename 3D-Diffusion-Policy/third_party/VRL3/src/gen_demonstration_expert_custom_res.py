# usage:
#       bash scripts/vrl3_gen_demonstration_expert.sh door
import mj_envs
from mjrl.utils.gym_env import GymEnv
from rrl_local.rrl_utils import make_basic_env, make_dir
from adroit import AdroitEnv
import matplotlib.pyplot as plt
import argparse
import os
import torch
from vrl3_agent import VRL3Agent
import utils
from termcolor import cprint
from PIL import Image
import zarr
from copy import deepcopy
import numpy as np
import yaml

from diffusion_policy_3d.gym_util.mjpc_wrapper import MujocoPointcloudWrapperAdroit
from diffusion_policy_3d.model.vision.sonic import load_vggt
from torchvision.transforms import ToTensor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='door', help='environment to run')
    parser.add_argument('--policy_name', type=str, default=None, help='policy to run')
    parser.add_argument('--num_episodes', type=int, default=100, help='number of episodes to run')
    parser.add_argument('--root_dir', type=str, default='data', help='directory to save data')
    parser.add_argument('--expert_ckpt_path', type=str, default=None, help='path to expert ckpt')
    parser.add_argument('--img_size', type=int, default=84, help='image size')
    parser.add_argument('--not_use_multi_view', action='store_true', help='not use multi view')
    parser.add_argument('--use_point_crop', action='store_true', help='use point crop')
    parser.add_argument('--feature_layer', type=int, default=None, help='use feature layer')
    args = parser.parse_args()
    return args


def render_camera(sim, camera_name="top", im_size=84):
    img = sim.render(im_size, im_size, camera_name=camera_name)
    return img

def render_high_res(sim, camera_name="top"):
    img = sim.render(1024, 1024, camera_name=camera_name)
    return img


def main():
    args = parse_args()
    # load env
    action_repeat = 2
    frame_stack = 1
    def create_env(cam_list=None):
        env = AdroitEnv(env_name=args.env_name+'-v0', cam_list=cam_list, test_image=False, num_repeats=action_repeat,
                        num_frames=frame_stack, env_feature_type='pixels',
                                            device='cuda', reward_rescale=True)
        env = MujocoPointcloudWrapperAdroit(env=env, env_name='adroit_'+args.env_name, use_point_crop=args.use_point_crop)
        return env
    num_episodes = args.num_episodes
    save_dir = os.path.join(args.root_dir, 'adroit_'+args.env_name+'_expert.zarr')
    if os.path.exists(save_dir):
        cprint('Data already exists at {}'.format(save_dir), 'red')
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        cprint("Do you want to overwrite? (y/n)", "red")
        # user_input = input()
        user_input = 'y'
        if user_input == 'y':
            cprint('Overwriting {}'.format(save_dir), 'red')
            os.system('rm -rf {}'.format(save_dir))
        else:
            cprint('Exiting', 'red')
            return
    os.makedirs(save_dir, exist_ok=True)
    
    # load expert ckpt
    loaded_dict = torch.load(args.expert_ckpt_path, map_location='cpu')
    expert_agent = loaded_dict['agent']
    expert_agent.to('cuda')
    
    cprint('Loaded expert ckpt from {}'.format(args.expert_ckpt_path), 'green')
    if args.policy_name:
        cprint(f'Policy name is: {args.policy_name}', 'green')

    total_count = 0
    img_arrays = []
    img_arrays_84 = []
    point_cloud_arrays = []
    depth_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    

    # loop over episodes
    minimal_episode_length = 100
    episode_idx = 0
    cam_list = None
    if args.policy_name:
        policy_conf_path = f'../../../3D-Diffusion-Policy/diffusion_policy_3d/config/{args.policy_name}.yaml'
        with open(policy_conf_path, "r") as f:
            policy_conf = yaml.safe_load(f)
        cam_list = policy_conf['policy']['cam_list']
    while episode_idx < num_episodes:
        env = create_env(cam_list)
        time_step = env.reset()
        input_obs_visual = time_step.observation # (3n,84,84), unit8
        input_obs_sensor = time_step.observation_sensor # float32, door(24,)q        

        total_reward = 0.
        n_goal_achieved_total = 0.
        step_count = 0
        
        img_arrays_sub = []
        img_arrays_84_sub = []
        point_cloud_arrays_sub = []
        depth_arrays_sub = []
        state_arrays_sub = []
        action_arrays_sub = []
        total_count_sub = 0
        
        while (not time_step.last()) or step_count < minimal_episode_length:
            with torch.no_grad(), utils.eval_mode(expert_agent):
                input_obs_visual = time_step.observation
                input_obs_sensor = time_step.observation_sensor

                # IMPT: Always ensure that the first camera is the camera used by the expert
                if args.not_use_multi_view:
                    input_obs_visual = input_obs_visual[:3] # (3,84,84)

                img_custom_res = []
                for cam_name in cam_list:
                    cur_img = render_camera(env.env._env.sim, camera_name=cam_name, im_size=168).transpose(2,0,1).copy()
                    img_custom_res.append(cur_img)
                img_custom_res =  np.concatenate(img_custom_res, axis=0)

                action = expert_agent.act(obs=input_obs_visual, step=0,
                                        eval_mode=True, 
                                        obs_sensor=input_obs_sensor) # (28,) float32

                # save data
                total_count_sub += 1
                img_arrays_sub.append(img_custom_res)
                img_arrays_84_sub.append(input_obs_visual)
                state_arrays_sub.append(input_obs_sensor)
                action_arrays_sub.append(action)
                point_cloud_arrays_sub.append(time_step.observation_pointcloud)
                depth_arrays_sub.append(time_step.observation_depth)
                
            time_step = env.step(action)
            obs = time_step.observation # np array, (3,84,84)
            obs = obs[:3] if obs.shape[0] > 3 else obs # (3,84,84)
            n_goal_achieved_total += time_step.n_goal_achieved
            total_reward += time_step.reward
            step_count += 1
            
        if n_goal_achieved_total < 10.:
            cprint(f"Episode {episode_idx} has {n_goal_achieved_total} goals achieved and {total_reward} reward. Discarding.", 'red')
        else:
            total_count += total_count_sub
            episode_ends_arrays.append(deepcopy(total_count)) # the index of the last step of the episode    
            img_arrays.extend(deepcopy(img_arrays_sub))
            img_arrays_84.extend(deepcopy(img_arrays_84_sub))
            point_cloud_arrays.extend(deepcopy(point_cloud_arrays_sub))
            depth_arrays.extend(deepcopy(depth_arrays_sub))
            state_arrays.extend(deepcopy(state_arrays_sub))
            action_arrays.extend(deepcopy(action_arrays_sub))
            print('Episode: {}, Reward: {}, Goal Achieved: {}'.format(episode_idx, total_reward, n_goal_achieved_total)) 
            episode_idx += 1

    # tracemalloc.stop()
    ###############################
    # save data
    ###############################
    # create zarr file
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    # save img, state, action arrays into data, and episode ends arrays into meta
    img_arrays = np.stack(img_arrays, axis=0)
    img_arrays_84 = np.stack(img_arrays_84, axis=0)
    img_arrays = np.transpose(img_arrays, (0,2,3,1)) # make channel last
    img_arrays_84 = np.transpose(img_arrays_84, (0,2,3,1)) # make channel last
    state_arrays = np.stack(state_arrays, axis=0)
    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
    depth_arrays = np.stack(depth_arrays, axis=0)
    action_arrays = np.stack(action_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
    img_84_chunk_size = (100, img_arrays_84.shape[1], img_arrays_84.shape[2], img_arrays_84.shape[3])
    state_chunk_size = (100, state_arrays.shape[1])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
    action_chunk_size = (100, action_arrays.shape[1])
    zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('img_84', data=img_arrays_84, chunks=img_84_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
    
    
    # print shape
    cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
    cprint(f'img_84 shape: {img_arrays_84.shape}, range: [{np.min(img_arrays_84)}, {np.max(img_arrays_84)}]', 'green')
    cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
    cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
    cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')

    if args.feature_layer is not None:
        cprint(f'Using feature layer {args.feature_layer}', 'yellow')
        device = 'cuda'
        batch_size = 32
        vggt, vggt_dtype = load_vggt(device=device)

        def test_input(img_batch):
            import cv2
            for i in range(img_batch.shape[1]):
                img = img_batch[0, i]
                img_np = img.detach().cpu().numpy()

                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img_np = np.transpose(img_np, (1, 2, 0))
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                cv2.imwrite(f"image_{i}.png", img_np)

        def test_output(img_batch, vggt):
            import open3d as o3d
            tokens, token_start_idx = vggt.aggregator(img_batch)
            point_map, point_conf = vggt.point_head(tokens, img_batch, token_start_idx)

            # Process only batch_idx = 0
            batch_idx = 0

            # Get point cloud and color data for batch_idx
            points = point_map[batch_idx]     # [3, 168, 168, 3]
            colors = img_batch[batch_idx]     # [3, 3, 168, 168]

            # Reshape
            points = points.view(-1, 3).cpu().numpy()  # [3*168*168, 3]
            colors = colors.to(torch.float32).permute(0, 2, 3, 1).contiguous().view(-1, 3).cpu().numpy()  # [3*168*168, 3]

            # Normalize RGB if necessary
            if colors.max() > 1.0:
                colors = colors / 255.0

            # Filter invalid points
            valid_mask = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1e-5)
            points = points[valid_mask]
            colors = colors[valid_mask]

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Save to PLY
            o3d.io.write_point_cloud("colored_pointcloud.ply", pcd)
            exit()


        features = []
        import time
        t0 = time.time()
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=vggt_dtype):
                num_images = img_arrays.shape[0]
                for i in range(0, num_images, batch_size):
                    img_batch_np = img_arrays[i:i+batch_size]
                    img_batch = torch.from_numpy(img_batch_np).permute(0, 3, 1, 2).float() / 255.0
                    B, C, H, W = img_batch.shape
                    n_imgs = C // 3
                    img_batch = img_batch.view(B, n_imgs, 3, H, W)
                    # test_input(img_batch)
                    img_batch = img_batch.to(dtype=vggt_dtype, device=device)
                    # test_output(img_batch, vggt)
                    # print("Img batch dtype: ", img_batch.dtype, end="\r")
                    tokens, token_start_idx = vggt.aggregator(img_batch, args.feature_layer)
                    batch_pred = tokens[-1][:, :, token_start_idx:, :]
                    features.append(batch_pred.detach().cpu().numpy())
        
        print(f'Time taken to extract features: {time.time() - t0}')
        features = np.concatenate(features, axis=0)
        features_chunk_size = (100, features.shape[1], features.shape[2], features.shape[3])
        zarr_data.create_dataset('features_{}'.format(args.feature_layer), data=features, chunks=features_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        cprint(f'features shape: {features.shape}, range: [{np.min(features)}, {np.max(features)}]', 'green')

    cprint(f'Saved zarr file to {save_dir}', 'green')

    # clean up
    del img_arrays, img_arrays_84, state_arrays, point_cloud_arrays, action_arrays, episode_ends_arrays
    del zarr_root, zarr_data, zarr_meta
    del env, expert_agent
    
    
if __name__ == '__main__':
    main()