import os
import numpy as np
import cv2
from diffusion_policy_3d.env.adroit.adroit import AdroitEnv
import torch

def test_adroit_cameras():
    # Create output directory if it doesn't exist
    output_dir = "adroit_camera_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define all available cameras for hammer-v0
    # cam_list = ['fixed', 'vil_camera', 'top', 'front_low', 'rear', 'left_side', 'top_high']
    # cam_list = ['fixed', 'vil_camera', 'top', 'front_low', 'rear', 'left_side', 'top_high', 'cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6', 'cam7']
    # cam_list = ['fixed', 'vil_camera', 'top', 'front', 'front_right', 'right', 'back_right', 'back', 'back_left', 'left', 'front_left']
    # cam_list = ['fixed', 'vil_camera', 'top', 'high_front', 'high_front_right', 'high_right', 'high_back_right', 'high_back', 'high_back_left', 'high_left', 'high_front_left']
    # cam_list = ['low_front', 'low_right', 'low_back', 'low_left']
    cam_list = ['top', 'right', 'front', 'top-right', 'top-front', 'left', 'back']

    # Initialize environment with all cameras
    env = AdroitEnv(
        env_name='hammer-v0',
        cam_list=cam_list,
        env_feature_type='pixels',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Reset environment
    obs = env.reset()
    
    # Get images from all cameras
    images = obs['image']
    # images = images.view(len(cam_list), env.height, env.width, 3)
    images = np.transpose(images, (1, 2, 0))

    cam_images = []
    print(images.shape)
    # Save images from each camera
    i = 0
    for cam_name in cam_list:
        img = images[:, :, i:i+3]
        # Convert to numpy if it's a tensor
        if torch.is_tensor(images):
            img = img.cpu().numpy()
          
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f'{cam_name}.png'), img_bgr)
        
        print(f"Saved image for camera: {cam_name}")
        i += 3
    print(f"All images saved to directory: {output_dir}")

if __name__ == "__main__":
    test_adroit_cameras()
