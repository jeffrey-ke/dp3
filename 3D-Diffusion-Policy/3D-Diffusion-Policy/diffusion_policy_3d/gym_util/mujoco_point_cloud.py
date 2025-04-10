# reference implementation: https://github.com/mattcorsaro1/mj_pc
# with personal modifications


import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PIL_Image
from typing import List
import open3d as o3d
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import preprocess_images

"""
Generates numpy rotation matrix from quaternion

@param quat: w-x-y-z quaternion rotation tuple

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""
def quat2Mat(quat):
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix.")
        raise ValueError

    # Note that the following code snippet can be used to generate the 3x3
    #    rotation matrix, we don't use it because this file should not depend
    #    on mujoco.
    '''
    from mujoco_py import functions
    res = np.zeros(9)
    functions.mju_quat2Mat(res, camera_quat)
    res = res.reshape(3,3)
    '''

    # This function is lifted directly from scipy source code
    #https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    rot_mat_arr = [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw), \
        2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw), \
        2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2]
    np_rot_mat = rotMatList2NPRotMat(rot_mat_arr)
    return np_rot_mat

"""
Generates numpy rotation matrix from rotation matrix as list len(9)

@param rot_mat_arr: rotation matrix in list len(9) (row 0, row 1, row 2)

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""
def rotMatList2NPRotMat(rot_mat_arr):
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat

"""
Generates numpy transformation matrix from position list len(3) and 
    numpy rotation matrix

@param pos:     list len(3) containing position
@param rot_mat: 3x3 rotation matrix as numpy array

@return t_mat:  4x4 transformation matrix as numpy array
"""
def posRotMat2Mat(pos, rot_mat):
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat

"""
Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
    matrix and image width and height

@param cam_mat: 3x3 numpy array representing camera intrinsic matrix
@param width:   image width in pixels
@param height:  image height in pixels

@return t_mat:  4x4 transformation matrix as numpy array
"""
def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# 
# and combines them into point clouds
"""
Class that renders depth images in MuJoCo, processes depth images from
    multiple cameras, converts them to point clouds, and processes the point
    clouds
"""
class PointCloudGenerator(object):
    """
    initialization function

    @param sim:       MuJoCo simulation object
    @param min_bound: If not None, list len(3) containing smallest x, y, and z
        values that will not be cropped
    @param max_bound: If not None, list len(3) containing largest x, y, and z
        values that will not be cropped
    """
    def __init__(self, sim, cam_names:List, img_size=84):
        super(PointCloudGenerator, self).__init__()

        self.sim = sim

        # this should be aligned with rgb
        self.img_width = img_size
        self.img_height = img_size

        self.cam_names = cam_names
        
        # List of camera intrinsic matrices
        self.cam_mats = []
        
        for idx in range(len(self.cam_names)):
            # get camera id
            cam_id = self.sim.model.camera_name2id(self.cam_names[idx])
            fovy = math.radians(self.sim.model.cam_fovy[cam_id])
            f = self.img_height / (2 * math.tan(fovy / 2))
            cam_mat = np.array(((f, 0, self.img_width / 2), (0, f, self.img_height / 2), (0, 0, 1)))
            self.cam_mats.append(cam_mat)

    def generateCroppedPointCloud(self, save_img_dir=None, device_id=0):
        o3d_clouds = []
        cam_poses = []
        depths = []
        for cam_i in range(len(self.cam_names)):
            # Render and optionally save image from camera corresponding to cam_i
            color_img, depth = self.captureImage(self.cam_names[cam_i], capture_depth=True, device_id=device_id)
            depths.append(depth)
            # If directory was provided, save color and depth images
            #    (overwriting previous)
            if save_img_dir != None:
                self.saveImg(depth, save_img_dir, "depth_test_" + str(cam_i))
                self.saveImg(color_img, save_img_dir, "color_test_" + str(cam_i))

            # convert camera matrix and depth image to Open3D format, then
            #    generate point cloud
            
            od_cammat = cammat2o3d(self.cam_mats[cam_i], self.img_width, self.img_height)
            od_depth = o3d.geometry.Image(depth)
            
            o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)
            
            # od_color = o3d.geometry.Image(color_img)  # Convert the color image to Open3D format                
            # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(od_color, od_depth)  # Create an RGBD image
            # o3d_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            #     rgbd_image,
            #     od_cammat)

            # Compute world to camera transformation matrix
            cam_body_id = self.sim.model.cam_bodyid[cam_i]
            cam_pos = self.sim.model.body_pos[cam_body_id] # translation vector w.r.t world frame
            c2b_r = rotMatList2NPRotMat(self.sim.model.cam_mat0[cam_i]) # converts [9] to [3,3]
            # In MuJoCo, we assume that a camera is specified in XML as a body
            #    with pose p, and that that body has a camera sub-element
            #    with pos and euler 0.
            #    Therefore, camera frame with body euler 0 must be rotated about
            #    x-axis by 180 degrees to align it with the world frame.
            b2w_r = quat2Mat([0, 1, 0, 0])
            c2w_r = np.matmul(c2b_r, b2w_r)
            c2w = posRotMat2Mat(cam_pos, c2w_r)
            transformed_cloud = o3d_cloud.transform(c2w)
            o3d_clouds.append(transformed_cloud)

        combined_cloud = o3d.geometry.PointCloud()
        for cloud in o3d_clouds:
            combined_cloud += cloud
        # get numpy array of point cloud, (position, color)
        combined_cloud_points = np.asarray(combined_cloud.points)
        # color is automatically normalized to [0,1] by open3d
        

        # combined_cloud_colors = np.asarray(combined_cloud.colors)  # Get the colors, ranging [0,1].
        combined_cloud_colors = color_img.reshape(-1, 3) # range [0, 255]
        combined_cloud = np.concatenate((combined_cloud_points, combined_cloud_colors), axis=1)
        depths = np.array(depths).squeeze()
        import pdb; pdb.set_trace()
        return combined_cloud, depths


     
    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    def depthimg2Meters(self, depth):
        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image

    def verticalFlip(self, img):
        return np.flip(img, axis=0)

    # Render and process an image
    def captureImage(self, camera_name, capture_depth=True, device_id=0):
        rendered_images = self.sim.render(self.img_width, self.img_height, camera_name=camera_name, depth=capture_depth, device_id=device_id)
        if capture_depth:
            img, depth = rendered_images
            depth = self.verticalFlip(depth)

            depth_convert = self.depthimg2Meters(depth)
            img = self.verticalFlip(img)
            return img, depth_convert
        else:
            img = rendered_images
            # Rendered images appear to be flipped about vertical axis
            return self.verticalFlip(img)

    # Normalizes an image so the maximum pixel value is 255,
    # then writes to file
    def saveImg(self, img, filepath, filename):
        normalized_image = img/img.max()*255
        normalized_image = normalized_image.astype(np.uint8)
        im = PIL_Image.fromarray(normalized_image)
        im.save(filepath + '/' + filename + ".jpg")




class VGGTPointCloudGenerator(PointCloudGenerator):
    """
    initialization function

    @param sim:       MuJoCo simulation object
    @param min_bound: If not None, list len(3) containing smallest x, y, and z
        values that will not be cropped
    @param max_bound: If not None, list len(3) containing largest x, y, and z
        values that will not be cropped
    """
    def __init__(self, sim, cam_names:List, img_size=84, device="cpu"):
        super(PointCloudGenerator, self).__init__()

        self.sim = sim
        self.img_width = img_size
        self.img_height = img_size
        self.vggt_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.vggt_device = device
        
        # self.vggt_model = VGGT().to(device)
        # checkpoint_path = "/home/rajath/workspace/capstone/dp3/3D-Diffusion-Policy/test_vggt/checkpoints/model.pt"
        # self.vggt_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        self.vggt_model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        self.vggt_model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        self.vggt_model.to(self.vggt_device)
        self.vggt_model.eval()
        
        self.cam_names = cam_names
        # List of camera intrinsic matrices
        self.cam_mats = []
        
        for idx in range(len(self.cam_names)):
            # get camera id
            cam_id = self.sim.model.camera_name2id(self.cam_names[idx])
            fovy = math.radians(self.sim.model.cam_fovy[cam_id])
            f = self.img_height / (2 * math.tan(fovy / 2))
            cam_mat = np.array(((f, 0, self.img_width / 2), (0, f, self.img_height / 2), (0, 0, 1)))
            self.cam_mats.append(cam_mat)


    def generateCroppedPointCloud(self, save_img_dir=None, device_id=0):
        o3d_clouds = []
        cam_poses = []
        depths = []
        imgs = []
        
        for cam_i in range(len(self.cam_names)):
            color_img = self.captureImage(self.cam_names[cam_i], capture_depth=False, device_id=device_id)
            imgs.append(color_img)
            # depths.append(depth)
            if save_img_dir != None:
                # self.saveImg(depth, save_img_dir, "depth_test_" + str(cam_i))
                self.saveImg(color_img, save_img_dir, "color_test_" + str(cam_i))
            
        images_tensor = preprocess_images(imgs) # (B, C, H, W)
        with torch.no_grad():
            with torch.amp.autocast(self.vggt_device, dtype=self.vggt_dtype):
                images = images_tensor[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = self.vggt_model.aggregator(images)        
                # Predict Point Maps
                point_map, point_conf = self.vggt_model.point_head(aggregated_tokens_list, images, ps_idx)
                depth_map, depth_conf = self.vggt_model.depth_head(aggregated_tokens_list, images, ps_idx)
            
        # Convert point map to numpy array and reshape
        point_colors = images_tensor.squeeze().permute(1, 2, 0).contiguous().view(-1, 3) # (B, H, W, C)
        point_map = point_map.squeeze().view(-1, 3) # (H*W, 3)
        point_conf = point_conf.squeeze().view(-1) # (H*W,)
        depth_map = depth_map.squeeze().view(-1) # (H*W,)
        depth_conf = depth_conf.squeeze().view(-1) # (H*W,)

        # Create masks based on confidence threshold
        # TODO: change to arg-based threshold
        # import pdb; pdb.set_trace()
        mask = (point_conf > 2) & (depth_conf > 5)

        # Reshape and concatenate
        point_map = point_map[mask]
        point_colors = point_colors[mask]
        depth_map = depth_map[mask]
        # Create open3d point cloud
       
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_map.cpu().to(dtype=torch.float32).numpy())
        
        
        # TODO: apply b2w_r = quat2Mat([0, 1, 0, 0]) to point_map, see above class!
        # Transform point cloud from camera to world frame
        # Get camera 0 transform since VGGT predicts w.r.t first camera
        # cam_body_id = self.sim.model.cam_bodyid[0]
        # cam_pos = self.sim.model.body_pos[cam_body_id]
        # c2b_r = rotMatList2NPRotMat(self.sim.model.cam_mat0[0])
        # b2w_r = quat2Mat([0, 1, 0, 0])  # 180 deg rotation about x-axis
        # c2w_r = np.matmul(c2b_r, b2w_r)
        # c2w = posRotMat2Mat(cam_pos, c2w_r)
        # pcd.transform(c2w)

        # get rgb colors
        # print('range of point_colors', point_colors.min(), point_colors.max())
        point_xyzrgb = torch.cat([point_map, point_colors], dim=-1)
        point_xyzrgb = point_xyzrgb.cpu().numpy()
        depth_map = depth_map.cpu().to(dtype=torch.float32).numpy()
        # TODO: check for depth dependency in other classes
        # import pdb; pdb.set_trace()
        return point_xyzrgb, depth_map


