import torch
from torch import nn
import sys 
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
from pathlib import Path
from jutils.utils import pdb
from vggt.models.vggt import VGGT

from diffusion_policy_3d.model.vision.pointnet_extractor import PointNetEncoderXYZ, create_mlp, DP3Encoder
from diffusion_policy_3d.vis_utils.img_utils import save_test_images

def load_vggt(device="cuda"):
    v = VGGT()
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    v.load_state_dict(torch.hub.load_state_dict_from_url(url))
    v.eval()
    vggt_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return v, vggt_dtype

def conv_downscaler(n_views, n_patches, patch_dim):
    '''Downscales the number of patches and tokens by 8X'''
    model = nn.Sequential(nn.Conv2d(n_views, n_views, kernel_size=(4, 4), stride=(2,2), padding=1),
                                    nn.BatchNorm2d(n_views),
                                    nn.ReLU(),
                                    nn.Conv2d(n_views, n_views, kernel_size=(4, 4), stride=(2,2), padding=1),
                                    nn.BatchNorm2d(n_views),
                                    nn.ReLU(),
                                    nn.Conv2d(n_views, n_views, kernel_size=(4, 4), stride=(2,2), padding=1),
                                    nn.BatchNorm2d(n_views),
                                    nn.ReLU(),
                                    )
    features_dim = (n_views, n_patches // 8, patch_dim // 8)
    return model, features_dim

def patch_dim_downscaler(n_views, n_patches, patch_dim):
    '''Downscales the patch dimension by 8X'''
    # input: B, n_views, n_patches, patch_dim
    features_dim = (n_views, n_patches, patch_dim//8)
    model = nn.Sequential(nn.Flatten(start_dim=0, end_dim=2),
                          nn.Linear(patch_dim, patch_dim // 4),
                          nn.ReLU(),
                          nn.Linear(patch_dim // 4, patch_dim // 8),
                          nn.ReLU(),
                          nn.Unflatten(dim=0, unflattened_size=(-1, n_views, n_patches)),
                          )
    cprint(f"[SonicEncoder] Using patch dim only downscaler", "red")
    return model, features_dim

class ConvBottleneck(nn.Module):
    def __init__(self, n_patches, dp3_encoder_dim, n_views=1, patch_dim=2048, **bottleneck_args):
        super().__init__()
        fusion_type = bottleneck_args.get("fusion_type", "no_pool")
        S = n_views
        # self.token_downscaler, (_, n_patches_conv, patch_dim_conv) = conv_downscaler(n_views, n_patches, patch_dim)
        # TODO: Needs testing
        self.token_downscaler, (_, n_patches_conv, patch_dim_conv) = patch_dim_downscaler(n_views, n_patches, patch_dim)
        if fusion_type == "no_pool":
            cprint(f"[{self.__class__.__name__}] Using no pooling", "red")
            final_reshaped_dim = (S * 1) * n_patches_conv * patch_dim_conv
        elif fusion_type == "patch_pool":
            cprint(f"[{self.__class__.__name__}] Using patch pooling", "red")
            final_reshaped_dim = S * patch_dim_conv
        elif fusion_type == "view_pool":
            cprint(f"[{self.__class__.__name__}] Using view pooling", "red")
            final_reshaped_dim = n_patches_conv * patch_dim_conv
        else:
            raise ValueError(f"Invalid fusion type: {fusion_type}")
        self.fusion_type = fusion_type
        self.proj = nn.Linear(final_reshaped_dim, dp3_encoder_dim)
        
    
    def forward(self, features):
        B, *_ = features.shape
        features = self.token_downscaler(features)
        
        # NOTE: Need ablations for avg pooling
        if self.fusion_type == "patch_pool": 
            features = torch.mean(features, dim=2).values  # Shape should be (B, S, patch_dim // 8)
        elif self.fusion_type == "view_pool": # NOTE: Not needed atm as only 1 view
            features = torch.max(features, dim=1).values # Shape should be (B, n_patches // 8, patch_dim // 8)
        
        # NOTE: Add abhishek's logic for summing
        features = features.view(B, -1)
        projected_features = self.proj(features)
        return projected_features

class SonicEncoder(nn.Module):
    VGGT_PATCH_SIZE=14
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 **bottleneck_args,
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.state_shape = observation_space[self.state_key]
        self.image_shape = observation_space['image']
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        
        cprint(f"[{self.__class__.__name__}] state shape: {self.state_shape}", "yellow")
        cprint(f"[{self.__class__.__name__}] imagination point shape: {self.imagination_shape}", "yellow")


        # defining vggt extractor + bottle necks
        self.vggt, self.vggt_dtype = load_vggt()
        self.vggt_batchsize = 16 # TODO: needs to go into config!
        
        n_patches = self.image_shape[-1]//self.VGGT_PATCH_SIZE * self.image_shape[-2]//self.VGGT_PATCH_SIZE
        self.bottleneck = ConvBottleneck(n_patches, out_channel, **bottleneck_args)
        print(self.bottleneck)
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]
        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[{self.__class__.__name__}] output dim: {self.n_output_channels}", "red")
        
    def forward(self, observations):
        robot_state = observations["agent_pos"] # B, 24
        robot_state_features = self.state_mlp(robot_state)

        if observations["img"].shape[1] != 3:
            images = observations["img"].permute(0, 3, 1, 2) # now, in shape B,C,H,W
        else:
            #NOTE: policy eval run retuns B, H, W, C as default
            images = observations["img"]
        
        images = images.unsqueeze(1) # VGGT expects B, N_views, C, H, W
        # save_test_images(observations)
        with torch.no_grad():
            self.vggt.to(images.device)
            with torch.amp.autocast('cuda', dtype=self.vggt_dtype):
                features = []
                for i in range(0, images.shape[0], self.vggt_batchsize):
                    minibatch = images[i:i+self.vggt_batchsize]
                    # NOTE: vggt returns features from all 24 attention layers, only using last layers features here!
                    tokens, token_start_idx = self.vggt.aggregator(minibatch) 
                    self.selected_feature = 6
                    features.append(tokens[self.selected_feature][:, :, token_start_idx:, :])
            self.vggt.to('cpu')

        features = torch.cat(features, dim=0) # recreating the batch dimension
        bottlenecked_features = self.bottleneck(features)
        cated_features = torch.cat([bottlenecked_features, robot_state_features], dim=-1)
        return cated_features

    def output_shape(self):
        return self.n_output_channels


if __name__ == "__main__":
    sonic = SonicEncoder(observation_space=None, img_crop_shape=None, out_channel=256, fusion_type="patch_pool")
    print(sonic.output_shape())
