import torch
# from jutils.utils import pdb
from torch import nn
import os 
import sys 

vggt_path = os.path.abspath('/home/san/dp3/vggt')
print(f'Loading VGGT path from path {vggt_path}')
if not os.path.exists(vggt_path):
    raise Exception("dp3 path does not exist: " + vggt_path)
sys.path.append(vggt_path)

from vggt.models.vggt import VGGT

import sys
from diffusion_policy_3d.model.vision.pointnet_extractor import PointNetEncoderXYZ, create_mlp

class SonicEncoder(nn.Module):
    vggt_feature_size = None

    class _ConvBottleneck(nn.Module):
        def __init__(self, vggt_feature_size: torch.Size, dp3_encoder_dim):
            super().__init__()
            B, S, n_patches, patch_dim = vggt_feature_size
            self.convs = nn.Sequential(nn.Conv2d(S, S, kernel_size=(4, 4), stride=(2,2), padding=1),
                                        nn.BatchNorm2d(S),
                                        nn.ReLU(),
                                        nn.Conv2d(S, S, kernel_size=(4, 4), stride=(2,2), padding=1),
                                        nn.BatchNorm2d(S),
                                        nn.ReLU(),
                                        nn.Conv2d(S, S, kernel_size=(4, 4), stride=(2,2), padding=1),
                                        nn.BatchNorm2d(S),
                                        nn.ReLU(),
                                        )
            final_reshaped_dim = (S * 1) * (n_patches // 8) * (patch_dim // 8)
            self.proj = nn.Linear(final_reshaped_dim, dp3_encoder_dim)
        def forward(self, features):
            B, *_ = features.shape
            features = self.convs(features)
            reshaped = features.view(B, -1)
            projected_features = self.proj(reshaped)
            return projected_features

    def __init__(self,
                 dp3_encoder_dim,
                 observation_space,
                 state_mlp_size=(64,64),
                 ):

        def construct_state_mlp():
            output_dim = state_mlp_size[-1]
            net_arch = state_mlp_size[:-1]
            #TODO might be an issue with the visibility and scope of self
            state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], 
                                                       output_dim, 
                                                       net_arch,
                                                       nn.ReLU)
                                           )
            return state_mlp
        def get_vggt_feature_size():
            if not SonicEncoder.vggt_feature_size:
                rand_input = torch.randn(self.image_shape).to("cuda")
                features, _ = self.vggt.aggregator(rand_input)
                SonicEncoder.vggt_feature_size = features[0].shape
            return SonicEncoder.vggt_feature_size
        

        def load_vggt():
            v = VGGT()
            url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            v.load_state_dict(torch.hub.load_state_dict_from_url(url))
            v.to("cuda")
            return v

        super().__init__()
        self.n_output_channels = dp3_encoder_dim
        self.image_shape = observation_space['image']
        self.state_shape = observation_space['agent_pos']
        self.vggt = load_vggt()
        self.state_mlp = construct_state_mlp()
        vggt_feature_size = get_vggt_feature_size()
        self.bottleneck = SonicEncoder._ConvBottleneck(vggt_feature_size, dp3_encoder_dim) 
        self.vggt_batchsize = 32 # TODO: needs to go into config!
        self.vggt_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    def forward(self, observations):
        robot_state = observations["agent_pos"] # B, 24
        robot_state_features = self.state_mlp(robot_state)
        images = observations["img"].permute(0, 3, 1, 2) # now, in shape B,C,H,W
        images = images.unsqueeze(1) # VGGT expects B, N_views, C, H, W
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.vggt_dtype):
                features = []
                for i in range(0, images.shape[0], self.vggt_batchsize):
                    minibatch = images[i:i+self.vggt_batchsize]
                    # NOTE: vggt returns features from all 24 attention layers, only using last layers features here!
                    features.append( self.vggt.aggregator(minibatch)[0][-1])
        
        # import pdb; pdb.set_trace()
        features = torch.cat(features, dim=0) 

        bottlenecked_features = self.bottleneck(features)
        print(f'cated features.shape : {bottlenecked_features.shape}')
        cated_features = torch.cat([bottlenecked_features, robot_state_features], dim=-1)
        print(f'cated features.shape : {cated_features.shape}')
        return cated_features

    def output_shape(self):
        return self.n_output_channels
