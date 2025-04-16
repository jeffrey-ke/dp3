import torch
from torch import nn
from vggt.models.vggt import VGGT
from diffusion_policy_3d.model.vision.pointnet_extractor import PointNetEncoderXYZ, create_mlp
from utils import vggt_process

"""
Now, go line by line and mark todos

"""
class SonicEncoder(nn.Module):
    vggt_feature_size = None

    class _AlternateBottlenecks(nn.Module):
        
        def __init__(self, vggt_feature_size: torch.Size, dp3_encoder_dim, args):
            super().__init__()
            self.args = args
            B, S, n_patches, patch_dim = vggt_feature_size

            self.convs = nn.Sequential(nn.Conv2d(S, S, kernel_size=(4, 4), stride=(2,2), padding=1),#TODO
                            nn.BatchNorm2d(S),
                            nn.ReLU(),
                            nn.Conv2d(S, S, kernel_size=(4, 4), stride=(2,2), padding=1),
                            nn.BatchNorm2d(S),
                            nn.ReLU(),
                            nn.Conv2d(S, S, kernel_size=(4, 4), stride=(2,2), padding=1),
                            nn.BatchNorm2d(S),
                            nn.ReLU(),
                            )
            
            # 2 new spatial dimensions
            n_patches_conv = n_patches // 8
            patch_dim_conv = patch_dim // 8

            # Change dimension based on what is pooled over
            if args.pool == "patches":
                final_reshaped_dim = S * patch_dim_conv
            elif args.pool == "views":
                final_reshaped_dim = n_patches_conv * patch_dim_conv
            
            # Projection layer from final dimension to dp3 dimensions
            if args.reshape == "concat":
                self.proj = nn.Linear(final_reshaped_dim, dp3_encoder_dim)
            elif args.reshape == "sum":
                self.proj = nn.Linear(patch_dim_conv, dp3_encoder_dim)          

        def forward(self, features):

            B, *_ = features.shape
            features = self.convs(features) # B, S, num_patches //8, patch_dim // 8

            if self.args.pool == "patches":
                # max over num patches
                pooled_features = torch.max(features, dim=2).values  # Shape should be (B, S, patch_dim // 8)
            elif self.args.pool == "views":
                # max over num views
                pooled_features = torch.max(features, dim=1).values  # Shape should be (B, n_patches // 8, patch_dim // 8)

            # reshape and project
            if self.reshape == "concat":
                # (B, S, patch_dim // 8) -> (B, S * patch_dim // 8) [Max over patches]
                # (B, n_patches // 8, patch_dim //8) -> (B, n_patches // 8 * patch_dim // 8) [Max over views]
                reshaped = pooled_features.view(B, -1)
            elif self.reshape == "sum":
                # (B, S, patch_dim // 8) -> (B, patch_dim // 8) [Max over patches]
                # (B, n_patches //8, patch_dim //8) -> (B, patch_dim // 8) [Max over views]
                reshaped = torch.sum(pooled_features, dim=1)
            projected_features = self.proj(reshaped)

            return projected_features

    def __init__(self,
                 dp3_encoder_dim,
                 observation_space,
                 input_image_dimensions: torch.Size,
                 state_mlp_size=(64,64),
                 pointcloud_encoder_cfg=None,
                 args=None):
        
        def construct_state_mlp():
            output_dim = state_mlp_size[-1]
            net_arch = state_mlp_size[:-1]
            #TODO might be an issue with the visibility and scope of self
            self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], 
                                                       output_dim, 
                                                       net_arch,
                                                       nn.ReLU)
                                           )
        def get_vggt_feature_size():

            if not SonicEncoder.vggt_feature_size:
                rand_input = torch.randn(input_image_dimensions)
                features, _ = self.vggt.aggregator(rand_input)
                SonicEncoder.vggt_feature_size = features.shape
            return vggt_feature_size

        super().__init__()
        self.vggt = VGGT.from_pretrained(args.model).to("cuda") 
        self.vggt_feature_mode = args.vggt_feature_mode
        self.state_shape = observation_space['agent_pos']
        self.state_mlp = construct_state_mlp()
        self.args = args
        vggt_feature_size = get_vggt_feature_size()
        if args.vggt_feature_mode:
            # Using my implementation of alternate bottlenecks
            self.bottleneck = SonicEncoder._AlternateBottlenecks(vggt_feature_size, dp3_encoder_dim, self.args) 
        else:
            self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)

    def forward(self, observations):
        robot_state = observations["agent_pos"]
        robot_state_features = self.state_mlp(robot_state)#TODO, actually implement this
        images = observations["image"]
        images = vggt_process(images)#TODO, actually implement this, because what format are the images in?
        features = self.vggt.aggregator(images)#TODO
        if self.args.vggt_feature_mode:
            bottlenecked_features = self.bottleneck(features)
            if self.args.bottleneck_use_norm:
                bottlenecked_features = self.norm(bottlenecked_features)
            cated_features = torch.cat([bottlenecked_features, robot_state_features], dim=-1)
            return cated_features
        else: #using point clouds
            pc = self.vggt.point_cloud_head(features)#TODO
            pn_feat = self.extractor(pc)#TODO
            cated_features = torch.cat([pn_feat, robot_state_features], dim=-1)
            return cated_features


