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

    class _ConvBottleneck(nn.Module):
        def __init__(self, vggt_feature_size: torch.Size, dp3_encoder_dim):
            super().__init__()
            B, S, n_patches, patch_dim = vggt_feature_size
            self.convs = nn.Sequential(nn.Conv2d(),#TODO
                                        nn.BatchNorm2d(),
                                        nn.ReLU(),
                                        nn.Conv2d(),
                                        nn.BatchNorm2d(),
                                        nn.ReLU(),
                                        nn.Conv2d(),
                                        nn.BatchNorm2d(),
                                        nn.ReLU(),
                                        )
            final_reshaped_dim = (S * 8) * (n_patches // 8) * (patch_dim // 8)
            self.proj = nn.Linear(final_reshaped_dim, dp3_encoder_dim)
        def forward(self, features):
            # shape is B,S,P,2C
            # say the dimensions are something like
            # 24, 64, 128
            features = self.convs(features)
            reshaped = features.view(B, -1)
            projected_featuers = self.proj(reshaped)

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
            if not vggt_feature_size:
                rand_input = torch.randn(input_image_dimensions)
                features, _ = self.vggt.aggregator(rand_input)
                vggt_feature_size = features.shape
            return vggt_feature_size
        super().__init__()
        self.vggt = VGGT.from_pretrained(args.model).to("cuda") 
        self.vggt_feature_mode = args.vggt_feature_mode
        self.state_shape = observation_space['agent_pos']
        self.state_mlp = construct_state_mlp()
        self.args = args
        """
        Think about some rationale behind these bottleneck layers.
        Sampling. Existing extractor is basically sampling. Can
        we sample in the weird feature space in VGGT?
        Why is the aggregator shape the way it is? That will inform
        how we bottleneck.
        """
        # feature shape is B, S, P, 2C
        # we need it to be one example, one feature ie (B,N)
        vggt_feature_size = get_vggt_feature_size()
        if args.vggt_feature_mode:#TODO
            # if args.bottleneck == "linear":
            #     self.bottleneck = SimpleLinearBottlenneck()#TODO
            # elif args.bottleneck == "mlp":
            #     self.bottleneck = MlpBottleneck()
            # elif args.bottleneck == "attn":
            #     self.bottleneck = MHAttnBottleneck(num_heads=4, reduced_dim=dp3_encoder_dim)#TODO
            if args.bottlenect == "conv":
                self.bottleneck = SonicEncoder._ConvBottleneck(vggt_feature_size, dp3_encoder_dim) #TODO

            if args.bottleneck_use_norm:
                self.norm = nn.LayerNorm()
        else:
            self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)#TODO

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


