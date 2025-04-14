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
    # class MHAttnBottleneck(nn.Module):
    #     def __init__(self, num_heads, feature_size, dp3_encoder_dim):
    #         super().__init__()
    #         _, S, P, C = feature_size
    #         self.mhattn = nn.MultiheadAttention(embed_dim=C, num_heads=num_heads)#TODO, because this is outdated
    #         self.proj = nn.Linear(C, dp3_encoder_dim)

    #     def forward(self, features):
    #         B, *_ = features.shape
    #         tokens = self.mhattn(features, features, features)
    #         tokens_cated = tokens.view(B, -1)
    #         projected_tokens = self.proj(tokens_cated)
    #         return projected_tokens


    # class SimpleLinearBottleneck(nn.Module):
    #     def __init__(self, feature_size: torch.Size, dp3_encoder_dim):#TODO
    #         _, S, P, C = feature_size
    #         self.linear = nn.Linear(S * P * C, dp3_encoder_dim)#TODO

    #     def forward(self, features):
    #         # shape is B,S,P,2C
    #         features_catd = features.view(B, S * P * C)#TODO
    #         bottlenecked = self.linear(features_catd)
    #         return bottlenecked

    # class MlpBottleneck(nn.Module):
    #     def __init__(self,):
    #         self.seq = nn.Sequential(mlp)#TODO

    #     def forward(self, features):
    #         features_catd = features.view(B, S * P * C)#TODO
    #         bottlenecked = self.seq(features)
    #         return bottlenecked

        def forward(self, features):
            # shape is B,S,P,2C
            # say the dimensions are something like
            # 24, 64, 128
            conv2d, output spatial dim to be S * 2, P/2,C # 48, 32, 64
            bn
            relu
            conv2d, output spatial dim to be S * 4,P/4,C/2 # 96, 16, 32
            bn
            relu
            conv2d, output dims are S * 8, P/8, C/4 #96, 8, 16
            bn
            relu
            reshape to B,-1
            linear transform to dp3_encoder_dim
            



    def __init__(self, pointcloud_encoder_cfg=None, args=None):
        super().__init__()
        self.vggt = VGGT.from_pretrained(args.model).to("cuda") 
        self.vggt_feature_mode = args.vggt_feature_mode
        """
        Think about some rationale behind these bottleneck layers.
        Sampling. Existing extractor is basically sampling. Can
        we sample in the weird feature space in VGGT?
        Why is the aggregator shape the way it is? That will inform
        how we bottleneck.
        """
        # feature shape is B, S, P, 2C
        # we need it to be one example, one feature ie (B,N)
        if args.vggt_feature_mode:#TODO
            if args.bottleneck == "linear":
                self.bottleneck = SimpleLinearBottlenneck()#TODO
            elif args.bottleneck == "mlp":
                self.bottleneck = MlpBottleneck()
            elif args.bottleneck == "attn":
                self.bottleneck = MHAttnBottleneck(num_heads=4, reduced_dim=dp3_extractor_dim)#TODO
            elif args.bottlenect = "conv":
                self.bottleneck = ConvBottleneck() #TODO

            if args.bottleneck_use_norm:
                self.norm = nn.LayerNorm(dp3_extractor_dim)
        else:
            self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)#TODO

    forward: observations#TODO
        robot_state = observations["agent_pos"]
        robot_state_features = self.state_mlp(robot_state)#TODO, actually implement this
        images = observations["image"]
        images = vggt_process(images)#TODO, actually implement this, because what format are the images in?
        features = self.vggt.aggregator(images)#TODO
        if args.vggt_feature_mode:
            bottlenecked_features = self.bottleneck(features)
            if args.bottleneck_use_norm:
                bottlenecked_features = self.norm(bottlenecked_features)
            cated_features = torch.cat([bottlenecked_features, robot_state_features], dim=-1)
            return cated_features
        else: #using point clouds
            pc = self.vggt.point_cloud_head(features)#TODO
            pn_feat = self.extractor(pc)#TODO
            cated_features = torch.cat([pn_feat, robot_state_features], dim=-1)
            return cated_features


