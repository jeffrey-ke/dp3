import torch
from torch import nn
import vggt
import PointNetEncoderXYZ

"""
Now, go line by line and mark todos
"""
class SonicEncoder(nn.Module):

    class MHAttnBottleneck(nn.Module):
        init: num_heads #TODO
            super().__init__()#TODO
            self.mhattn = nn.MultiheadAttention(embed_dim = vggt_dim, num_heads=num_heada)
            self.proj = nn.Linear(vggt_dim, dp3_extractor_dim)

        forward: self, X where X is B,S,vggt_dim#TODO
            tokens = self.mhattn(X, X, X)
            projected_tokens = self.proj(tokens)
            return projected_tokens
    init:
        super() stuff#TODO
        self.vggt = VGGT()#TODO
        self.vggt_feature_mode = args.vggt_feature_mode
        pointcloud_encoder_cfg = get_cfgs()#TODO
        if args.vggt_feature_mode:
            if args.bottleneck == "linear":
                self.bottleneck = nn.Linear(vggt_dim, dp3_extractor_dim) 
            elif args.bottleneck == "mlp":
                self.bottleneck = nn.Sequential([nn.Linear(),
                                                 nn.ReLU(),
                                                 nn.Linear()]
                                                )
            elif args.bottleneck == "attn":
                self.bottleneck = MHAttnBottleneck(num_heads=4)

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


