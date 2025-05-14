import torch
from torch import nn
from diffusion_policy_3d.model.vision.pointnet_extractor import create_mlp
from termcolor import cprint

class DinoV2Encoder(nn.Module):
    def __init__(self, observation_space, out_channel, dino_v2_variant='dinov2_vitl14', **encoder_cfg):
        super().__init__()

        cprint(f'Using dino v2 variant: {dino_v2_variant}', 'yellow')

        ## state information
        state_mlp_output_dim = 64                          
        self.final_dim = out_channel + state_mlp_output_dim
        self.n_output_channels = self.final_dim
        self.robot_state_dim = observation_space['agent_pos']
        self.image_shape = observation_space['image']
        ## layers
        self.feature_layer = encoder_cfg.get("feature_layer", 23)
        self.dino = torch.hub.load('facebookresearch/dinov2', dino_v2_variant)
        self.dino_proj = nn.Linear(self.dino.embed_dim, out_channel)
        self.state_mlp = nn.Sequential(*create_mlp(self.robot_state_dim[0],
                                                   state_mlp_output_dim,
                                                   [state_mlp_output_dim],
                                                   nn.ReLU))
    def forward(self, obs):
        # images ought to be of shape B,3,H,W
        robot_state = obs["agent_pos"]
        robot_state_features = self.state_mlp(robot_state) #B,64

        if obs["img"].shape[1] != 3:
            images = obs["img"].permute(0, 3, 1, 2) 
        else:
            images = obs["img"]

        with torch.no_grad():
            self.dino.eval()
            # dino_features = self.dino(images)
            layer_feats = self.dino.get_intermediate_layers(images, [self.feature_layer], return_class_token=True)
            layer_cls = layer_feats[0][1]

        proj_dino_features = self.dino_proj(layer_cls) #B,out_channel
        cated_features = torch.cat([proj_dino_features, robot_state_features], dim=-1)
        return cated_features

    def output_shape(self):
        return self.final_dim
