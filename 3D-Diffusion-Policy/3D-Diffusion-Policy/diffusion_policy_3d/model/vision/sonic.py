import torch
from torch import nn
import sys 
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint

sys.path.append('/home/san/dp3/vggt')
from vggt.models.vggt import VGGT

from diffusion_policy_3d.model.vision.pointnet_extractor import PointNetEncoderXYZ, create_mlp, DP3Encoder
from diffusion_policy_3d.vis_utils.img_utils import save_test_images

def load_vggt(device="cuda"):
    v = VGGT()
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    v.load_state_dict(torch.hub.load_state_dict_from_url(url))
    v.to(device)
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


class AttentionBottleneck(nn.Module):
    def __init__(self, n_patches, dp3_encoder_dim, n_views=1, patch_dim=2048, n_heads=8, dropout=0.1, use_hierarchical=True):
        super().__init__()
        self.n_patches = n_patches
        self.n_views = n_views
        self.patch_dim = patch_dim
        self.use_hierarchical = use_hierarchical
        
        # Specialized token for diffusion policy conditioning
        self.policy_token = nn.Parameter(torch.randn(1, 1, patch_dim))
        
        # Positional embeddings for patches + policy token
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, patch_dim))
        
        # View embedding to differentiate between camera views
        self.view_embedding = nn.Parameter(torch.randn(1, n_views, patch_dim))
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=patch_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout
        )
        
        # Cross-view attention (only used if n_views > 1 and hierarchical)
        if n_views > 1 and use_hierarchical:
            self.cross_view_attention = nn.MultiheadAttention(
                embed_dim=patch_dim,
                num_heads=n_heads,
                batch_first=True,
                dropout=dropout
            )
        
        # Layer norm
        self.ln = nn.LayerNorm(patch_dim)
        
        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(patch_dim, dp3_encoder_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, features):
        B, V, P, D = features.shape  # Batch, Views, Patches, Dimension
        assert V == self.n_views, f"Expected {self.n_views} views, got {V}"
        assert P == self.n_patches, f"Expected {self.n_patches} patches, got {P}"
        
        if self.use_hierarchical and self.n_views > 1:
            return self._forward_hierarchical(features)
        else:
            return self._forward_flat(features)
    
    def _forward_flat(self, features):
        """Process all patches from all views together in a flat structure."""
        B, V, P, D = features.shape
        
        # Reshape to combine batch and views
        features = features.reshape(B * V, P, D)  # (B*V), P, D
        
        # Append policy token to each sequence
        policy_tokens = self.policy_token.expand(B * V, -1, -1)
        features = torch.cat((policy_tokens, features), dim=1)  # (B*V), P+1, D
        
        # Add positional embeddings
        features = features + self.pos_embedding
        
        # Apply attention without residual connection
        # This forces the attention to learn completely new relationships
        attn_in = self.ln(features)
        attn_out = self.attention(attn_in, attn_in, attn_in)[0]
        
        # Extract policy token output
        policy_out = attn_out[:, 0]  # (B*V), D
        
        # Reshape back to separate batch and views
        policy_out = policy_out.view(B, V, -1)  # B, V, D
        
        # Add view embeddings
        policy_out = policy_out + self.view_embedding
        
        # Aggregate across views with learned weights
        view_weights = F.softmax(torch.sum(policy_out, dim=2), dim=1).unsqueeze(-1)  # B, V, 1
        policy_agg = torch.sum(policy_out * view_weights, dim=1)  # B, D
        
        # Project to desired output dimension
        out = self.proj(policy_agg)  # B, dp3_encoder_dim
        
        return out
    
    def _forward_hierarchical(self, features):
        """Process each view separately first, then perform cross-view attention."""
        B, V, P, D = features.shape
        
        # First stage: process each view separately
        view_representations = []
        
        for v in range(V):
            view_features = features[:, v]  # B, P, D
            
            # Append policy token
            policy_tokens = self.policy_token.expand(B, -1, -1)
            combined = torch.cat((policy_tokens, view_features), dim=1)  # B, P+1, D
            
            # Add positional embeddings
            combined = combined + self.pos_embedding
            
            # Apply attention without residual connection
            attn_in = self.ln(combined)
            attn_out = self.attention(attn_in, attn_in, attn_in)[0]
            
            # Extract policy token representation
            policy_rep = attn_out[:, 0]  # B, D
            view_representations.append(policy_rep)
        
        # Stack all view representations
        view_stack = torch.stack(view_representations, dim=1)  # B, V, D
        
        # Add view embeddings
        view_stack = view_stack + self.view_embedding
        
        # Second stage: cross-view attention
        # Apply cross-view attention
        cross_attn_in = self.ln(view_stack)
        cross_attn_out = self.cross_view_attention(
            cross_attn_in, cross_attn_in, cross_attn_in
        )[0]  # B, V, D
        
        # Use attention to compute weights for views
        view_importance = cross_attn_out.mean(dim=2)  # B, V
        view_weights = F.softmax(view_importance, dim=1).unsqueeze(-1)  # B, V, 1
        
        # Weighted average of view representations
        agg_representation = torch.sum(cross_attn_out * view_weights, dim=1)  # B, D
        
        # Apply final projection
        out = self.proj(agg_representation)  # B, dp3_encoder_dim
        
        return out


class Bottleneck(nn.Module):
    def __init__(self, n_patches, dp3_encoder_dim, n_views=1, patch_dim=2048, **bottleneck_args):
        super().__init__()
        fusion_type = bottleneck_args.get("fusion_type", "no_pool")
        S = n_views
        # self.token_downscaler, (_, n_patches_conv, patch_dim_conv) = conv_downscaler(n_views, n_patches, patch_dim)
        self.token_downscaler, (_, n_patches_conv, patch_dim_conv) = patch_dim_downscaler(n_views, n_patches, patch_dim)
        if fusion_type == "no_pool":
            cprint(f"[{self.__class__.__name__}] Using no pooling", "red")
            final_reshaped_dim = (S * 1) * n_patches_conv * patch_dim_conv
        elif fusion_type in ["patch_pool_max", "patch_pool_mean"]:
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
        
        if self.fusion_type == "patch_pool_max": 
            features = torch.max(features, dim=2).values  # Shape should be (B, S, patch_dim // 8)
        elif self.fusion_type == "patch_pool_mean":
            features = torch.mean(features, dim=2)
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
        self.bottleneck = Bottleneck(n_patches, out_channel, **bottleneck_args)
        print(self.bottleneck)
        # defining agent state mlp
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
        # import pdb; pdb.set_trace()

        if observations["img"].shape[1] != 3:
            images = observations["img"].permute(0, 3, 1, 2) # now, in shape B,C,H,W
        else:
            #NOTE: policy eval run retuns B, H, W, C as default
            images = observations["img"]
        
        images = images.unsqueeze(1) # VGGT expects B, N_views, C, H, W
        # save_test_images(observations)
        # print(images.shape)
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            self.vggt.to(images.device)
            with torch.amp.autocast('cuda', dtype=self.vggt_dtype):
                features = []
                for i in range(0, images.shape[0], self.vggt_batchsize):
                    minibatch = images[i:i+self.vggt_batchsize]
                    # NOTE: vggt returns features from all 24 attention layers, only using last layers features here!
                    tokens, token_start_idx = self.vggt.aggregator(minibatch) 
                    features.append(tokens[-1][:, :, token_start_idx:, :])
            self.vggt.to('cpu')

        features = torch.cat(features, dim=0) 
        bottlenecked_features = self.bottleneck(features)
        cated_features = torch.cat([bottlenecked_features, robot_state_features], dim=-1)
        return cated_features

    def output_shape(self):
        return self.n_output_channels


if __name__ == "__main__":
    sonic = SonicEncoder(observation_space=None, img_crop_shape=None, out_channel=256, fusion_type="patch_pool")
    print(sonic.output_shape())