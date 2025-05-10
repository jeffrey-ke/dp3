# Running Adroit Tasks

## Task YAML Configuration

### Default Changes
The following changes should be made to every task file:

```yaml
image_shape: &image_shape [3, 168, 168]
shape_meta: &shape_meta
  # acceptable types: rgb, low_dim
  obs:
    image:
      shape: *image_shape
      type: rgb

env_runner:
  render_size: 168
  cam_list: ${policy.cam_list}

dataset:
  # If you're using a feature encoder and the expert zarr has features column, add:
  zarr_keys: ${policy.expert_zarr_keys}
  feature_layer: ${policy.encoder_feature_layer}
  
```


### Sonic Configuration
In `sonic.yaml`, set:
```yaml


policy:
  _target_: diffusion_policy_3d.policy.dp3.DP3
  # cam_list: ['top', 'fixed', 'high_back_right', 'high_back', 'high_back_left', 'high_left']
  cam_list: ['top']

  encoder_cfg:
    encoder_feature_layer: 6
    fusion_type: "patch_pool_max"

  # use the below to use cached features while training
  expert_zarr_keys: ['state', 'action', 'point_cloud', 'img', 'features_6']
  # expert_zarr_keys: ['state', 'action', 'point_cloud', 'img']
```

