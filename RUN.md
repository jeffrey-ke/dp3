
## To run adroit task:

> Task yaml changes:

default change to every task file:

image_shape: &image_shape [3, 168, 168]
shape_meta: &shape_meta
  # acceptable types: rgb, low_dim
  obs:
    image:
      shape: *image_shape
      type: rgb



If you're using a feature encoder and the expert zarr has features column:
zarr_keys: ${policy.expert_zarr_keys}
feature_layer: ${policy.encoder_feature_layer}


In <sonic.yaml>:
- encoder_feature_layer: 6

