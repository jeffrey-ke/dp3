
Current branch changes

- Multi-view images rendered in Adroit
- VGGT features dumped in zarr
- Attention bottleneck added, not checked!
- can run with algo name as sonic now!! yay!



Later
[] Obs['img'] vs obs["image"] - standardize everywhere
[] Removed ema model based gradient flowing : (error due to vggt model no_grad model)

[] Need better way to integrate VGGT
[] Other envs(Metaworld etc) need changes to work with observation dict!

--

- To setup VGGT:
```
cd 3D-Diffusion-Policy/third_party/vggt/
pip install -e .
```
