# DP3 + VGGT

- Clone the repo:
```
git clone --recursive https://github.com/jeffrey-ke/dp3
```

- Follow the instructions in [INSTALL.md](3D-Diffusion-Policy/INSTALL.md) to first setup DP3.
- After DP3 setup is done, setup vggt with these steps:
```
cd 3D-Diffusion-Policy/third_party/vggt/
pip install -e .
```
- VGGT can be tested by running [run_vggt.py](3D-Diffusion-Policy/test_vggt/run_vggt.py). (Update `checkpoint_path` before running)
