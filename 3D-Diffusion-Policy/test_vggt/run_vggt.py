import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model = VGGT().to(device)
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
# model = VGGT().to(device)
# checkpoint_path = "./checkpoints/model.pt"
# checkpoint_path = "/home/rajath/workspace/capstone/dp3/3D-Diffusion-Policy/test_vggt/checkpoints/model.pt"
# model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# Load and preprocess example images (replace with your own image paths)
image_names = [
    "test_vggt/sample_images/0001_0000.png",
    "test_vggt/sample_images/0002_0000.png",
    "test_vggt/sample_images/0003_0000.png",
    "test_vggt/sample_images/0004_0000.png",
    "test_vggt/sample_images/0005_0000.png",
]

images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    predictions = model(images)

for key in predictions.keys():
    print(f'{key}: {predictions[key].shape}')

import pdb; pdb.set_trace()

# with torch.no_grad():
#     # with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
#         # Predict attributes including cameras, depth maps, and point maps.
#         predictions = model(images)
