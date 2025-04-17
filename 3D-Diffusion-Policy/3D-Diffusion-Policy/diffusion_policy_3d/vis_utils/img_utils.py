import os
from PIL import Image
import numpy as np

def save_test_images(observations, save_dir="test_images"):
    # Create test directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get images from observations
    images = observations["img"]  # Assuming images are in "img" key
    if len(images.shape) == 5:
        images = images[0]
    # Save a few sample images
    for i in range(min(3, len(images))):  # Save up to 3 images
        # Convert tensor to numpy array and transpose to HWC format
        print(images[i].shape, images[i].min(), images[i].max())
        img_np = images[i].cpu().numpy()
        
        # Convert to uint8 if needed
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
        
        # Create PIL Image and save
        img = Image.fromarray(img_np)
        img.save(os.path.join(save_dir, f"test_image_{i}.png"))
        print(f"Saved test image {i} to {save_dir}/test_image_{i}.png")
