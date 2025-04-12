import torch
from torch import nn
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import os


if __name__ == "main":
    args = get_args()
    main(args)

def get_args():
    return None
    
def main(args=None):
    model = VGGT.from_pretrained("facebook/VGGT-1B").cuda()
    def get_paths():
        names = [f"img{i}.png" for i in range(1,4)]
        cwd = os.getcwd()
        img_dir = os.path.join(cwd, "data", "jeff_desk")
        return [os.path.join(img_dir, name) for name in names]
    paths = get_paths()
    images = load_and_preprocess_images(paths).to("cuda")
    with torch.no_grad():
        predictions = model(images)
