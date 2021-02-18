from PIL import Image
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import torch

input_data_dir = Path('input/2019_2020_merged/train_images')
output_data_dir = input_data_dir.parent / 'depth_maps'
os.makedirs(str(output_data_dir), exist_ok=True)

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform

for fpath in tqdm(input_data_dir.iterdir()):
    if not fpath.suffix == '.jpg': pass
    
    out_path = output_data_dir / fpath.name
    if  out_path.is_file():
        print(f'Skipping file {fpath.name}, depth map file already exists')
        continue
    img = Image.open(str(fpath)).convert('RGB')
    model_in = np.array(img)

    input_batch = midas_transforms(model_in)
    with torch.no_grad():
        model_out = midas(input_batch)

        model_out = torch.nn.functional.interpolate(
            model_out.unsqueeze(1),
            size=model_in.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = model_out.cpu().numpy()
    # Rescale to [0, 1]
    depth_map = np.interp(depth_map, (depth_map.min(), depth_map.max()), (0., 255.))
    depth_map = depth_map.astype(np.uint8)
    Image.fromarray(depth_map).save(str(out_path))