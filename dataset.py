import albumentations.augmentations.crops.transforms as crops
import cv2
import numpy as np
import pandas as pd
from PIL import Image as PImage
from torch.utils.data import Dataset
import torchvision
from transforms import rectangle_cutmix

config = pd.read_json('train_config.json', typ='series')
MIXUP_ALPHA = config['mixup_alpha']
TARGET_SIZE = int(config['target_size'])

class TrainDataset(Dataset):
    def __init__(
            self,
            image_map, # (image ID, label) tuples
            input_dir,
            transforms=None, # Albumentations transforms
            do_mixup=False,
            depth_map_dir=None,
            depth_biased_rrc=False, # Use depth maps to select random crops toward foreground
            depth_biased_rrc_presize_transforms=None,
            depth_biased_rrc_postsize_transforms=None
            ):
        super(TrainDataset, self).__init__()
        self.image_map = image_map
        self.input_dir = input_dir
        self.transforms = transforms
        self.do_mixup = do_mixup
        self.depth_map_dir = depth_map_dir
        self.depth_biased_rrc = depth_biased_rrc
        self.depth_biased_rrc_presize_transforms = depth_biased_rrc_presize_transforms
        self.depth_biased_rrc_postsize_transforms = depth_biased_rrc_postsize_transforms
        
    def __len__(self):
        return len(self.image_map)
    
    def __getitem__(self, idx):
        """Sample image key, processed image features and one-hot-encoded label from dataset.
        The mixup implementation is implemented according to this paper: https://arxiv.org/pdf/1710.09412v2.pdf
        
        """
        key, label = self.image_map[idx]
        mixup_lam, mixup_label = None, None
        pimage = PImage.open(f'{self.input_dir}/{key}').convert('RGB')
        image = np.array(pimage)
        
        if self.depth_biased_rrc:
            # Augment the image several times and keep the one with the highest mean depth map (for foreground)
            depth_map = PImage.open(f'{self.depth_map_dir}/{key}').convert('L')
            depth_map = np.array(depth_map)
            top_depth_map_aug_mean = None
            top_depth_map_aug_image = None
            
            for _ in range(3):
                aug = self.depth_biased_rrc_presize_transforms(image=image, mask=depth_map)
                image_aug = aug['image']
                depth_map_aug = aug['mask']
                depth_map_aug_mean = np.mean(depth_map_aug)
                
                if top_depth_map_aug_mean is None or depth_map_aug_mean > top_depth_map_aug_mean:
                    top_depth_map_aug_mean = depth_map_aug_mean
                    top_depth_map_aug_image = image_aug
            
            image = self.depth_biased_rrc_postsize_transforms(image=top_depth_map_aug_image)['image']
        else:
            image = self.transforms(image=image)['image']
        
        if self.do_mixup:
            mix_img_key, mixup_label = self.image_map[np.random.randint(len(self.image_map))]
            mixup_pimage = PImage.open(f'{self.input_dir}/{mix_img_key}').convert('RGB')
            mixup_image = np.array(mixup_pimage)
            mixup_image = self.transforms(image=mixup_image)['image']
            mixup_lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)

            image = ((mixup_lam * image) + ((1 - mixup_lam) * mixup_image))
     
        return key, image, label, mixup_lam, mixup_label

    def image_at(self, idx):
        """Return viewable, PIL-image version of instance at idx.
        """
        _, tensor_image, _, _, _ = self.__getitem__(idx)
        tensor_image = torchvision.transforms.Normalize([-0.485/0.229, -0.546/0.224, -0.406/0.225], 
                             [1/0.229, 1/0.224, 1/0.225])(tensor_image)
        array_image = tensor_image.permute(1, 2, 0).cpu().numpy()
        array_image = np.clip(array_image * 255, 0, 255).astype(np.uint8)
        pimage = PImage.fromarray(array_image)
        return pimage


class TrainDatasetV2(Dataset):
    def __init__(
            self,
            image_map, # (image ID, label) tuples
            input_dir,
            presize_transforms=None,
            combine_transforms=None,
            image_source_map=None,
            do_mixup=False,
            do_cutmix=False,
            p_cutmix=0.0,
            do_depthmix=False,
            p_depthmix=False,
            depth_map_dir=None,
            ):
        super(TrainDatasetV2, self).__init__()
        self.image_map = image_map
        self.input_dir = input_dir
        self.presize_transforms = presize_transforms
        self.combine_transforms = combine_transforms
        self.image_source_map = image_source_map
        self.do_mixup = do_mixup
        self.do_cutmix = do_cutmix
        self.p_cutmix = p_cutmix
        self.do_depthmix = do_depthmix
        self.p_depthmix = p_depthmix
        self.depth_map_dir = depth_map_dir
        
    def __len__(self):
        return len(self.image_map)
    
    def __getitem__(self, idx):
        """Sample image key, processed image features and one-hot-encoded label from dataset.
        The mixup implementation is implemented according to this paper: https://arxiv.org/pdf/1710.09412v2.pdf
        
        """
        key, label = self.image_map[idx]
        mixup_lam, mixup_label = None, None
        pimage = PImage.open(f'{self.input_dir}/{key}').convert('RGB')
    
        image = np.array(pimage)
        depth_map = PImage.open(f'{self.depth_map_dir}/{key}').convert('L')
        depth_map = np.array(depth_map)

            
        presized = self.presize_transforms(image=image, mask=depth_map)
        image = presized['image']
        depth_map = presized['mask']
            
        if self.do_depthmix and np.random.uniform() < self.p_depthmix:
            depthmix_img_key, depthmix_label = self.image_map[np.random.randint(len(self.image_map))]
            depthmix_pimage = PImage.open(f'{self.input_dir}/{depthmix_img_key}').convert('RGB')
            depthmix_image = np.array(depthmix_pimage)
            depthmix_image = self.presize_transforms(image=depthmix_image)['image']
            
            depth_map = depth_map.astype(np.float32) / 255.
            depth_map = np.expand_dims(depth_map, axis=-1)
            depth_map = np.log(1 + 1e-12 + depth_map) 
            depth_map = np.interp(depth_map, (depth_map.min(), depth_map.max()), (.3, 1.0))
  
            image = (image * depth_map + depthmix_image * (1 - depth_map))
            
        if self.do_cutmix and np.random.uniform() < self.p_cutmix:
            cutmix_img_key, cutmix_label = self.image_map[np.random.randint(len(self.image_map))]
            cutmix_pimage = PImage.open(f'{self.input_dir}/{cutmix_img_key}').convert('RGB')
            cutmix_image = np.array(cutmix_pimage)
            cutmix_image = self.presize_transforms(image=cutmix_image)['image']
            image = rectangle_cutmix(image, cutmix_image, 16, 128)
            
            
        if self.do_mixup:
            mix_img_key, mixup_label = self.image_map[np.random.randint(len(self.image_map))]
            mixup_pimage = PImage.open(f'{self.input_dir}/{mix_img_key}').convert('RGB')
            mixup_image = np.array(mixup_pimage)
            mixup_image = self.presize_transforms(image=mixup_image)['image']
            mixup_lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)

            image = ((mixup_lam * image) + ((1 - mixup_lam) * mixup_image))
        image = image.astype(np.uint8)
        image = self.combine_transforms(image=image)['image']
     
        return key, image, label, mixup_lam, mixup_label

    def image_at(self, idx):
        """Return viewable, PIL-image version of instance at idx.
        """
        _, tensor_image, _, _, _ = self.__getitem__(idx)
        tensor_image = torchvision.transforms.Normalize([-0.485/0.229, -0.546/0.224, -0.406/0.225], 
                             [1/0.229, 1/0.224, 1/0.225])(tensor_image)
        array_image = tensor_image.permute(1, 2, 0).cpu().numpy()
        array_image = np.clip(array_image * 255, 0, 255).astype(np.uint8)
        pimage = PImage.fromarray(array_image)
        return pimage

    
class ValidDataset(Dataset):
    def __init__(
            self,
            image_map, # (image ID, label) tuples
            input_dir,
            transforms=None, # Albumentations transforms
            image_source_map=None
            ):
        super(ValidDataset, self).__init__()
        self.image_map = image_map
        self.input_dir = input_dir
        self.transforms = transforms
        self.image_source_map = image_source_map
        
    def __len__(self):
        return len(self.image_map)
    
    def __getitem__(self, idx):
        """Sample image key, processed image features and one-hot-encoded label from dataset.
        The mixup implementation is implemented according to this paper: https://arxiv.org/pdf/1710.09412v2.pdf
        
        """
        key, label = self.image_map[idx]
        pimage = PImage.open(f'{self.input_dir}/{key}').convert('RGB')
        image = np.array(pimage)

        image = self.transforms(image=image)['image']
     
        return key, image, label

    def image_at(self, idx):
        """Return viewable, PIL-image version of instance at idx.
        """
        key, tensor_image, label = self.__getitem__(idx)
        tensor_image = torchvision.transforms.Normalize([-0.485/0.229, -0.546/0.224, -0.406/0.225], 
                             [1/0.229, 1/0.224, 1/0.225])(tensor_image)
        array_image = tensor_image.permute(1, 2, 0).cpu().numpy()
        array_image = np.clip(array_image * 255, 0, 255).astype(np.uint8)
        pimage = PImage.fromarray(array_image)
        return pimage
