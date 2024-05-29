"""coco_data.py
Data pre-processing and loading pipeline for COCO-Stuff-27.
"""

import os
import pickle
import numpy as np
import torch
import torchvision
from torchvision.io import ImageReadMode
import torchvision.transforms as T
from torch.utils.data import Dataset

def get_fine_to_coarse():
  """Map fine label indexing to coarse label indexing."""
  voc_labelmap = np.vectorize(lambda x: -1 if x==255 else x, otypes=[int])
  return voc_labelmap

class ADE20KDataset(Dataset):
  def __init__(self, ROOT, augment_data=False, resolution=1024, split="validation"):
    self.image_list, self.label_list = self.create_path(ROOT, split)
    self.augment_data = augment_data
    self.RESOLUTION = resolution

  def __len__(self):
    return len(self.image_list)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()
        
    image, label = self.process_image(self.image_list[idx], self.label_list[idx])
    caption = [""]
    sample = self.prepare_dict(image, label, caption)
    return sample
  
  def create_path(self, root, split="validation/"):
    """This function creates data loading paths."""
    image_path = []
    label_path = []

    image_folder = root + "/images/" + split
    label_folder = root + "/annotations/" + split

    for img, label in zip(sorted(os.listdir(image_folder)), sorted(os.listdir(label_folder))):
        image_path.append(os.path.join(image_folder, img))
        label_path.append(os.path.join(label_folder, label))
    return image_path, label_path
  
  def augmenter(self, x, crop_size, augment_size, data_type="image"):
    interpolation = T.InterpolationMode.NEAREST if data_type == "label" else T.InterpolationMode.BILINEAR
    x = T.Compose([T.CenterCrop((crop_size, crop_size)),
                         T.Resize((augment_size, augment_size), interpolation=interpolation, antialias=None)])(x)

    return x
  
  def custom_transform(self, img, augment_size, data_type="image"):
    interpolation = T.InterpolationMode.NEAREST if data_type == "label" else T.InterpolationMode.BILINEAR
    my_transform = T.Compose([
        T.RandomCrop((756, 756)),
        T.Resize((augment_size, augment_size), interpolation=interpolation, antialias=None),
        T.RandomHorizontalFlip()])
    
    random_apply = T.RandomApply([T.Lambda(my_transform)], p=0.5)
    x = random_apply(img)
    return x
  
  def process_image(self, image_path, label_path):
    """This function reads and resizes images and labels."""
    image = torchvision.io.read_image(image_path, ImageReadMode.RGB) / 255.
    label = torchvision.io.read_image(label_path)

    s = image.shape
    w, h = s[1], s[2]
    c = min(w, h) - 1
    image = self.augmenter(image, c, self.RESOLUTION)
    label = self.augmenter(label, c, self.RESOLUTION, data_type="label")

    if self.augment_data:
      seed = torch.randint(100000000, (1,)).item()
      torch.manual_seed(seed)
      image = self.custom_transform(image, self.RESOLUTION, data_type="image")
      torch.manual_seed(seed)
      label = self.custom_transform(label, self.RESOLUTION, data_type="label")

    return image, label
  
  def prepare_dict(self, image_batch, label_batch, caption_batch):
    return {
        "images": image_batch,
        "labels": label_batch,
        "captions": caption_batch
    }

