"""cityscapes_data.py
Data pre-processing and loading pipeline for Cityscapes.
"""

import os
import numpy as np
import torch
import torchvision
from torchvision.io import ImageReadMode
import torchvision.transforms as T
from torch.utils.data import Dataset
import glob

def id2label():
  labels = {'unlabeled':-1 ,
          'road'         : 0 ,
          'sidewalk'     : 1 ,
          'parking'      : 2 ,
          'rail track'   : 3 ,
          'building'     : 4 ,
          'wall'         : 5 ,
          'fence'        : 6 ,
          'guard rail'   : 7 ,
          'bridge'       : 8 ,
          'tunnel'       : 9 ,
          'pole'         : 10 ,
          'polegroup'    : 11 ,
          'traffic light': 12 ,
          'traffic sign' : 13 ,
          'vegetation'   : 14 ,
          'terrain'      : 15 ,
          'sky'          : 16 ,
          'person'       : 17 ,
          'rider'        : 18 ,
          'car'          : 19 ,
          'truck'        : 20 ,
          'bus'          : 21 ,
          'caravan'      : 22 ,
          'trailer'      : 23 ,
          'train'        : 24 ,
          'motorcycle'   : 25 ,
          'bicycle'      : 26 ,
          'license plate': -1 }
    
  return labels

def get_fine_to_coarse():
  """Map fine label indexing to coarse label indexing."""
  label_dict = {
      0: -1,
      1: -1,
      2: -1,
      3: -1,
      4: -1,
      5: -1,
      6: -1,
      7: 0,
      8: 1,
      9: 2,
      10: 3,
      11: 4,
      12: 5,
      13: 6,
      14: 7,
      15: 8,
      16: 9,
      17: 10,
      18: 11,
      19: 12,
      20: 13,
      21: 14,
      22: 15,
      23: 16,
      24: 17,
      25: 18,
      26: 19,
      27: 20,
      28: 21,
      29: 22,
      30: 23,
      31: 24,
      32: 25,
      33: 26,
      -1: -1,
  }
  cityscape_labelmap = np.vectorize(lambda x: label_dict[x])
  return cityscape_labelmap

class CityscapesDataset(Dataset):
  def __init__(self, ROOT, augment_data=False, resolution=1024, split="val"):
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
  
  def create_path(self, root, split="val"):
    """This function creates data loading paths."""
    image_path = []
    label_path = []

    image_folder = "leftImg8bit/" + split
    label_folder = "gtFine/" + split

    for folder in os.listdir(os.path.join(root, image_folder)):
      for file_path in glob.glob(
          os.path.join(root, image_folder, folder, "*.png")
      ):
        image_path.append(file_path)
        label_path.append(file_path.replace("leftImg8bit","gtFine").replace(".png","_labelIds.png"))
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