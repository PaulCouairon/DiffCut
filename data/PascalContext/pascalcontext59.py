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

def id2label():
  labels = {
    "background": -1,
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "table": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19,
    "bag": 20,
    "bed": 21,
    "bench": 22,
    "book": 23,
    "building": 24,
    "cabinet": 25,
    "ceiling": 26,
    "cloth": 27,
    "computer": 28,
    "cup": 29,
    "door": 30,
    "fence": 31,
    "floor": 32,
    "flower": 33,
    "food": 34,
    "grass": 35,
    "ground": 36,
    "keyboard": 37,
    "light": 38,
    "mountain": 39,
    "mouse": 40,
    "curtain": 41,
    "platform": 42,
    "sign": 43,
    "plate": 44,
    "road": 45,
    "rock": 46,
    "shelves": 47,
    "sidewalk": 48,
    "sky": 49,
    "snow": 50,
    "bedclothes": 51,
    "track": 52,
    "tree": 53,
    "truck": 54,
    "wall": 55,
    "water": 56,
    "window": 57,
    "wood": 58
}
  return labels

def get_fine_to_coarse():
  """Map fine label indexing to coarse label indexing."""
  voc_labelmap = np.vectorize(lambda x: -1 if x==255 else x, otypes=[int])
  return voc_labelmap

class ContextDataset(Dataset):
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
    label_folder = root + "/annotations_ctx59/" + split

    for img, label in zip(sorted(os.listdir(image_folder)), sorted(os.listdir(label_folder))):
        image_path.append(os.path.join(image_folder, img))
        label_path.append(os.path.join(label_folder, label))
    return image_path, label_path
  
  def augmenter(self, x, crop_size, augment_size, data_type="image"):
    interpolation = T.InterpolationMode.NEAREST_EXACT if data_type == "label" else T.InterpolationMode.BILINEAR
    x = T.Compose([T.CenterCrop((crop_size, crop_size)),
                         T.Resize((augment_size, augment_size), interpolation=interpolation, antialias=None)])(x)
    # x = T.Compose([T.Resize((augment_size, augment_size), interpolation=interpolation, antialias=None)])(x)

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

