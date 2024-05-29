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
from pycocotools.coco import COCO

def id2label():
  labels = {0: 'unlabeled',
            1: 'electronic-things',
            2: 'appliance-things',
            3: 'food-things',
            4: 'furniture-things',
            5: 'indoor-things',
            6: 'kitchen-things',
            7: 'accessory-things',
            8: 'animal-things',
            9: 'outdoor-things',
            10: 'person-things',
            11: 'sports-things',
            12: 'vehicle-things',
            13: 'ceiling-stuff',
            14: 'floor-stuff',
            15: 'food-stuff',
            16: 'furniture-stuff',
            17: 'rawmaterial-stuff',
            18: 'textile-stuff',
            19: 'wall-stuff',
            20: 'window-stuff',
            21: 'building-stuff',
            22: 'ground-stuff',
            23: 'plant-stuff',
            24: 'sky-stuff',
            25: 'solid-stuff',
            26: 'structural-stuff',
            27: 'water-stuff',
          }
  return labels

def get_fine_to_coarse_stuff(fine_to_coarse_path):
  """Map fine label indexing to coarse label indexing."""
  with open(fine_to_coarse_path, "rb") as f:
    d = pickle.load(f)
  fine_to_coarse_dict = d["fine_index_to_coarse_index"]
  fine_to_coarse_dict[255] = -1
  fine_to_coarse_map = np.vectorize(
      lambda x: fine_to_coarse_dict[x]
  )  # not in-place.
  return fine_to_coarse_map

def get_fine_to_coarse_object(fine_to_coarse_path):
  """Map fine label indexing to coarse label indexing."""
  with open(fine_to_coarse_path, "rb") as f:
    fine_to_coarse_dict = pickle.load(f)
  fine_to_coarse_dict = {key: fine_to_coarse_dict[key] if key < 91 else -1 for key in fine_to_coarse_dict}
  #fine_to_coarse_dict[255] = -1
  fine_to_coarse_map = np.vectorize(
      lambda x: fine_to_coarse_dict[x]
  )  # not in-place.
  return fine_to_coarse_map

def load_imdb(imdb_path):
  with open(imdb_path, "r") as f:
    imdb = tuple(f)
    imdb = [id_.rstrip() for id_ in imdb]
    return imdb

class COCODataset(Dataset):
  def __init__(self, ROOT, file_list, augment_data=False, resolution=1024, split="val"):
    self.image_list, self.label_list, self.coco_anns = self.create_path(ROOT, file_list, split)
    self.augment_data = augment_data
    self.RESOLUTION = resolution

  def __len__(self):
    return len(self.image_list)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()

    image_id = int(self.image_list[idx].split("/")[-1].split(".")[0])
    image, label = self.process_image(self.image_list[idx], self.label_list[idx])
    caption_id = self.coco_anns.getAnnIds(image_id)[0]
    caption = self.coco_anns.anns[caption_id]["caption"]

    sample = self.prepare_dict(image, label, caption)
    return sample
  
  def create_path(self, root, file_list, split="val"):
    """This function creates data loading paths."""
    image_path = []
    label_path = []

    image_folder = split + "2017"
    label_folder = "annotations/panoptic_semseg_" + split + "2017"
    caption_file = "annotations/captions_" + split +"2017.json"

    for file in file_list:
      image_path.append(os.path.join(root, image_folder, file + ".jpg"))
      label_path.append(os.path.join(root, label_folder, file + ".png"))

    coco_anns = COCO(os.path.join(root, caption_file))
    return image_path, label_path, coco_anns
  
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