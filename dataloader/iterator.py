import os
import glob
import pickle
import numpy as np
import torch
import torchvision
from torchvision.io import ImageReadMode
import torchvision.transforms as T
from torch.utils.data import Dataset

def get_fine_to_coarse(dataset_name, fine_to_coarse_path=None):
    if dataset_name == "COCO-Stuff":
        with open(fine_to_coarse_path, "rb") as f:
            d = pickle.load(f)
        fine_to_coarse_dict = d["fine_index_to_coarse_index"]
        fine_to_coarse_dict[255] = -1
        labelmap = np.vectorize(
            lambda x: fine_to_coarse_dict[x]
        )  # not in-place.

    elif dataset_name == "COCO-Object":
        with open(fine_to_coarse_path, "rb") as f:
            fine_to_coarse_dict = pickle.load(f)
        fine_to_coarse_dict = {key: fine_to_coarse_dict[key] if key < 91 else -1 \
                               for key in fine_to_coarse_dict}
        labelmap = np.vectorize(
            lambda x: fine_to_coarse_dict[x]
        )  # not in-place.

    elif dataset_name == "Cityscapes":
        with open(fine_to_coarse_path, "rb") as f:
            fine_to_coarse_dict = pickle.load(f)
        labelmap = np.vectorize(lambda x: fine_to_coarse_dict[x])

    else:
        labelmap = np.vectorize(lambda x: -1 if x==255 else x, otypes=[int])

    return labelmap

def load_imdb(imdb_path):
    with open(imdb_path, "r", encoding="utf-8") as f:
        imdb = tuple(f)
        imdb = [id_.rstrip() for id_ in imdb]
        return imdb


class DataIterator(Dataset):
    def __init__(self, dataset_name, root_path, split, file_list=None, resolution=1024):
        self.dataset_name = dataset_name
        self.image_list, self.label_list = self.create_path(root_path, file_list, split)
        self.resolution = resolution

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, label = self.process_image(self.image_list[idx], self.label_list[idx])
        sample = self.prepare_dict(image, label)
        return sample

    def create_path(self, root, file_list, split="validation/"):
        """This function creates data loading paths."""
        image_path = []
        label_path = []

        paths = {
            "VOC20": (root + "/images/" + split, root + "/annotations_pascal20/" + split),
            "Context": (root + "/images/" + split, root + "/annotations_ctx59/" + split),
            "COCO-Stuff": (split + "2017", root + "/annotations/" + split + "2017"),
            "COCO-Object": (split + "2017", root + "/annotations/" + split + "2017"),
            "Cityscapes": ("leftImg8bit/" + split, "gtFine/" + split),
            "ADE20K": (root + "/images/" + split, root + "/annotations/" + split)
            }

        image_folder, label_folder = paths[self.dataset_name]

        if self.dataset_name == "Cityscapes":
            for folder in os.listdir(os.path.join(root, image_folder)):
                for file_path in glob.glob(
                    os.path.join(root, image_folder, folder, "*.png")
                    ):
                    image_path.append(file_path)
                    label_path.append(
                        file_path.replace("leftImg8bit","gtFine").replace(".png","_labelIds.png"))

        elif "COCO" in self.dataset_name:
            for file in file_list:
                image_path.append(os.path.join(root, image_folder, file + ".jpg"))
                label_path.append(os.path.join(root, label_folder, file + ".png"))

        else:
            sort_images = sorted(os.listdir(image_folder))
            sort_labels = sorted(os.listdir(label_folder))
            for img, label in zip(sort_images, sort_labels):
                image_path.append(os.path.join(image_folder, img))
                label_path.append(os.path.join(label_folder, label))

        return image_path, label_path

    def augmenter(self, x, crop_size, augment_size, data_type="image"):
        interpolation = {"image": T.InterpolationMode.BILINEAR,
                         "label": T.InterpolationMode.NEAREST_EXACT}
        x = T.Compose([T.CenterCrop((crop_size, crop_size)),
                       T.Resize((augment_size, augment_size),
                                interpolation=interpolation[data_type], antialias=None)])(x)

        return x

    def process_image(self, image_path, label_path):
        """This function reads and resizes images and labels."""
        image = torchvision.io.read_image(image_path, ImageReadMode.RGB) / 255.
        label = torchvision.io.read_image(label_path)

        s = image.shape
        w, h = s[1], s[2]
        c = min(w, h) - 1
        image = self.augmenter(image, c, self.resolution)
        label = self.augmenter(label, c, self.resolution, data_type="label")

        return image, label

    def prepare_dict(self, image_batch, label_batch):
        return {
            "images": image_batch,
            "labels": label_batch
        }
