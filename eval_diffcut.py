import os
import argparse
from typing import Literal
import logging

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from scipy.ndimage import median_filter

from diffcut.recursive_normalized_cut import DiffCut
from tools.ldm import LdmExtractor
from tools.pamr import PAMR
from tools.utils import hungarian_matching

from dataloader.iterator import DataIterator, get_fine_to_coarse, load_imdb

np.seterr(divide='ignore', invalid='ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Benchmark_Segmentation:
    def __init__(self,
                 model_name: Literal["SSD-1B", "SSD-vega", "SD1.4"] = "SSD-1B",
                 dataset_name: Literal["COCO-Stuff", "COCO-Object", "VOC20", "Cityscapes", \
                                       "Context", "ADE20K"] = "VOC20",
                 step: int = 50,
                 img_size: int = 1024,
                 refinement: bool = False,
                 alpha: int = 10,
                 ):

        refining = "pamr" if refinement else "no_pamr"

        self.root_path = f'./Evaluation/{dataset_name}/{refining}'
        self.folder_path = os.path.join(self.root_path)
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.img_size = img_size
        self.step = step
        self.refinement = refinement
        self.dataset_name = dataset_name
        self.alpha = alpha
        self.diffcut = DiffCut()

        if dataset_name == "COCO-Stuff":
            file_list = load_imdb("./dataloader/coco/val2017/Coco164kFull_Stuff_Coarse_7.txt")
            root = "./datasets/coco"
            dataset = DataIterator(dataset_name, root, "val", file_list, self.img_size)
            fine_to_coarse_map = get_fine_to_coarse("COCO-Stuff", "./dataloader/coco/fine_to_coarse_dict.pickle")
            self.N_CLASS = 27
        elif dataset_name == "COCO-Object":
            file_list = load_imdb("./dataloader/coco/val2017/Coco164kFull_Stuff_Coarse_7.txt")
            root = "./datasets/coco"
            dataset = DataIterator(dataset_name, root, "val", file_list, self.img_size)
            fine_to_coarse_map = get_fine_to_coarse("COCO-Object", "./dataloader/coco/coco_object_mapping.pickle")
            self.N_CLASS = 81 # 80 classes + background
        elif dataset_name == "VOC20":
            root = "./datasets/pascal_voc_d2"
            dataset = DataIterator(dataset_name, root, "validation", None, self.img_size)
            fine_to_coarse_map = get_fine_to_coarse("VOC20")
            self.N_CLASS = 21 # 20 classes + background
        elif dataset_name == "Context":
            root = "./datasets/pascal_ctx_d2"
            dataset = DataIterator(dataset_name, root, "validation", None, self.img_size)
            fine_to_coarse_map = get_fine_to_coarse("Context")
            self.N_CLASS = 60 # 59 classes + background
        elif dataset_name == "Cityscapes":
            root = "./datasets/cityscapes"
            dataset = DataIterator(dataset_name, root, "val", None, self.img_size)
            fine_to_coarse_map = get_fine_to_coarse("Cityscapes", "./dataloader/cityscapes/cityscapes_27_mapping.pickle")
            self.N_CLASS = 27
        elif dataset_name == "ADE20K":
            root = "./datasets/ADEChallengeData2016"
            dataset = DataIterator(dataset_name, root, "validation", None, self.img_size)
            fine_to_coarse_map = get_fine_to_coarse("ADE20K")
            self.N_CLASS = 150

        self.dataset = dataset
        self.fine_to_coarse_map = fine_to_coarse_map

        self.extractor = LdmExtractor(model_name=model_name)

    def get_features(self, images):
        features = self.extractor(images, step=self.step, img_size=self.img_size)
        return features

    def pamr(self, labels, image):
        masks = torch.cat([1. * (labels == label) for label in torch.unique(labels)], dim=1)
        labels = PAMR(num_iter=10, dilations=[1, 2, 4, 8])(image, masks) # 1, 2, 4, 8
        labels = 1. * torch.argmax(labels, dim=1)
        labels = median_filter(labels.cpu().numpy(), 3).astype(int)
        return labels

    def evaluate(self,
                 tau: int = 0.5):

        #Dataloader
        validation_dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=5)

        TP = np.zeros(self.N_CLASS)
        FP = np.zeros(self.N_CLASS)
        FN = np.zeros(self.N_CLASS)
        ALL = 0

        for i, batch in enumerate(validation_dataloader):
            # Transfer to GPU
            batch_size = batch["images"].shape[0]
            images = batch["images"].to("cuda")
            labels = self.fine_to_coarse_map(batch["labels"])

            features = self.extractor(images, step=self.step, img_size=self.img_size)

            for j in range(batch_size):
                img_feat = features[j:j+1].to(torch.float32)
                label_map = labels[j:j+1]

                pred = self.diffcut.generate_masks(img_feat, tau, mask_size=(128, 128), alpha=self.alpha, img_size=self.img_size)

                # Interpolate label_map on gpu
                label_map = F.interpolate(torch.Tensor(label_map).to("cuda"), size=(128, 128))
                label_map = label_map.cpu().numpy().astype(int)

                # Many-to-one matching for background
                if self.dataset_name in ["VOC20", "Context", "COCO-Object"]:
                    _, _, _, _, hist, col_ind = hungarian_matching(pred, label_map, self.N_CLASS-1)
                    # Assign a valid label to the background in ground truth labels
                    label_map[label_map==-1] = self.N_CLASS-1

                    # Assign a unique label to the background in pred maps
                    assigned_gt_clusters = np.where(hist.max(axis=1)>0)[0].tolist()
                    assigned_pred_clusters = [col_ind[i] for i in assigned_gt_clusters]
                    background_clusters = list(set(np.unique(pred)) - set(assigned_pred_clusters))
                    bg_label = pred.max() + 1

                    for bg_cls in background_clusters:
                        pred[pred==bg_cls] = bg_label

                if self.refinement:
                    pred = torch.Tensor(pred).to("cuda")
                    im =  F.interpolate(images[j][None], size=(128, 128), mode='bilinear')
                    pred = self.pamr(pred, im)[None]

                tp, fp, fn, all, _, _ = hungarian_matching(pred, label_map, self.N_CLASS)
                TP += tp
                FP += fp
                FN += fn
                ALL += all

            # Print accuracy and mean IoU occasionally.
            if (i+1) % 10 == 0:
                acc = TP.sum()/ALL
                iou = TP / (TP + FP + FN)
                miou = np.nanmean(iou)
                logging.info("pixel accuracy: %s mIoU: %s", acc, miou)

        # Print final accuracy and mean IoU.
        acc = TP.sum()/ALL
        iou = TP / (TP + FP + FN)
        miou = np.nanmean(iou)
        logging.info("pixel accuracy: %s mIoU: %s", acc, miou)

        # Save results in a csv file.
        new_row = [{'Acc': acc, 'mIoU': miou}]

        df = pd.DataFrame(new_row, columns = ['Acc', 'mIoU'])
        file_path = self.folder_path + f'/{tau}'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        df.to_csv(os.path.join(file_path, f'eval_alpha_{self.alpha}_tau_{tau}.csv'), index=False)


def parse_args():
    parser = argparse.ArgumentParser("Segmentation Benchmark Script")
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--dataset_name", type=str, choices=["COCO-Stuff", "COCO-Object", "VOC20", "Cityscapes", "Context", "ADE20K"], help="dataset")
    parser.add_argument("--step", type=int, default=50, help="Denoising timestep")
    parser.add_argument("--img_size", type=int, default=1024, help="Size of input images")
    parser.add_argument("--refinement", dest='refinement', default=False, action='store_true', help="Mask refinement with PAMR")
    parser.add_argument("--tau", type=float, default=0.5, help="Threshold value for Recursive NCut")
    parser.add_argument("--alpha", type=int, default=10, help="Affinity matrix exponent value")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    logging.info(args)
    sem_seg = Benchmark_Segmentation(model_name=args.model_name, dataset_name=args.dataset_name,
                                     step=args.step, img_size=args.img_size, refinement=args.refinement,
                                     alpha=args.alpha)
    sem_seg.evaluate(tau=args.tau)
