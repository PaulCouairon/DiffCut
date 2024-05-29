import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import pandas as pd
from utils import hungarian_matching
import argparse
from ldm import LdmExtractor
from features_extraction import FeatureExtractor

import torch
import torch.nn.functional as F

from data.PascalVOC import pascalvoc20
from data.PascalVOC.pascalvoc20 import VOCDataset

from data.PascalContext import pascalcontext59
from data.PascalContext.pascalcontext59 import ContextDataset

from data.cityscapes import cityscapes
from data.cityscapes.cityscapes import CityscapesDataset

from data.ADE20K import ade20k
from data.ADE20K.ade20k import ADE20KDataset

from data.coco import coco
from data.coco.coco import COCODataset, load_imdb

from typing import Literal
import os

from diffcut import DiffCut
from pamr import PAMR
from scipy.ndimage import median_filter

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('message')

class Benchmark_Segmentation:
    def __init__(self, 
                 model_name: str = "SSD-1B", 
                 dataset_name: Literal["COCO-Stuff", "COCO-Object", "VOC20", "Cityscapes", \
                                       "Context", "ADE20K"] = "VOC20", 
                 inverse_scheduler: str = "inv_ddim", 
                 guidance_scale: float = 1.,
                 steps: tuple = (50,),
                 n_steps: int = 10,
                 encode_method: str = "noise",
                 img_size: int = 1024,
                 layer_idx: int = -1,
                 refinement: bool = False,
                 alpha: int = 10
                 ):
        
        refining = "pamr" if refinement else "no_pamr"

        self.root_path = './DiffCut_Evaluation/{}/{}/{}/{}'.format(alpha, img_size, dataset_name, refining)
        subfolder_path = '{}/{}'.format(encode_method, steps[0])
        self.folder_path = os.path.join(self.root_path, subfolder_path)
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        
        self.model_name = model_name
        self.img_size = img_size
        self.steps = steps
        self.n_steps = n_steps
        self.encode_method = encode_method
        self.guidance_scale = guidance_scale
        self.layer_idx = layer_idx
        self.refinement = refinement
        self.dataset_name = dataset_name
        self.alpha = alpha
        self.diffcut = DiffCut()

        if dataset_name == "COCO-Stuff":
            file_list = load_imdb("../data/coco/val2017/Coco164kFull_Stuff_Coarse_7.txt")
            dataset = COCODataset("./datasets/coco", file_list, resolution=self.img_size)
            fine_to_coarse_map = coco.get_fine_to_coarse_stuff("../data/coco/fine_to_coarse_dict.pickle")
            self.N_CLASS = 27
        elif dataset_name == "COCO-Object":
            file_list = load_imdb("../data/coco/val2017/Coco164kFull_Stuff_Coarse_7.txt")
            dataset = COCODataset("./datasets/coco", file_list, resolution=self.img_size)
            fine_to_coarse_map = coco.get_fine_to_coarse_object("../data/coco/coco_object_mapping.pickle")
            self.N_CLASS = 81 # 80 classes + background
        elif dataset_name == "VOC20":
            dataset = VOCDataset("./datasets/pascal_voc_d2", resolution=self.img_size)
            fine_to_coarse_map = pascalvoc20.get_fine_to_coarse()
            self.N_CLASS = 21 # 20 classes + background
        elif dataset_name == "Context":
            dataset = ContextDataset("./datasets/pascal_ctx_d2", resolution=self.img_size)
            fine_to_coarse_map = pascalcontext59.get_fine_to_coarse()
            self.N_CLASS = 60 # 59 classes + background
        elif dataset_name == "Cityscapes":
            dataset = CityscapesDataset("./datasets/cityscapes", resolution=self.img_size)
            fine_to_coarse_map = cityscapes.get_fine_to_coarse()
            self.N_CLASS = 27
        elif dataset_name == "ADE20K":
            dataset = ADE20KDataset("./datasets/ADEChallengeData2016", resolution=self.img_size)
            fine_to_coarse_map = ade20k.get_fine_to_coarse()
            self.N_CLASS = 150

        self.dataset = dataset
        self.fine_to_coarse_map = fine_to_coarse_map
        
        self.extractor = LdmExtractor(model_name=model_name, inverse_scheduler=inverse_scheduler)

    def get_features(self, images, captions=[""]):
        batched_input = {'img': images, 'caption': captions}
        features = self.extractor(batched_input, guidance_scale=self.guidance_scale, \
                                    n_steps=self.n_steps, steps=self.steps, encode_method=self.encode_method)
        return features
    
    def pamr(self, labels, image):
        masks = torch.cat([1. * (labels == label) for label in torch.unique(labels)], dim=1)
        labels = PAMR(num_iter=10, dilations=[1, 2, 4, 8])(image, masks) # 1, 2, 4, 8
        labels = 1. * torch.argmax(labels, dim=1)
        labels = median_filter(labels.cpu().numpy(), 3).astype(int)
        return labels

    def evaluate(self,
                 use_captions: bool = False, 
                 tau : int = 0.5,
                 batch_size: int = 5):
                 
        #Dataloader
        validation_dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size)

        TP = np.zeros(self.N_CLASS)
        FP = np.zeros(self.N_CLASS)
        FN = np.zeros(self.N_CLASS)
        ALL = 0

        for i, batch in enumerate(validation_dataloader):
            # Transfer to GPU
            images = batch["images"].to("cuda")
            labels = self.fine_to_coarse_map(batch["labels"])
            if use_captions:
                captions = batch["captions"]
            else:
                captions = [""] * batch_size

            features = self.get_features(images, captions)
            features = features['down_blocks.2.attentions.1.transformer_blocks.3']
            h = self.img_size//32
            features = features.reshape(batch_size, h, h, -1).permute(0, 3, 1, 2)
            
            for j in range(batch_size):
                img_feat = features[j][None].to(torch.float32)
                label_map = labels[j]

                pred = self.diffcut.generate_masks(img_feat, tau, mask_size=(128, 128), alpha=self.alpha)
                
                # Interpolate label_map on gpu
                label_map = F.interpolate(torch.Tensor(label_map)[None].to("cuda"), size=(128, 128))
                label_map = label_map.cpu().numpy().astype(int)

                # Many-to-one matching for background
                if self.dataset_name in ["VOC20", "Context", "COCO-Object"]:
                    _, _, _, _, (hist, col_ind) = hungarian_matching(pred, label_map, self.N_CLASS-1)
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

                tp, fp, fn, all, _ = hungarian_matching(pred, label_map, self.N_CLASS)
                TP += tp
                FP += fp
                FN += fn
                ALL += all

            # Print accuracy and mean IoU occasionally.
            if (i+1) % 10 == 0:
                acc = TP.sum()/ALL
                iou = TP / (TP + FP + FN)
                miou = np.nanmean(iou)
                logging.info("pixel accuracy:{} mIoU:{}".format(acc, miou))

        # Print final accuracy and mean IoU.
        acc = TP.sum()/ALL
        iou = TP / (TP + FP + FN)
        miou = np.nanmean(iou)

        new_row = [{'Map Size': h, 'Acc': acc, 'mIoU': miou}]

        df = pd.DataFrame(new_row, columns = ['Map Size', 'Acc', 'mIoU'])
        file_path = self.folder_path + '/{}'.format(tau)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        df.to_csv(os.path.join(file_path, '{}_{}.csv'.format("down_self_32", h)), index=False)
        return
        
def parse_args():
    parser = argparse.ArgumentParser("Segmentation Benchmark Script")
    parser.add_argument("--model_name", type=str, help="model to evaluate")
    parser.add_argument("--dataset_name", type=str, help="dataset", choices=["COCO-Stuff", "COCO-Object", "VOC20", "Cityscapes", "Context", "ADE20K"])
    parser.add_argument("--inverse_scheduler", type=str, default=None, choices=[None, 'inv_ddim', 'inv_dpmsolver++'], help="Inverse scheduler of the diffusion model")
    parser.add_argument("--guidance_scale", type=float, default=1., help="guidance scale")
    parser.add_argument("--steps", type=int, default=50, help="Timesteps at which we extract the features")
    parser.add_argument("--n_steps", type=int, default=10, help="Scheduler's number of timesteps")
    parser.add_argument("--encode_method", type=str, default="noise", choices=["noise", "inversion"], help="Encoding method")
    parser.add_argument("--img_size", type=int, default=1024, help="Size of input images")
    parser.add_argument("--refinement", dest='refinement', default=False, action='store_true')
    parser.add_argument("--tau", type=float, default=0.5, help="Threshold for Recursive NCut")
    parser.add_argument("--alpha", type=int, default=10)

    args = parser.parse_args()
    logging.info('args parsed')

    return args

if __name__ == "__main__":
    args = parse_args()
    logging.info(args)
    sem_seg = Benchmark_Segmentation(model_name=args.model_name, dataset_name=args.dataset_name, steps=(args.steps,), \
                                 n_steps=args.n_steps, encode_method=args.encode_method, img_size=args.img_size, refinement=args.refinement, alpha=args.alpha)
    sem_seg.evaluate(tau=args.tau)