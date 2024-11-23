import os
import argparse
from typing import Literal
import logging

import torch
import torch.nn.functional as F
import pandas as pd

from scipy.ndimage import median_filter
from torchmetrics import JaccardIndex

from dataloader.iterator import DataIterator, get_fine_to_coarse, load_imdb

from diffcut.recursive_normalized_cut import DiffCut
from tools.ldm import LdmExtractor
from tools.pamr import PAMR
from tools.utils import MaskPooling
from tools.clip_classifier import CLIP
from tools.clip_classifier import open_vocabulary, get_classification_logits

from detectron2.data import MetadataCatalog
from data.metadata import datasets

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Benchmark_Segmentation:
    def __init__(self,
                 model_name: Literal["SSD-1B", "SSD-vega", "SD1.4"] = "SSD-1B",
                 dataset_name: Literal["COCO-Object", "VOC20", "Context"] = "VOC20",
                 step: int = 50,
                 img_size: int = 1024,
                 refinement: bool = False,
                 alpha: int = 10,
                 ):
        
        refining = "pamr" if refinement else "no_pamr"

        self.root_path = f'./OpenVocabulary_Evaluation/{dataset_name}/{refining}'
        self.folder_path = os.path.join(self.root_path)
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.img_size = img_size
        self.step = step
        self.refinement = refinement
        self.dataset_name = dataset_name
        self.alpha = alpha
        self.diffcut = DiffCut()
        self.mask_pooling = MaskPooling()

        if dataset_name == "COCO-Object":
            file_list = load_imdb("./dataloader/coco/val2017/Coco164kFull_Stuff_Coarse_7.txt")
            root = "./datasets/coco"
            dataset = DataIterator(dataset_name, root, "val", file_list, self.img_size)
            fine_to_coarse_map = get_fine_to_coarse("COCO-Object", "./dataloader/coco/coco_object_mapping.pickle")
            self.test_metadata = MetadataCatalog.get("openvocab_coco_2017_val_panoptic")
            self.test_metadata.thing_classes.insert(0, 'background')
            self.N_CLASS = 81 # 80 classes + background
        elif dataset_name == "VOC20":
            root = "./datasets/pascal_voc_d2"
            dataset = DataIterator(dataset_name, root, "validation", None, self.img_size)
            fine_to_coarse_map = get_fine_to_coarse("VOC20")
            self.test_metadata = MetadataCatalog.get("openvocab_pascal21_sem_seg_val")
            self.N_CLASS = 21 # 20 classes + background
        elif dataset_name == "Context":
            root = "./datasets/pascal_ctx_d2"
            dataset = DataIterator(dataset_name, root, "validation", None, self.img_size)
            fine_to_coarse_map = get_fine_to_coarse("Context")
            self.test_metadata = MetadataCatalog.get("openvocab_pascal_ctx59_sem_seg_val")
            self.test_metadata.stuff_classes.insert(0, 'background')
            self.N_CLASS = 60 # 59 classes + background

        self.dataset = dataset
        self.fine_to_coarse_map = fine_to_coarse_map

        self.extractor = LdmExtractor(model_name=model_name)

        with torch.no_grad():
            self.clip_backbone = CLIP(model_name="convnext_large_d_320", pretrained="laion2b_s29b_b131k_ft_soup").to("cuda")
            self.clip_backbone.clip_model.transformer.batch_first = False

        self.ov = open_vocabulary(self.clip_backbone, self.test_metadata, self.test_metadata)
        self.text_classifier, self.num_templates = self.ov.get_text_classifier()

    def get_features(self, images):
            features = self.extractor(images, step=self.step, img_size=self.img_size)
            return features
    
    def pamr(self, labels, image):
        masks = torch.cat([1. * (labels == label) for label in torch.unique(labels)], dim=1)
        labels = PAMR(num_iter=15, dilations=[1, 2, 4, 8, 12, 24, 32, 64])(image, masks) 
        labels = 1. * torch.argmax(labels, dim=1)
        labels = median_filter(labels.cpu().numpy(), 3).astype(int)
        return labels
    
    def associate_label(self, image, mask):
        final_mask = torch.zeros_like(mask).to("cuda")

        if self.dataset_name == "COCO-Object":
            image = F.interpolate(image, size=(2048, 2048), mode='bilinear')

        with torch.no_grad():
            features = self.clip_backbone(image)
        clip_features = features["clip_vis_dense"]

        for i in torch.unique(mask):
            cls_idx = 1.*(mask == i.item())
            mask_embed = self.mask_pooling(clip_features, cls_idx)[0]
            pooled_clip_feature = mask_embed.reshape(1, 1, -1)

            with torch.no_grad():
                pooled_clip_feature = self.clip_backbone.visual_prediction_forward_convnext(pooled_clip_feature)

            out_vocab_cls_results = get_classification_logits(pooled_clip_feature, self.text_classifier, self.clip_backbone.clip_model.logit_scale, self.num_templates)
            idx = torch.argmax(out_vocab_cls_results[..., :-1].softmax(-1)).item()
            final_mask[mask==i] = idx
        return final_mask

    def evaluate(self,
                 tau: int = 0.5):
                 
        #Dataloader
        validation_dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1)
        jaccard = JaccardIndex(task="multiclass", num_classes=self.N_CLASS).to("cuda")

        ALL = 0
        IoU = 0

        for i, batch in enumerate(validation_dataloader):
            # Transfer to GPU
            batch_size = batch["images"].shape[0]
            images = batch["images"].to("cuda")
            labels = self.fine_to_coarse_map(batch["labels"])

            features = self.extractor(images, step=self.step, img_size=self.img_size)
            
            for j in range(batch_size):
                img_feat = features[j][None].to(torch.float32)
                label_map = labels[j]
                pred = self.diffcut.generate_masks(img_feat, tau, mask_size=(1024, 1024), alpha=self.alpha, img_size=self.img_size)
                pred += 1

                if self.refinement:
                    pred = torch.Tensor(pred).to("cuda")
                    pred = self.pamr(pred, images[j][None])[None]

                pred = self.associate_label(images[j][None], torch.Tensor(pred).to("cuda"))

                # Interpolate label_map on gpu
                label_map = F.interpolate(torch.Tensor(label_map)[None].to("cuda"), size=(1024, 1024), mode='nearest-exact')
                label_map += 1

                IoU += jaccard(pred, label_map)
                if torch.isnan(IoU):
                    IoU += 0
                ALL += 1

            # Print accuracy and mean IoU occasionally.
            if (i+1) % 10 == 0:
                mIoU = IoU / ALL
                logging.info("mIoU:{}".format(mIoU))

        # Print final mean IoU.
        mIoU = IoU / ALL
        logging.info("mIoU: %s", mIoU)

        # Save results in a csv file.
        new_row = [{'mIoU': mIoU}]
        df = pd.DataFrame(new_row, columns = ['mIoU'])
        file_path = self.folder_path + f'/{tau}'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        df.to_csv(os.path.join(file_path, f'eval_alpha_{self.alpha}_tau_{tau}.csv'), index=False)

        
def parse_args():
    parser = argparse.ArgumentParser("Segmentation Benchmark Script")
    parser.add_argument("--model_name", type=str, default="SSD-1B", help="Model name")
    parser.add_argument("--dataset_name", type=str, default="VOC20", choices=["COCO-Object", "VOC20", "Context"], help="dataset")
    parser.add_argument("--step", type=int, default=50, help="Denoising timestep")
    parser.add_argument("--img_size", type=int, default=1024, help="Size of input images")
    parser.add_argument("--refinement", dest='refinement', default=True, action='store_true', help="Mask refinement with PAMR")
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