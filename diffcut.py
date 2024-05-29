import sys
sys.path.append('../')
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from scipy.linalg import eigh
from utils import MaskPooling

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('message')

class DiffCut:
    def get_affinity_matrix(self, affinity, level):
        size = affinity.shape[0]
        null_values_idx = set(torch.unique(torch.where(affinity==0)[0]).tolist())
        all_idx = set(np.arange(0, size, 1))
        if level == 0:
            idx2keep = list(all_idx)
            A = affinity
        else:
            idx2keep = list(all_idx - null_values_idx)
            A = affinity[:, idx2keep][idx2keep, :]
        D = self.get_degree_matrix(A)
        return A, D, idx2keep
    
    def get_degree_matrix(self, A):
        d_i = torch.sum(A, axis=1)
        D = torch.diag(d_i)
        return D
    
    def eigen_decomposition(self, A, D):
        D_inv_sqrt = torch.diag(1 / torch.diag(D) ** 0.5)
        eigenvalues, eigenvectors = torch.linalg.eigh(D_inv_sqrt @ (D - A) @ D_inv_sqrt)
        eigenvectors = D_inv_sqrt @ eigenvectors
        return eigenvalues, eigenvectors

    def second_smallest_eigenvector(self, A, D):
        # get the second smallest eigenvector from affinity matrix
        D_inv_sqrt = torch.diag(1 / torch.diag(D) ** 0.5)
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(D_inv_sqrt @ (D - A) @ D_inv_sqrt)
            second_smallest_vec = D_inv_sqrt @ eigenvectors[:, 1]
        except torch._C._LinAlgError:
            second_smallest_vec = None
        return eigenvalues, second_smallest_vec

    def get_bipartition(self, y, A, D, k=100):
        xmin, xmax = y.min() * 0.99, y.max() * 0.99
        thresholds = torch.linspace(xmin, xmax, k)
        partitions = [1. * (y > thresh) for thresh in thresholds]
        N_cut_values = []
        L = (D - A)
        sum_D = torch.sum(D)
        diag_D = torch.diag(D)

        for x in partitions:
            x = 2 * x - 1
            k = torch.sum(diag_D * (x > 0)) / sum_D
            b = k / (1 - k)
            y = (1 + x) - b * (1 - x)
            E = (torch.t(y) @ L @ y) / (torch.t(y) @ D @ y)
            N_cut_values.append(E.item())

        ncut = min(N_cut_values)
        bipartition = partitions[np.argmin(N_cut_values)]
        return bipartition, ncut

    def get_masked_affinity_matrix(self, painting, affinity, mask):
        # mask out affinity matrix based on the painting matrix 
        painting = painting + mask.unsqueeze(0)
        painting[painting > 0] = 1
        painting[painting <= 0] = 0
        mask_affinity = (torch.t(painting.reshape(1, -1)) @ painting.reshape(1, -1))
        affinity = affinity * (1 - mask_affinity)
        return affinity, painting

    def deterministic_vector_sign_flip(self, u):
        max_abs_idx = torch.argmax(torch.abs(u))
        sign = torch.sign(u[max_abs_idx])
        u *= sign
        return u

    def recursive_ncut(self, feats, tau=None, dims=(64, 64), painting=None, mask=None, accumulator_masks=[], level=0, alpha=10):
        """
        Implementation of recursive NCut.
        Inputs
        feats: the pixel/patch features of an imageL
        tau: float value that stops the recursion.
        dims: dimension of the map from which the features are used.
        painting: registers the explored areas.
        mask: area of current bipartition.
        accumulator: registers the segmentation masks.
        level: counts the level of recursion.
        alpha: exponent
        """
        if level == 0:
            feats = F.normalize(feats, p=2, dim=0)
            painting = torch.zeros(dims).to("cuda")
            mask = torch.zeros(dims).to("cuda")
            affinity = ((torch.t(feats) @ feats))
            affinity = (affinity - affinity.min()) / (affinity.max() - affinity.min())
            affinity = affinity ** alpha

        if level > 0:
            affinity = feats

        # Mask the explored area in the affinity matrix
        affinity, painting = self.get_masked_affinity_matrix(painting, affinity, mask)

        # Construct a smaller affinity matrix (A) and degree matrix (D)
        A, D, idx2keep = self.get_affinity_matrix(affinity, level)

        if A.shape[0] > 1:
            # get the second smallest eigenvector
            eigenvalues, second_smallest_vec = self.second_smallest_eigenvector(A, D)
            
            second_smallest_vec = self.deterministic_vector_sign_flip(second_smallest_vec)
            null_vec = torch.zeros((1, dims[0]*dims[1])).to("cuda")
            null_vec[:, idx2keep] = second_smallest_vec

            # get current graph bipartition
            bipartition, ncut = self.get_bipartition(second_smallest_vec, A, D)

            null_vec = torch.zeros((1, dims[0]*dims[1])).to("cuda")
            null_vec[:, idx2keep] = bipartition
            bipartition = (1. * null_vec.reshape(dims))

            if (ncut < tau):
                accumulator_masks.append(bipartition)

                self.recursive_ncut(affinity, tau, dims, painting.to("cuda"), 1 - bipartition, accumulator_masks, level+1)
                self.recursive_ncut(affinity, tau, dims, painting.to("cuda"), bipartition, accumulator_masks, level+1)

        return accumulator_masks


    def assemble_clusters(self, clusters, h, w):
        mask = torch.zeros((h, w)).to("cuda")
        final_mask = torch.zeros((h, w)).to("cuda")
        max_value = 1
        for cluster_mask in clusters:
            mask += max_value * cluster_mask
            max_value = mask.max() + 1

        for i, cls_idx in enumerate(torch.unique(mask)):
            final_mask[mask==cls_idx] = i

        return final_mask

    
    def generate_masks(self, features, tau=0.5, mask_size=(128, 128), masks=None, alpha=10):
        # Step 1: Recursive Feature Clustering
        _, c, h, w = features.shape
        h_mask, w_mask = mask_size
        feats_norm = F.normalize(features, p=2, dim=1).to(torch.float32)
        mask_pooling = MaskPooling()

        if masks is None:
            x = feats_norm.reshape(c, h*w)
            clusters = self.recursive_ncut(x, tau=tau, dims=(h, h), accumulator_masks=[], alpha=alpha)

            if len(clusters) == 0:
                clusters.append(torch.zeros((h, w)).to("cuda"))

            # Assemble clusters
            masks = self.assemble_clusters(clusters, h, w)[None, None]
        
        # Step 2: Masked Spatial Marginal Mean
        clusters_embeddings = torch.zeros(len(torch.unique(masks)), c).to("cuda")

        for k, cls_idx in enumerate(torch.unique(masks)):
            mask = (masks == cls_idx)
            mask_embed = mask_pooling(feats_norm, mask)[0]
            clusters_embeddings[k, :] = mask_embed

        # Step 3: High-Resolution Concept Assignment
        # Feature upsampling
        upsampled_feats = torch.nn.Upsample(size=mask_size, mode='bilinear')(feats_norm)

        # Label association
        masks = torch.argmax(torch.t(upsampled_feats.reshape(c, h_mask*w_mask)) @ torch.t(clusters_embeddings), dim=1).reshape(1, 1, h_mask, w_mask)

        masks = masks.cpu().numpy().astype(int)

        return masks