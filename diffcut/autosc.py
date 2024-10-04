import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import qr, svd
from tools.utils import MaskPooling

class DiffCut_AutoSC:
    def get_affinity(self, features, rw=1):
        feats_norm = F.normalize(features, p=2, dim=0)
        affinity = ((torch.t(feats_norm) @ feats_norm))
        affinity = (affinity - affinity.min()) / (affinity.max() - affinity.min()) 
        A = affinity ** rw
        return A
    
    def get_degree(self, A):
        d_i = torch.sum(A, axis=1)
        D = torch.diag(d_i)
        return D

    def eigen_decomposition(self, A, D):
        D_inv_sqrt = torch.diag(1 / torch.diag(D) ** 0.5)
        eigenvalues, eigenvectors = torch.linalg.eigh(D_inv_sqrt @ (D - A) @ D_inv_sqrt)
        eigenvectors = D_inv_sqrt @ eigenvectors
        return eigenvalues, eigenvectors

    def deterministic_vector_sign_flip(self, u):
        max_abs_idx = torch.argmax(torch.abs(u))
        sign = torch.sign(u[max_abs_idx])
        u *= sign
        return u
    
    def cluster_qr(self, vectors):
        k = vectors.shape[1]
        _, _, piv = qr(vectors.T, pivoting=True)
        ut, _, v = svd(vectors[piv[:k], :].T)
        vectors = abs(np.dot(vectors, np.dot(ut, v.conj())))
        return vectors.argmax(axis=1)

    def ncut(self, feats, dims=(64, 64)):
        max_gap = 0
        affinity = self.get_affinity(feats)
        for alpha in [1, 5, 10, 15]:
            A = affinity ** alpha
            D = self.get_degree(A)
            eigenvalues, eigenvectors = self.eigen_decomposition(A, D)

            # Eigengap Heuristic
            cumsum_eigenvalues = torch.cumsum(eigenvalues, dim=0)
            index = torch.arange(1, cumsum_eigenvalues.size(0) + 1, device=cumsum_eigenvalues.device)
            cummean = cumsum_eigenvalues / index
            reg_L = (eigenvalues - cummean) / (cummean + 1e-6)

            if torch.max(reg_L) > max_gap:
                max_gap = torch.max(reg_L)
                index_largest_gap = torch.argmax(reg_L)
                nb_clusters = index_largest_gap
                vectors = eigenvectors

        vectors = eigenvectors[:, :nb_clusters].cpu().numpy()
        masks = self.cluster_qr(vectors)
        masks = masks.reshape(dims)
        return masks
    
    def generate_masks(self, features, mask_size=(128, 128), masks=None, img_size=1024):
        # Step 1: Features clustering
        h_mask, w_mask = mask_size
        mask_pooling = MaskPooling()
        feats = features.reshape(1, img_size//32, img_size//32, -1).permute(0, 3, 1, 2).to(torch.float32)
        feats_norm = F.normalize(feats, dim=1)
        _, c, h, w = feats_norm.shape

        if masks is None:
            x = feats_norm.reshape(c, h*w)
            masks = self.ncut(x, dims=(h, h))
            masks = torch.Tensor(masks[None, None]).to("cuda")
        
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
