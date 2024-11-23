# Visualization code from https://github.com/Tsingularity/dift/blob/main/src/utils/visualization.py

import gc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

class Demo:

    def __init__(self, imgs, ft, img_size):
        self.ft = ft # N+1, C, H, W
        self.imgs = imgs
        self.num_imgs = len(imgs)
        self.img_size = img_size

    def plot_img_pairs(self, fig_size=3, alpha=0.5, scatter_size=8, show_marker=False):

        fig, axes = plt.subplots(1, self.num_imgs, figsize=(fig_size*self.num_imgs, fig_size))

        plt.tight_layout()

        for i in range(self.num_imgs):
            axes[i].imshow(self.imgs[i])
            axes[i].axis('off')
            if i == 0:
                axes[i].set_title('source image')
            else:
                axes[i].set_title('target image')

        num_channel = self.ft.size(1)

        def onclick(event):
            if event.inaxes == axes[0]:
                with torch.no_grad():

                    x, y = int(np.round(event.xdata)), int(np.round(event.ydata))

                    src_ft = self.ft[0].unsqueeze(0)
                    src_ft = nn.Upsample(size=(self.img_size[0], self.img_size[1]), mode='bilinear')(src_ft)
                    src_vec = src_ft[0, :, y, x].view(1, num_channel)  # 1, C

                    del src_ft
                    gc.collect()
                    torch.cuda.empty_cache()

                    trg_ft = nn.Upsample(size=(self.img_size[0], self.img_size[1]), mode='bilinear')(self.ft[1:]) # N, C, H, W
                    trg_vec = trg_ft.view(self.num_imgs - 1, num_channel, -1) # N, C, HW

                    del trg_ft
                    gc.collect()
                    torch.cuda.empty_cache()

                    src_vec = F.normalize(src_vec.to(torch.float32), dim=1, p=2) # 1, C
                    trg_vec = F.normalize(trg_vec.to(torch.float32), dim=1, p=2) # N, C, HW
                    cos_map = torch.matmul(src_vec, trg_vec).view(self.num_imgs - 1, self.img_size[0], self.img_size[1])
                    cos_map = cos_map.cpu().numpy() # N, H, W

                    axes[0].clear()
                    axes[0].imshow(self.imgs[0])
                    axes[0].axis('off')
                    axes[0].plot(x, y, marker='x', c='r', markersize=scatter_size, mew=1)
                    axes[0].set_title('source image')

                    for i in range(1, self.num_imgs):
                        max_yx = np.unravel_index(cos_map[i-1].argmax(), cos_map[i-1].shape)
                        axes[i].clear()

                        heatmap = cos_map[i-1]
                        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]
                        heatmap = gaussian_filter(heatmap, sigma=3)
                        axes[i].imshow(self.imgs[i])
                        axes[i].imshow(255 * heatmap, alpha=alpha, cmap='jet')
                        axes[i].axis('off')
                        if show_marker:
                            axes[i].plot(max_yx[1].item(), max_yx[0].item(), marker='x', c='r', markersize=scatter_size, mew=1)
                        axes[i].set_title('target image')

                    del cos_map
                    del heatmap
                    gc.collect()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()