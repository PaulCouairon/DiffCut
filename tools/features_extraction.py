import torch
from torch import nn
import timm
from timm.data import resolve_data_config
import torchvision.transforms as T

class FeatureExtractor(nn.Module):
    def __init__(self, model_name, img_size=384, device="cuda"):
        super().__init__()
        self.model_name = model_name

        self.model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
        img_size=img_size).to(device)

        config = resolve_data_config(self.model.pretrained_cfg, model=self.model)

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),  # Resize the image
            T.Normalize(mean=config['mean'], std=config['std']),  # Normalize
            ])

    def forward(self, img):
        img = self.transform(img)

        with torch.no_grad():
            features = self.model.get_intermediate_layers(
                img,
                n = len(self.model.blocks),
                reshape = True,
                return_prefix_tokens = True,
                norm = True,
            )

        return features
