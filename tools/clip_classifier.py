# Code adapted from https://github.com/bytedance/fc-clip/

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

VILD_PROMPT = [
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]
 
def get_classification_logits(x, text_classifier, logit_scale, num_templates=None):
        # x in shape of [B, *, C]
        # text_classifier in shape of [num_classes, C]
        # logit_scale is a learnable scalar https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py#L201
        # return: [B, *, num_classes]
        x = F.normalize(x, dim=-1)
        logit_scale = torch.clamp(logit_scale.exp(), max=100)
        pred_logits = logit_scale * x @ text_classifier.T # B, *, N + 1
        # max ensembel as in OpenSeg/ODISE
        final_pred_logits = []
        cur_idx = 0
        for num_t in num_templates: 
            final_pred_logits.append(pred_logits[:, :, cur_idx: cur_idx + num_t].max(-1).values)
            cur_idx += num_t
        final_pred_logits.append(pred_logits[:, :, -1]) # the last classifier is for void
        final_pred_logits = torch.stack(final_pred_logits, dim=-1)
        return final_pred_logits

class CLIP(nn.Module):
    def __init__(self, model_name, pretrained):
        super().__init__()
        self.clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.text_tokenizer = open_clip.get_tokenizer(model_name)

    def tokenize_text(self, text):
        return self.text_tokenizer(text)
    
    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.clip_model.transformer.get_cast_dtype()

        x = self.clip_model.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def visual_prediction_forward_convnext(self, x):
        batch, num_query, channel = x.shape
        x = x.reshape(batch*num_query, channel, 1, 1) # fake 2D input
        x = self.clip_model.visual.trunk.head(x)
        x = self.clip_model.visual.head(x)
        return x.view(batch, num_query, x.shape[-1]) # B x num_queries x 640

    def get_text_classifier(self, text_list, device):
        self.eval()
        with torch.no_grad():
            # reference for templates: https://github.com/mlfoundations/open_clip/blob/91f6cce16b7bee90b3b5d38ca305b5b3b67cc200/src/training/imagenet_zeroshot_data.py
            text_tokens = self.tokenize_text(text_list)
            text_tokens = text_tokens.to(device)
            # we return un-normalized text feature.
            text_features = self.encode_text(text_tokens, normalize=False)
            return text_features
    
    def extract_features_convnext(self, x):
        out = {}
        x = self.clip_model.visual.trunk.stem(x)
        out['stem'] = x.contiguous() # os4
        for i in range(4):
            x = self.clip_model.visual.trunk.stages[i](x)
            out[f'res{i+2}'] = x.contiguous() # res 2 (os4), 3 (os8), 4 (os16), 5 (os32)
        
        x = self.clip_model.visual.trunk.norm_pre(x)
        out['clip_vis_dense'] = x.contiguous()
        return out
    
    def forward(self, x):
        self.eval()
        with torch.no_grad():
            return self.extract_features_convnext(x)   

class open_vocabulary:
    def __init__(self, 
                 clip_backbone, 
                 train_metadata,
                 test_metadata,
                 ):
        
        self.backbone = clip_backbone
        self.train_metadata = train_metadata
        self.test_metadata = test_metadata
        self.device = "cuda"

        _, self.train_num_templates, self.train_class_names = self.prepare_class_names_from_metadata(train_metadata, train_metadata)
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(test_metadata, train_metadata)

    def prepare_class_names_from_metadata(self, metadata, train_metadata):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',') # there can be multiple synonyms for single class
                res.append(x_)
            return res
        # get text classifier
        try:
            class_names = split_labels(metadata.stuff_classes) # it includes both thing and stuff
            train_class_names = split_labels(train_metadata.stuff_classes)
        except:
            # this could be for insseg, where only thing_classes are available
            class_names = split_labels(metadata.thing_classes)
            train_class_names = split_labels(train_metadata.thing_classes)
        train_class_names = {l for label in train_class_names for l in label}
        category_overlapping_list = []
        for test_class_names in class_names:
            is_overlapping = not set(train_class_names).isdisjoint(set(test_class_names))
            category_overlapping_list.append(is_overlapping)
        category_overlapping_mask = torch.tensor(
            category_overlapping_list, dtype=torch.long)
        
        def fill_all_templates_ensemble(x_=''):
            res = []
            for x in x_:
                for template in VILD_PROMPT:
                    res.append(template.format(x))
            return res, len(res) // len(VILD_PROMPT)
       
        num_templates = []
        templated_class_names = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num) # how many templates for current classes
        class_names = templated_class_names
        #print("text for classification:", class_names)
        return category_overlapping_mask, num_templates, class_names

    def set_metadata(self, metadata):
        self.test_metadata = metadata
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(metadata, self.train_metadata)
        self.test_text_classifier = None
        return
    
    def get_text_classifier(self):
        text_classifier = []
        # this is needed to avoid oom, which may happen when num of class is large
        bs = 128
        for idx in range(0, len(self.test_class_names), bs):
            text_classifier.append(self.backbone.get_text_classifier(self.test_class_names[idx:idx+bs], self.device).detach())
        text_classifier = torch.cat(text_classifier, dim=0)

        # average across templates and normalization.
        text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
        text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
        text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
        self.test_text_classifier = text_classifier
        return self.test_text_classifier, self.test_num_templates