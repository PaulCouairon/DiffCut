<div align="center">
<h1>
DiffCut: Catalyzing Zero-Shot Semantic Segmentation with Diffusion Features and Recursive Normalized Cut
<br>
</h1><br>

<p style="font-size: 20px"> NeurIPS 2024</p>

<p></p>

<p></p>

<h2>
<a href="https://scholar.google.fr/citations?user=yQRnP7YAAAAJ&hl=fr">Paul Couairon</a>&ensp;
<a href="https://scholar.google.com/citations?user=lhp9mRgAAAAJ&hl=en">Mustafa Shukor</a>&ensp;

<a href="https://fr.linkedin.com/in/jean-emmanuel-haugeard">Jean-Emmanuel Haugeard</a>&ensp;
<a href="https://cord.isir.upmc.fr">Matthieu Cord</a>&ensp;
<a href="https://thome.isir.upmc.fr">Nicolas Thome</a>&ensp;
</h2>


<p></p>
<a href="https://arxiv.org/abs/2406.02842v1"><img
src="https://img.shields.io/badge/arXiv-DiffCut-b31b1b.svg" height=25em></a>
<a href="https://diffcut-segmentation.github.io"><img 
src="https://img.shields.io/static/v1?label=Project&message=Website&color=green" height=25em></a>


![main_figure.png](./assets/main_figure.png)

</div>

## Environment
```
conda create -n diffcut python=3.10
conda activate diffcut
pip install -r requirements.txt
```

For evaluation, install [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Demo
Try __DiffCut__ by running the notebook ``diffcut.ipynb``

Visualize the __semantic coherence__ of vision encoders with ``semantic_coherence.ipynb``




## Evaluation

### Datasets Preparation
In the paper, we evaluate DiffCut on 6 benchmarks: PASCAL VOC (20 classes + background), PASCAL Context (59 classes + background), COCO-Object (80 classes + background), COCO-Stuff (27 classes), Cityscapes (27 classes) and ADE20k (150 classes). See [Preparing Datasets for DiffCut](datasets/README.md).

### Run Evaluation
```
python eval_diffcut.py --dataset_name Cityscapes --tau 0.5 --alpha 10 --refinement
```



## Citation
```
@misc{couairon2024zeroshot,
    title={Zero-Shot Image Segmentation via Recursive Normalized Cut on Diffusion Features},
    author={Paul Couairon and Mustafa Shukor and Jean-Emmanuel Haugeard and Matthieu Cord and Nicolas Thome},
    year={2024},
    eprint={2406.02842},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledgements
This repo relies on the following projects:

[Diffuse, Attend, and Segment: Unsupervised Zero-Shot Segmentation using Stable Diffusion](https://github.com/google/diffseg)

[Open Vocabulary Semantic Segmentation with Patch Aligned Contrastive Learning](https://arxiv.org/abs/2212.04994)

[Convolutions Die Hard: Open-Vocabulary Segmentation with Single Frozen Convolutional CLIP](https://github.com/bytedance/fc-clip)

[Cut and Learn for Unsupervised Image & Video Object Detection and Instance Segmentation](https://github.com/facebookresearch/CutLER)


