<div align="center">
<h1>
Zero-Shot Image Segmentation via Recursive Normalized Cut on Diffusion Features </h1><br>
<p></p>

<p></p>

<h2>
<a href="">Paul Couairon</a>&ensp;
<a href="">Mustafa Shukor</a>&ensp;

<a href="">Jean-Emmanuel Haugeard</a>&ensp;
<a href="">Matthieu Cord</a>&ensp;
<a href="">Nicolas Thome</a>&ensp;
</h2>


<p></p>
<a href=""><img
src="https://img.shields.io/badge/arXiv-DiffCut-b31b1b.svg" height=25em></a>
<a href="https://diffcut.github.io"><img 
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
```

## Acknowledgements
This repo relies on the following projects:

[Diffuse, Attend, and Segment: Unsupervised Zero-Shot Segmentation using Stable Diffusion](https://github.com/google/diffseg)

[Open Vocabulary Semantic Segmentation with Patch Aligned Contrastive Learning](https://arxiv.org/abs/2212.04994)

[Convolutions Die Hard: Open-Vocabulary Segmentation with Single Frozen Convolutional CLIP](https://github.com/bytedance/fc-clip)

[Cut and Learn for Unsupervised Image & Video Object Detection and Instance Segmentation](https://github.com/facebookresearch/CutLER)


