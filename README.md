# Sphere Confidence Face (SCF)

This repository contains the PyTorch implementation of Sphere Confidence Face (SCF) proposed in the CVPR2021 paper: [Spherical Confidence Learning for Face Recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Spherical_Confidence_Learning_for_Face_Recognition_CVPR_2021_paper.pdf) with [Appendices](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Li_Spherical_Confidence_Learning_CVPR_2021_supplemental.pdf).

An emerging line of research has found that spherical spaces better match the underlying geometry of facial images, as evidenced by the state-of-the-art facial recognition methods which benefit empirically from spherical representations.Yet, these approaches rely on deterministic embeddings and hence suffer from the feature ambiguity dilemma, whereby ambiguous or noisy images are mapped into poorly learned regions of representation space, leading to inaccuracies. PFE is the first attempt to address this dilemma. However, we theoretically and empirically identify two main failures of PFE when it is applied to spherical deterministic embeddings aforementioned. To address these issues, in this paper, we propose a novel framework for face confidence learning in spherical space. Mathematically, we extend the von Mises Fisher density to its r-radius counterpart and derive a new optimization objective in closed form. We theoretically show that the proposed framework allows for better interpretability, leading to principled feature comparison and pooling. Extensive experimental results on multiple challenging benchmarks confirm our hypothesis and theory, and showcase the superior performance of our framework against prior probabilistic methods and conventional spherical deterministic embeddings both in risk-controlled recognition tasks and in face verification and identification tasks.

<p align="center">
   <img src="scf_illustr.pdf" title="roc" width="500" />
</p>

## Main Results
...

## Requirements
* python==3.6.0
* torch==1.6.0
* torchvision==0.7.0
* tensorboard==2.4.0

## Getting Started
### Training
1. Train ResNet100 imported from backbones.py as the deterministic backbone using spherical loss, e.g. [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) loss.
2. Train SCF based on the pretrained backbone by specifying [GPU_IDS], [OUTPUT_DIR], [PATH_BACKBONE_CKPT] (the path of the pretrained backbone checkpoint) and [PATH_FC_CKPT] (the path of the pretrained fc-layer checkpoint) and running:

``` bash
python train.py \
    --dataset "ms1m" \
    --seed 777 \
    --gpu_ids [GPU_IDS] \
    --batch_size 1024 \
    --output_dir [OUTPUT_DIR] \
    --saved_bkb [PATH_BACKBONE_CKPT] \
    --saved_fc [PATH_FC_CKPT] \
    --num_workers 8 \
    --epochs 30 \
    --lr 3e-5 \
    --lr_scheduler "StepLR" \
    --step_size 2 \
    --gamma 0.5 \
    --convf_dim 25088 \
    --z_dim 512 \
    --radius 64 \
    --max_grad_clip 0 \
    --max_grad_norm 0 \
    --tensorboard
```

### Test

IJB benchmark: use $\kappa$ as confidence score for each face image to aggregate representations as in Eqn (14). Refer to [the standard IJB benchmark](https://github.com/deepinsight/insightface/tree/master/recognition/_evaluation_/ijb) for implementation.

1v1 verification benchmark: use Eqn (13) as the similarity score.

## Other Implementations
SCF in TFace: [SCF](https://github.com/Tencent/TFace/tree/master/tasks/scf)

## Citation
```
@inproceedings{li2021spherical,
  title={Spherical Confidence Learning for Face Recognition},
  author={Li, Shen and Xu, Jianqing and Xu, Xiaqing and Shen, Pengcheng and Li, Shaoxin and Hooi, Bryan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15629--15637},
  year={2021}
}
```
