# Sphere Confidence Face (SCF)

This repository contains the PyTorch implementation of Sphere Confidence Face (SCF) proposed in the CVPR2021 paper: Shen Li, Xu Jianqing, Xiaqing Xu, Pengcheng Shen, Shaoxin Li, and Bryan Hooi. [Spherical Confidence Learning for Face Recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Spherical_Confidence_Learning_for_Face_Recognition_CVPR_2021_paper.pdf), IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2021] with [Appendices](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Li_Spherical_Confidence_Learning_CVPR_2021_supplemental.pdf).

<p align="center">
   <img src="scf_illustr.png" title="roc" width="850" />
</p>

## Empirical Results
|    IJB-B    | ResNet100 1e-5 | ResNet100 1e-4  |    IJB-C    | ResNet100 1e-5 | ResNet100 1e-4  |
| :------------: | :--------------: | :------: | :------------: | :--------------: | :------: |
| CosFace |       89.81       | 94.59  | CosFace |       93.86       | 95.95  |
| + PFE-G |       89.96       | 94.64  | + PFE-G |       94.09       | 96.04  |
| + PFE-v  |      N/A       |  N/A  |  + PFE-v  |      N/A       |  N/A  |
| + SCF-G  |       89.97       | 94.56  |  + SCF-G  |       94.15       | 96.02  |
| + **SCF**     |       91.02      | 94.95  |  + **SCF**     |       94.78     | 96.22  |
| ArcFace |       89.33       | 94.20  | ArcFace |       93.15       | 95.60  |
| + PFE-G |       89.55       | 94.30  |  + PFE-G |       92.95       | 95.32 |
| + PFE-v  |      N/A       |  N/A  |  + PFE-v  |      N/A       |  N/A  |
| + SCF-G  |       89.52       | 94.24  |  + SCF-G  |       93.85       | 95.33  |
| + **SCF**     |       90.68      | 94.74  |  + **SCF**     |       94.04      | 96.09  |

## Requirements
* python==3.6.0
* torch==1.6.0
* torchvision==0.7.0
* tensorboard==2.4.0

## Getting Started
### Training
Training consists of two separate steps:
1. Train ResNet100 imported from backbones.py as the deterministic backbone using spherical loss, e.g. [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) loss.
2. Train SCF based on the pretrained backbone by specifying the arguments including [GPU_IDS], [OUTPUT_DIR], [PATH_BACKBONE_CKPT] (the path of the pretrained backbone checkpoint) and [PATH_FC_CKPT] (the path of the pretrained fc-layer checkpoint) and then running the command:

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
