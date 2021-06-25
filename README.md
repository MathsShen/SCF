# Sphere Confidence Face (SCF)

This repository contains the PyTorch implementation of Sphere Confidence Face (SCF) proposed in the CVPR2021 paper: [Spherical Confidence Learning for Face Recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Spherical_Confidence_Learning_for_Face_Recognition_CVPR_2021_paper.pdf) with [Appendices](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Li_Spherical_Confidence_Learning_CVPR_2021_supplemental.pdf).

## Main Results
...

## Requirements
* python==3.6.0
* torch==1.6.0
* torchvision==0.7.0
* tensorboard==2.4.0

## Getting Started
### Training
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
