import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision

from datasets import get_MS1M
from backbones import ResNet100

from uncer_net import UncertaintyNet
from kl_delta_vmf import *

from tensorboardX import SummaryWriter

import pdb

import argparse
import os
import os.path as osp
import json
import shutil
import random


def check_manual_seed(seed):
    random.seed(seed)
    
    torch.manual_seed(seed) # for cpu
    torch.cuda.manual_seed(seed) # for single GPU
    torch.cuda.manual_seed_all(seed) # for all GPUs

    print("Using seed: {seed}".format(seed=seed))


def check_dataset(dataset):
    if dataset == "ms1m":
        ms1m = get_MS1M()
        input_size, num_classes, train_dataset, test_dataset = ms1m

    return input_size, num_classes, train_dataset, test_dataset


def get_learning_rate(optimizer):
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list += [param_group['lr']]
    return lr_list


def main(
    gpu_ids,
    dataset,
    dataroot,
    seed,
    batch_size,
    eval_batch_size,
    epochs,
    num_workers,
    output_dir,
    saved_bkb,
    saved_fc,
    cuda,
    tensorboard,
    lr,
    lr_scheduler,
    step_size,
    gamma,
    convf_dim,
    z_dim,
    radius,
    max_grad_clip,
    max_grad_norm,
    finetune_bkb):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    gpus = gpu_ids.split(',')
    nGPUs = len(gpus)

    device = "cpu" if (not torch.cuda.is_available() or not cuda) else "cuda"
    check_manual_seed(seed)

    ds = check_dataset(dataset)
    image_shape, num_classes, train_dataset, test_dataset = ds

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    backbone = ResNet100(image_shape).to(device)
    bkb_ckpt = torch.load(saved_bkb, map_location=device)
    backbone.load_state_dict(bkb_ckpt) ##########################
    if not finetune_bkb:
        backbone.eval()

    # The original weights were UNNORMALIZED!
    fc_ckpt = torch.load(saved_fc, map_location=device)
    fc_ckpt = fc_ckpt['arc_face_state_dict']['weight']
    fc_norm = torch.norm(fc_ckpt, dim=1, keepdim=True) #[N, 512]
    fc_ckpt = fc_ckpt / fc_norm * radius # $ w_c \in rS^{d-1} $
    # of shape [85742, 512]

    #dummy_x = torch.zeros([128, 3, 112, 112], dtype=torch.float32).to(device)
    #dummy_fc_feat, dummy_conv_feat = backbone(dummy_x)
    # [128, 512], [128, 25088]

    uncer_net = UncertaintyNet(convf_dim, z_dim).to(device)

    if nGPUs > 1:
        backbone = torch.nn.DataParallel(backbone, range(nGPUs))
        uncer_net = torch.nn.DataParallel(uncer_net, range(nGPUs))

    kl = KLDiracVMF(z_dim, radius)

    vars2opt = [{"params": backbone.module.bn2.parameters(), "lr": lr*1}, {"params": backbone.module.dropout.parameters(), "lr": lr*1}, {"params": backbone.module.fc.parameters(), "lr": lr*1}, {"params": backbone.module.features.parameters(), "lr": lr*1}] if finetune_bkb else []
    vars2opt.append({"params": uncer_net.parameters()})
    optimizer = optim.Adam(vars2opt, lr=lr, weight_decay=5e-4)

    if lr_scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20000, verbose=True) ## should be put into the inner loop
    
    runs_folder = osp.join(output_dir, "runs")
    if not osp.exists(runs_folder): os.makedirs(runs_folder)
    ckpts_folder = osp.join(output_dir, "ckpts")
    if not osp.exists(ckpts_folder):    os.makedirs(ckpts_folder)

    if tensorboard:
        writer = SummaryWriter(logdir=runs_folder)

    print("Start training...")
    epoch = 0
    uncer_net.train()

    for epoch in range(epochs):
        lr_list = get_learning_rate(optimizer)
        print("Learninig rate used: ", lr_list)

        for itr, sample in enumerate(train_loader):
            global_step = itr + epoch * len(train_loader)

            optimizer.zero_grad()
            
            face = sample["face"].to(device)
            label = sample["label"].to(device).long()

            unnormed_mu, convf = backbone(face)
            mu_norm = torch.norm(unnormed_mu, dim=1, keepdim=True) # --> [2B,]
            mu = unnormed_mu / mu_norm # mu \in S^{d-1}

            log_kappa = uncer_net(convf)
            kappa = torch.exp(log_kappa)
            kappa_mean = kappa.mean()
            
            wc = fc_ckpt[label, :]
            losses, l1, l2, l3 = kl(mu, kappa, wc)

            total_loss = losses.mean()
            neg_kappa_times_cos_theta = l1.mean()
            neg_dim_scalar_times_log_kappa = l2.mean()
            log_iv_kappa = l3.mean()

            total_loss.backward()

            if max_grad_clip > 0:
                torch.nn.utils.clip_grad_value_(uncer_net.parameters(), max_grad_clip)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(uncer_net.parameters(), max_grad_norm)
            
            optimizer.step()

            if tensorboard:
                writer.add_scalar("kappa", kappa_mean.item(), global_step)
                writer.add_scalar("total_loss", total_loss.item(), global_step)
                writer.add_scalar("neg_kappa_times_cos_theta", neg_kappa_times_cos_theta.item(), global_step)
                writer.add_scalar("neg_dim_scalar_times_log_kappa", neg_dim_scalar_times_log_kappa.item(), global_step)
                writer.add_scalar("log_iv_kappa", log_iv_kappa.item(), global_step)

            print("Epoch {} Iter {}: Loss {}".format(epoch+1, itr+1, total_loss.item()))

            if lr_scheduler == "ReduceLROnPlateau":
                scheduler.step(total_loss.item())

        # End of each epoch
        print("Saving models...")
        dict2save = {'epoch': epoch,
                     'global_step': global_step,
                     'uncer_net_state_dict': uncer_net.state_dict() if nGPUs == 1 else uncer_net.module.state_dict(),
                     'opt_state': optimizer.state_dict()}
        if finetune_bkb:
            dict2save['backbone_state_dict'] = backbone.state_dict() if nGPUs == 1 else backbone.module.state_dict()

        torch.save(dict2save, osp.join(ckpts_folder, "ckpt_e{}.pt".format(epoch+1)))

        #print("Evaluating test set...")
        #model.eval()
        ######
        
        ######
        #model.train()

        if lr_scheduler == "StepLR":
            scheduler.step()
    
    print("Training completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--dataroot", type=str, default="")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu_ids", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--saved_bkb", type=str)
    parser.add_argument("--saved_fc", type=str)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--no_cuda", action="store_false", dest="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_scheduler", type=str)
    parser.add_argument("--step_size", type=int)
    parser.add_argument("--gamma", type=float)

    parser.add_argument("--convf_dim", type=int, default=25088) #### should be consistent with dummy data size
    parser.add_argument("--z_dim", type=int, default=512) ########## should be consistent with dummy data size
    parser.add_argument("--radius", type=float)

    parser.add_argument("--max_grad_clip", type=float)
    parser.add_argument("--max_grad_norm", type=float)

    parser.add_argument("--tensorboard", action="store_true", dest="tensorboard")
    parser.add_argument("--finetune_bkb", action="store_true", dest="finetune_bkb")

    args = parser.parse_args()
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    kwargs = vars(args)

    with open(osp.join(args.output_dir, "hparams.json"), 'w') as fp:
        json.dump(kwargs, fp, sort_keys=True, indent=4)

    #del kwargs["gpu_id"]

    main(**kwargs)

    print("DONE.")
