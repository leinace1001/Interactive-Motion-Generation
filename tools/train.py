import os
from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import TrainCompOptions
from utils.plot_script import *

from models import InteractionMotionTransformer
from trainers import InterhumanTrainer
from datasets import InterHumanDataset

import torch
import torch.distributed as dist
import dill


def build_models(opt, dim_pose):
    encoder = InteractionMotionTransformer(
        input_feats=dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff)
    return encoder


if __name__ == '__main__':
    parser = TrainCompOptions()
    opt = parser.parse()
    

    opt.device = torch.device("cuda")
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    

    

  
    opt.data_root = './data/interhuman'
    opt.motion_dir = pjoin(opt.data_root, 'joints')
    opt.text_file = pjoin(opt.data_root, 'texts_en_revised.txt')
    opt.joints_num = 22
    radius = 4
    fps = 20
    opt.max_motion_length = 200
    dim_pose = 22 * 3
    kinematic_chain = paramUtil.t2m_kinematic_chain
    

    opt.model_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    dim_word = 300
    


    train_split_file = pjoin(opt.data_root, 'train.txt')

    encoder = build_models(opt, dim_pose)
    
    encoder = encoder.cuda()

  
    train_dataset = InterHumanDataset(opt, train_split_file, argumentation=8)
    #with open('checkpoints/t2m/dataset/train_dataset.pkl', 'rb') as f:
      #train_dataset = dill.load(f)

    trainer = InterhumanTrainer(opt, encoder)
    trainer.train(train_dataset)