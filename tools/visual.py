import torch
import numpy as np
import argparse
from os.path import join as pjoin

from trainers import InterhumanTrainer
from models import InteractionMotionTransformer
from utils.paramUtil import t2m_kinematic_chain
from utils.get_opt import get_opt
from utils.plot_script import *

def build_models(opt, dim_pose):
    encoder = InteractionMotionTransformer(
        input_feats=dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff)
    return encoder

def smooth(motion):
  for t in range(motion.shape[0]-2):
    motion[t+1,:,:] = motion[t+1,:,:]/2 + motion[t,:,:]/4 + motion[t+2,:,:]/4
  

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--caption", default = "Two people are dancing togather.")
parser.add_argument("--root", default = "./checkpoints/interhuman/interhuman_1")
parser.add_argument("save_path", default="./checkpoints/interhuman/example.mp4")

caption = ["Two people are dancing togather."]
motion_length = 100
device = torch.device('cpu')
root = "./checkpoints/interhuman/interhuman_1"
opt = get_opt(pjoin(root,"opt.txt"), device)
encoder = build_models(opt, 22*3).to(device)
#opt.diffusion_steps = 1000
trainer = InterhumanTrainer(opt, encoder)
trainer.load(pjoin(root,"latest.tar"))
trainer.eval_mode()
trainer.to(opt.device)

with torch.no_grad():
  m_len = torch.LongTensor([motion_length]).to(device)
  pred_motions = trainer.generate(caption, m_len, opt.dim_pose)
  motions = pred_motions[0]

  motions = motions.cpu().numpy()
 
  motions = motions.reshape(-1,22*2,3)
  smooth(motions)
save_path = "./checkpoints/interhuman/example.mp4"
plot_3d_motion(save_path, t2m_kinematic_chain, motions[:,:22,:], motions[:,22:,:], caption[0], figsize=(10, 10), fps=20, radius=4)