import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from transformers import BartTokenizer, BartModel

def rotation(angle, motion):
    rotation_matrix = np.array([[np.cos(angle), 0, np.sin(angle)],
                                [0, 1, 0],
                                [-np.sin(angle), 0, np.cos(angle)]])
    t, j, d = motion.shape
    motion = rotation_matrix @ (motion.reshape((t,j,d,1)))
    motion = motion.reshape((t,j,d))
    return motion
  
class InterHumanDataset(data.Dataset):
  def __init__(self, opt, split_file, argumentation=1, times=1, eval_mode=False):
    self.opt = opt
    self.data={"motion":[], "text":[],"length":[]}
    with open(split_file, 'r') as f:
      file_list = f.readlines()
    with open(opt.text_file, 'r') as f:
      text = f.readlines()
    self.argumentation=argumentation
    for i,f in enumerate(file_list):
      

        
        motion = np.load(pjoin(opt.motion_dir, f[:-1]))
        person1 = motion["person1"]
        person2 = motion["person2"]

        
        if not np.isfinite(person1).all():
          continue
        if not np.isfinite(person2).all():
          continue
        length1, j, d = person1.shape
        length2 = person2.shape[0]
        length = min(length1,length2,opt.max_motion_length)
        person1-=np.array([[[0.8586, 0, 0.2866]]])
        person2-=np.array([[[0.8586, 0, 0.2866]]])
        person1 = person1.reshape(length1, -1)[:length]
        person2 = person2.reshape(length2, -1)[:length]
        padding = np.zeros((opt.max_motion_length - length, j*d))
        person1 = np.concatenate([person1, padding],axis=0)
        person2 = np.concatenate([person2, padding],axis=0)
        motion = np.concatenate([person1, person2], axis=1)
        idx = int(f[:-5])
        f_text = text[idx]
        if i%10==0:
          print(".",end='')
        if i%500==0:
          print()
        self.data["motion"].append(motion)
        self.data["text"].append(f_text)
        self.data["length"].append(length)




  def __getitem__(self, index):
    if self.argumentation == 1:
        return self.data["motion"][index], self.data["text"][index], self.data["length"][index]
    t = index%self.argumentation
    index = index//self.argumentation
    motion = self.data["motion"][index].reshape(self.opt.max_motion_length,-1,3)
    text = self.data["text"][index]
    if t%2 ==0:
      motion[:,:,0]= - motion[:,:,0]
      text = text.replace("left", "rig_ht")
      text = text.replace("right", "le_ft")
      text = text.replace("_", "")
    if t%4 == 0 or t%4 == 3:
      motion[:,:22,:], motion[:,22:,:] = motion[:,22:,:], motion[:,:22,:]
    
    angle = np.random.rand()*2*np.pi
    motion = rotation(angle, motion)
    motion = motion.reshape(self.opt.max_motion_length,-1)

    return motion, text, self.data["length"][index]

  def __len__(self):
    return len(self.data["motion"])*self.argumentation


class InterHumanDatasetEval(data.Dataset):
  def __init__(self, opt, split_file, times=1, argumentation =1, eval_mode=False):
    self.opt = opt
    self.data=[]
    self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    self.model = BartModel.from_pretrained('facebook/bart-base')
    with open(split_file, 'r') as f:
      file_list = f.readlines()
    with open(opt.text_file, 'r') as f:
      text = f.readlines()
    self.argumentation = argumentation
    for f in file_list:
      

        
        motion = np.load(pjoin(opt.motion_dir, f[:-1]))
        person1 = motion["person1"]
        person2 = motion["person2"]


        
        if not np.isfinite(person1).all():
          continue
        if not np.isfinite(person2).all():
          continue
        length1, j, d = person1.shape
        length2 = person2.shape[0]
        length = min(length1,length2,opt.max_motion_length)
        person1-=np.array([[[0.8586, 0, 0.2866]]])
        person2-=np.array([[[0.8586, 0, 0.2866]]])
        person1 = person1.reshape(length1, -1)[:length]
        person2 = person2.reshape(length2, -1)[:length]
        padding = np.zeros((opt.max_motion_length - length, j*d))
        person1 = np.concatenate([person1, padding],axis=0)
        person2 = np.concatenate([person2, padding],axis=0)
        motion = np.concatenate([person1, person2], axis=1)
        idx = int(f[:-5])
        f_text = text[idx]
        tokens = self.tokenizer(f_text,return_tensors="pt")['input_ids']
        sent_len = min(tokens.shape[1],self.opt.max_text_len+2)
        cur_data = {"motion":motion,"text":f_text,"length":length,"text_len":sent_len}
        self.data.append(cur_data)
    
    

  def __getitem__(self, index):
      t = index%self.argumentation
      index = index//self.argumentation
      item = self.data[index]
      tokens = self.tokenizer(item["text"], return_tensors="pt")['input_ids']
      if tokens.shape[1] < self.opt.max_text_len+2:
                # pad with "unk"
          sent_len = tokens.shape[1]
          paddings = torch.ones(1,self.opt.max_text_len + 2 - sent_len).long()
          tokens = torch.cat([tokens,paddings],dim=1)
      else:
                # crop
          tokens = tokens[:,:self.opt.max_text_len+1]
          paddings = 2*torch.ones(1,1).long()
          tokens = torch.cat([tokens,paddings],dim=1)
          sent_len = self.opt.max_text_len+2
      word_emb = self.model.encoder(tokens).last_hidden_state[0]
      motion = item["motion"].reshape(self.opt.max_motion_length,-1,3)
      
      if t%2==0:
        motion[:,:,[0,2]]= - motion[:,:,[0,2]]
      if t%4 == 0 or t%4 == 3:
        motion[:,:22,:], motion[:,22:,:] = motion[:,22:,:], motion[:,:22,:]
      angle = 0
      if self.argumentation == 1:
        amgle = 0
      else:
        angle = np.random.rand()*2*np.pi
      
      motion = rotation(angle, motion)
      motion = motion.reshape(self.opt.max_motion_length,-1)
      return word_emb, item["text"], sent_len, motion, item["length"]

  def __len__(self):
    return len(self.data)*self.argumentation



