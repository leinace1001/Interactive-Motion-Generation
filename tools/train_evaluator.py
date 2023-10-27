import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartModel
import torch.optim as optim
import itertools
from os.path import join as pjoin

from torch.utils.data._utils.collate import default_collate
from options.train_options import TrainCompOptions
from datasets import InterHumanDatasetEval, TextEncoderBiGRUCo, MotionEncoderBiGRUCo, MovementConvEncoder
from options.train_options import TrainCompOptions
from trainers import TextMotionMatchTrainer

def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    return default_collate(batch)

parser = TrainCompOptions()
opt = parser.parse()
opt.device = torch.device("cuda")
opt.data_root = './data/interhuman'
opt.motion_dir = pjoin(opt.data_root, 'joints')
opt.text_file = pjoin(opt.data_root, 'texts_en_revised.txt')
opt.joints_num = 22
opt.save_dir = './checkpoints/interhuman_eval'
opt.log_dir = './checkpoints/interhuman_eval'
opt.max_motion_length = 200
opt.dim_coemb_hidden = 512
opt.max_text_len = 40

movement_enc = MovementConvEncoder(22*3*2, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
text_enc = TextEncoderBiGRUCo(word_size=768,
                              hidden_size=opt.dim_text_hidden,
                              output_size=opt.dim_coemb_hidden,
                              device=opt.device)
motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                  hidden_size=opt.dim_pos_hidden,
                                  output_size=opt.dim_coemb_hidden,
                                  device=opt.device)

train_split_file = pjoin(opt.data_root, 'all.txt')
test_split_file = pjoin(opt.data_root, 'test.txt')
train_data = InterHumanDatasetEval(opt, train_split_file,argumentation=4)
train_loader = DataLoader(train_data,32,shuffle=True, collate_fn=collate_fn)
test_data = InterHumanDatasetEval(opt, test_split_file)
test_loader = DataLoader(test_data,32,shuffle=True, collate_fn=collate_fn)

trainer = TextMotionMatchTrainer(opt, text_enc, motion_enc, movement_enc)
trainer.train(train_loader, test_loader)