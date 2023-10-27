import torch
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
# import tensorflow as tf
from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
import codecs as cs
import numpy as np


def print_current_loss_decomp(start_time, niter_state, total_niters, losses, epoch=None, inner_iter=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    print('epoch: %03d inner_iter: %5d' % (epoch, inner_iter), end=" ")
    # now = time.time()
    message = '%s niter: %07d completed: %3d%%)'%(time_since(start_time, niter_state / total_niters), niter_state, niter_state / total_niters * 100)
    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    print(message)


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=3.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class Logger(object):
  def __init__(self, log_dir):
    # self.writer = tf.summary.create_file_writer(log_dir)
    pass

  def scalar_summary(self, tag, value, step):
    #   with self.writer.as_default():
    #       tf.summary.scalar(tag, value, step=step)
    #       self.writer.flush()
    pass

class TextMotionMatchTrainer(object):

    def __init__(self, args, text_encoder, motion_encoder, movement_encoder):
        self.opt = args
        self.text_encoder = text_encoder
        self.motion_encoder = motion_encoder
        self.movement_encoder = movement_encoder
        self.device = args.device

        if args.is_train:
            # self.motion_dis
            self.logger = Logger(args.log_dir)
            self.contrastive_loss = ContrastiveLoss()

    def resume(self, model_dir):
        checkpoints = torch.load(model_dir, map_location=self.device)
        self.text_encoder.load_state_dict(checkpoints['text_encoder'])
        self.motion_encoder.load_state_dict(checkpoints['motion_encoder'])
        self.movement_encoder.load_state_dict(checkpoints['movement_encoder'])

        self.opt_text_encoder.load_state_dict(checkpoints['opt_text_encoder'])
        self.opt_motion_encoder.load_state_dict(checkpoints['opt_motion_encoder'])
        return checkpoints['epoch'], checkpoints['iter']

    def save(self, model_dir, epoch, niter):
        state = {
            'text_encoder': self.text_encoder.state_dict(),
            'motion_encoder': self.motion_encoder.state_dict(),
            'movement_encoder': self.movement_encoder.state_dict(),

            'opt_text_encoder': self.opt_text_encoder.state_dict(),
            'opt_motion_encoder': self.opt_motion_encoder.state_dict(),
            'epoch': epoch,
            'iter': niter,
        }
        torch.save(state, model_dir)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def to(self, device):
        self.text_encoder.to(device)
        self.motion_encoder.to(device)
        self.movement_encoder.to(device)

    def train_mode(self):
        self.text_encoder.train()
        self.motion_encoder.train()
        self.movement_encoder.eval()

    def forward(self, batch_data):
        word_emb, caption, cap_lens, motions, m_lens = batch_data
        word_emb = word_emb.detach().to(self.device).float()
        
        motions = motions.detach().to(self.device).float()

        # Sort the length of motions in descending order, (length of text has been sorted)
        self.align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        # print(self.align_idx)
        # print(m_lens[self.align_idx])
        motions = motions[self.align_idx]
        m_lens = m_lens[self.align_idx]
        

        '''Movement Encoding'''
        movements = self.movement_encoder(motions).detach()
        m_lens = m_lens // self.opt.unit_length
        self.motion_embedding = self.motion_encoder(movements, m_lens)

        '''Text Encoding'''
        # time0 = time.time()
        # text_input = torch.cat([word_emb, pos_ohot], dim=-1)
        self.text_embedding = self.text_encoder(word_emb, cap_lens)
        self.text_embedding = self.text_embedding.clone()[self.align_idx]

    def backward(self):

        batch_size = self.text_embedding.shape[0]
        '''Positive pairs'''
        pos_labels = torch.zeros(batch_size).to(self.text_embedding.device)
        self.loss_pos = self.contrastive_loss(self.text_embedding, self.motion_embedding, pos_labels)

        '''Negative Pairs, shifting index'''
        neg_labels = torch.ones(batch_size).to(self.text_embedding.device)
        shift = np.random.randint(0, batch_size-1)
        new_idx = np.arange(shift, batch_size + shift) % batch_size
        self.mis_motion_embedding = self.motion_embedding.clone()[new_idx]
        self.loss_neg = self.contrastive_loss(self.text_embedding, self.mis_motion_embedding, neg_labels)
        self.loss = self.loss_pos + self.loss_neg

        loss_logs = OrderedDict({})
        loss_logs['loss'] = self.loss.item()
        loss_logs['loss_pos'] = self.loss_pos.item()
        loss_logs['loss_neg'] = self.loss_neg.item()
        return loss_logs


    def update(self):

        self.zero_grad([self.opt_motion_encoder, self.opt_text_encoder])
        loss_logs = self.backward()
        self.loss.backward()
        self.clip_norm([self.text_encoder, self.motion_encoder])
        self.step([self.opt_text_encoder, self.opt_motion_encoder])

        return loss_logs


    def train(self, train_dataloader, val_dataloader):
        self.to(self.device)

        self.opt_motion_encoder = optim.Adam(self.motion_encoder.parameters(), lr=self.opt.lr)
        self.opt_text_encoder = optim.Adam(self.text_encoder.parameters(), lr=self.opt.lr)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.save_dir, 'finest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.num_epochs * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        logs = OrderedDict()
        epoch = 0
        min_val_loss = np.inf
        while epoch < self.opt.num_epochs:
            # time0 = time.time()
            for i, batch_data in enumerate(train_dataloader):
                self.train_mode()

                self.forward(batch_data)
                # time3 = time.time()
                log_dict = self.update()
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v


                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss})
                    self.logger.scalar_summary('val_loss', val_loss, it)

                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss_decomp(start_time, it, total_iters, mean_loss, epoch, i)

                    if it % self.opt.save_latest == 0:
                        self.save(pjoin(self.opt.save_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.save_dir, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.save_dir, 'E%04d.tar' % (epoch)), epoch, it)
            
            print('Validation time:')

            loss_pos_pair = 0
            loss_neg_pair = 0
            val_loss = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)
                    self.backward()
                    loss_pos_pair += self.loss_pos.item()
                    loss_neg_pair += self.loss_neg.item()
                    val_loss += self.loss.item()
            
            loss_pos_pair /= len(val_dataloader) + 1
            loss_neg_pair /= len(val_dataloader) + 1
            val_loss /= len(val_dataloader) + 1
            print('Validation Loss: %.5f Positive Loss: %.5f Negative Loss: %.5f' %
                  (val_loss, loss_pos_pair, loss_neg_pair))

            if val_loss < min_val_loss:
                self.save(pjoin(self.opt.save_dir, 'finest.tar'), epoch, it)
                min_val_loss = val_loss

            if epoch % self.opt.eval_every_e == 0:
                pos_dist = F.pairwise_distance(self.text_embedding, self.motion_embedding)
                neg_dist = F.pairwise_distance(self.text_embedding, self.mis_motion_embedding)

                pos_str = ' '.join(['%.3f' % (pos_dist[i]) for i in range(pos_dist.shape[0])])
                neg_str = ' '.join(['%.3f' % (neg_dist[i]) for i in range(neg_dist.shape[0])])

                save_path = pjoin(self.opt.save_dir, 'E%03d_eval.txt' % (epoch))
                with cs.open(save_path, 'w') as f:
                    f.write('Positive Pairs Distance\n')
                    f.write(pos_str + '\n')
                    f.write('Negative Pairs Distance\n')
                    f.write(neg_str + '\n')