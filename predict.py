import gc
import os
import math
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from data import data_loader
from utils import get_dset_path
from utils import relative_to_abs
from utils import cal_ade,cal_fde
from utils import gan_g_loss, gan_d_loss, l2_loss, displacement_error, final_displacement_error
from models import TrajectoryGenerator, TrajectoryDiscriminator

from constants import *

class Infer(object):
    def __init__(self, use_cuda=1, dataset_name = 'asia',version='test'):
        self.use_cuda = use_cuda
        self.dataset_name = dataset_name
        self.version = version
        self.path = get_dset_path(self.dataset_name, self.version)
        self.generator = TrajectoryGenerator()


    def get_loader(self):
        return data_loader(self.path)[-1]

    def evaluate_helper(error):
        error = torch.stack(error, dim=1)
        error = torch.sum(error, dim=0)
        error = torch.min(error)
        return error


    def infer(self):
        with torch.no_grad():
            # print(len(self.loader))
            for batch in self.get_loader():
                if self.use_cuda == 1:
                    batch = [tensor.cuda() for tensor in batch]
                else:
                    batch = [tensor for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel) = batch

                # [8, 4, 2]
                pred_traj_fake_rel = self.generator(
                    obs_traj, obs_traj_rel
                )
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1, :, 0, :])

                obs_traj = obs_traj.cpu().numpy()
                pred_traj_fake = pred_traj_fake.cpu().numpy()
                pred_traj_gt = pred_traj_gt.cpu().numpy()

                return obs_traj, pred_traj_fake, pred_traj_gt

    def check_accuracy(self, generator):
        ade_outer, fde_outer = [], []
        total_traj = 0
        metrics = {}
        with torch.no_grad():
            for batch in self.get_loader():
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel) = batch

                ade, fde = [], []
                total_traj += pred_traj_gt.size(1)

                for _ in range(NUM_SAMPLES):
                    pred_traj_fake_rel = generator(obs_traj, obs_traj_rel)
                    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1, :, 0, :])
                    ade.append(displacement_error(pred_traj_fake, pred_traj_gt, mode='raw'))
                    fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'))

                # ade_sum = evaluate_helper(ade)
                # fde_sum = evaluate_helper(fde)

                # ade_outer.append(ade_sum.item())
                # fde_outer.append(fde_sum)
            ade = sum(ade_outer) / (total_traj * PRED_LEN)
            fde = sum(fde_outer) / (total_traj)
            metrics['ade'] = ade
            metrics['fde'] = fde

            return metrics

    def predict(self, obs_traj, pred_traj_gt, obs_traj_rel):
        pred_traj_fake_rel = self.generator(
            obs_traj, obs_traj_rel
        )

        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1, :, 0, :])

        ade = cal_ade(pred_traj_gt, pred_traj_fake)
        fde = cal_fde(pred_traj_gt, pred_traj_fake)

        pred_traj_fake = pred_traj_fake.cpu().detach().numpy()
        ade = ade.cpu().detach().numpy()
        fde = fde.cpu().detach().numpy()

        return pred_traj_fake, ade, fde


    def get_one_data(self):
        with torch.no_grad():
            # print(len(self.loader))
            for batch in self.get_loader():
                if self.use_cuda == 1:
                    batch = [tensor.cuda() for tensor in batch]
                else:
                    batch = [tensor for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel) = batch

                return obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel


    def load_model(self, path):
        # torch.load最后返回的是一个dict，里面包含了保存模型时的一些参数和模型
        checkpoint = torch.load(path)
        generator = self.get_generator(checkpoint)
        return generator

    def get_generator(self, checkpoint):
        self.generator.load_state_dict(checkpoint['g'])
        if self.use_cuda == 1:
            self.generator.cuda()
        self.generator.eval()

path = 'models/model.pt'
infer = Infer(use_cuda=1)
infer.load_model(path)
obs_traj, pred_traj_fake, pred_traj_gt = infer.infer()

print(obs_traj[:, : ,0])



# load_and_evaluate(generator, 'train')
# load_and_evaluate(generator, 'val')
# load_and_evaluate(generator, 'test')
