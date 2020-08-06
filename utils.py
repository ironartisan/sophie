import os
import math
import random
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy
import torch

from constants import *

def get_dset_path(dset_name, dset_type):
    return os.path.join('datasets', dset_name, dset_type)

def relative_to_abs(rel_traj, start_pos):
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)

def bce_loss(input, target):
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def gan_g_loss(scores_fake):
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)

def gan_d_loss(scores_real, scores_fake):
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake

def l2_loss(pred_traj, pred_traj_gt, mode='average'):
    seq_len, batch, _ = pred_traj.size()
    loss = (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / (seq_len * batch)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)

def displacement_error(pred_traj, pred_traj_gt, mode='sum'):
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss

def final_displacement_error(
    pred_pos, pred_pos_gt, mode='sum'
):
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)


def cal_ade(pred_traj_gt, pred_traj_fake, mode='sum'):
    ade = displacement_error(pred_traj_fake, pred_traj_gt, mode=mode)
    return ade


def cal_fde(pred_traj_gt, pred_traj_fake, mode='sum'):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode=mode)
    return fde


def process_normal(traj_data, normal_parm):
    traj_data[:, :, 0] = traj_data[:, :, 0] * normal_parm['lat_field'] + normal_parm['lat_min']

    traj_data[:, :, 1] = traj_data[:, :, 1] * normal_parm['long_field'] + normal_parm['long_min']

    traj_data[:, :, 2] = traj_data[:, :, 2] * normal_parm['alt_field'] + normal_parm['alt_min']

    return traj_data


def show_t(obs_traj, pred_traj_gt, predictions, param):
    obs_traj = process_normal(obs_traj, param)
    pred_traj_gt = process_normal(pred_traj_gt, param)

    for i in range(len(predictions)):
        predictions[i][0] = process_normal(predictions[i][0], param)
        predictions[i][0] = numpy.concatenate((numpy.expand_dims(obs_traj[-1, :, :], axis=0), predictions[i][0]),
                                              axis=0)

    pred_traj_gt = numpy.concatenate((numpy.expand_dims(obs_traj[-1, :, :], axis=0), pred_traj_gt), axis=0)

    fig = plt.figure()
    ax = Axes3D(fig)
    line_width = 1
    marker_size = 3

    ax.plot(obs_traj[:, 0, 0], obs_traj[:, 0, 1], obs_traj[:, 0, 2], color='r', label='input', linewidth=line_width)
    ax.plot(pred_traj_gt[:, 0, 0], pred_traj_gt[:, 0, 1], pred_traj_gt[:, 0, 2], color='g', label='groudtruth',
            linewidth=line_width, marker='D', markersize=marker_size)

    result_text = ''
    for i in range(len(predictions)):
        (prediction, ade, fde, label) = predictions[i]
        # print('%s len is %d' % (label, len(prediction)))
        ax.plot(prediction[:, 0, 0], prediction[:, 0, 1], prediction[:, 0, 2], color=CORLOR_LIST[i], label=label,
                linewidth=line_width, marker=MARKER_LIST[i], markersize=marker_size)
        result_text = result_text + '%s ade  = %f    %s fde = %f\n' % (label, ade, label, fde)

    plt.title(result_text, y=-0.2)

    ax.legend()
    plt.show()


def get_min_ade_fde(metrics, key='ade'):
    min_ade = 100000
    for i in range(len(metrics[key])):
        if metrics[key][i] < min_ade:
            min_ade = metrics['ade'][i]
            min_fde = metrics['fde'][i]

    return min_ade, min_fde