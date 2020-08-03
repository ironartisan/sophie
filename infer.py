import argparse
import os
import torch

from attrdict import AttrDict

from data import data_loader
from models import TrajectoryGenerator
from utils import relative_to_abs, get_dset_path

from utils import cal_ade, cal_fde

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='models', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--use_cuda', default=0, type=int)


class Infer(object):
    def __init__(self, use_cuda=0):
        self.use_cuda = use_cuda

    
    def get_accuracy(self):
        return self.metrics_train, self.metrics_val

    def load_model(self, path):
        # torch.load最后返回的是一个dict，里面包含了保存模型时的一些参数和模型
        checkpoint = torch.load(path, map_location='cpu')
        self.generator = self.get_generator(checkpoint)



    def infer(self):
        with torch.no_grad():
            # print(len(self.loader))
            for batch in self.loader:
                if self.use_cuda == 1:
                    batch = [tensor.cuda() for tensor in batch]
                else:
                    batch = [tensor for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = batch

                # [8, 4, 2]
                pred_traj_fake_rel = self.generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )

                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )

                obs_traj = obs_traj.cpu().numpy()
                pred_traj_fake = pred_traj_fake.cpu().numpy()
                pred_traj_gt = pred_traj_gt.cpu().numpy()

                return obs_traj, pred_traj_fake, pred_traj_gt

    def load_model(self, path):
        # torch.load最后返回的是一个dict，里面包含了保存模型时的一些参数和模型
        checkpoint = torch.load(path, map_location='cpu')
        self.generator = self.get_generator(checkpoint)
        # AttrDict是根据参数中的dict内容生成一个更加方便访问的dict实例
        self.args = AttrDict(checkpoint['args'])
        train_path = get_dset_path(self.args.dataset_name, "train")
        test_path = get_dset_path(self.args.dataset_name, "test")
        self.args.batch_size = 1
        _, self.loader = data_loader(self.args, train_path)
        _, self.test_loader = data_loader(self.args, test_path)

        self.metrics_val = checkpoint['metrics_val']
        self.metrics_train = checkpoint['metrics_train']

    def predict(self, obs_traj, pred_traj_gt, obs_traj_rel, seq_start_end):
        pred_traj_fake_rel = self.generator(
            obs_traj, obs_traj_rel, seq_start_end
        )

        pred_traj_fake = relative_to_abs(
            pred_traj_fake_rel, obs_traj[-1]
        )

        ade = cal_ade(pred_traj_gt, pred_traj_fake)
        fde = cal_fde(pred_traj_gt, pred_traj_fake)

        pred_traj_fake = pred_traj_fake.cpu().detach().numpy()
        ade = ade.cpu().detach().numpy()
        fde = fde.cpu().detach().numpy()

        return pred_traj_fake, ade, fde


    def get_one_data(self):
        with torch.no_grad():
            # print(len(self.loader))
            for batch in self.loader:
                if self.use_cuda == 1:
                    batch = [tensor.cuda() for tensor in batch]
                else:
                    batch = [tensor for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = batch

                return obs_traj, pred_traj_gt, obs_traj_rel, seq_start_end


    def get_loader(self, loader_type='test'):
        if loader_type == 'test':
            return self.test_loader
        else:
            return self.loader


    def check_accuracy(self, loader_type='test', loader=None, limit=False):
        if loader_type == 'spec':
            if loader == None:
                raise Exception('loader is not defined')
            loader = loader
        elif loader_type == 'test':
            loader = self.test_loader
        else:
            loader = self.loader

        args = self.args
        metrics = {}
        disp_error, f_disp_error = [], []
        total_traj = 0
        loss_mask_sum = 0
        with torch.no_grad():
            for batch in loader:
                if args.use_gpu == 1:
                    batch = [tensor.cuda() for tensor in batch]
                else:
                    batch = [tensor for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,non_linear_ped, loss_mask, seq_start_end) = batch
                linear_ped = 1 - non_linear_ped
                loss_mask = loss_mask[:, args.obs_len:]
    
                pred_traj_fake_rel = self.generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
    
                ade = cal_ade(pred_traj_gt, pred_traj_fake)
    
                fde = cal_fde(pred_traj_gt, pred_traj_fake)
    
                disp_error.append(ade.item())
                f_disp_error.append(fde.item())
    
                total_traj += pred_traj_gt.size(1)
                if limit and total_traj >= args.num_samples_check:
                    break
    
        metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
        metrics['fde'] = sum(f_disp_error) / total_traj
    
        return metrics


    def get_generator(self, checkpoint):
        generator = TrajectoryGenerator()
        generator.load_state_dict(checkpoint['g'])
        if self.use_cuda == 1:
            generator.cuda()
        generator.eval()
        return generator


if __name__ == '__main__':
    """
    1. load model
        1.1 加载genenrator,self.generator
    2. infer 获取坐标
        2.1   self.loader:

    """
    args = parser.parse_args()
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]
    path = paths[0]

    print(path)
    infer = Infer(use_cuda=1)
    infer.load_model(path)
    obs_traj, pred_traj_fake, pred_traj_gt = infer.infer()
    print(obs_traj)
    print(pred_traj_fake)
    print(pred_traj_gt)





