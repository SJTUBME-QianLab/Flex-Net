from __future__ import print_function
import os
import time
import yaml
import pickle
from shutil import copytree, ignore_patterns
import argparse
import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from net import model
from tools import regularization
from tools import data_load_nat as loader
from tools.utils import seed_torch, str2bool, eval_metric, get_auc


def get_parser():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./../results/')
    parser.add_argument('--data_root', type=str, default='./../data/')
    parser.add_argument('--data_name', type=str, default='nat1')
    parser.add_argument('--rand', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--fold', type=int, default=0, help='fold index, [0,1,2,3,4]')

    parser.add_argument('--exp_name', type=str, default='', help='Name of the experiment')
    parser.add_argument('--label_num', type=int, default=3, help='Number of labels')
    parser.add_argument('--test_N_way', type=int, default=5, help='Number of classes for each classification run')
    parser.add_argument('--train_N_way', type=int, default=5, help='Number of classes for each training comparison')
    parser.add_argument('--test_N_shots', type=int, default=1, help='Number of shots in test')
    parser.add_argument('--train_N_shots', type=int, default=1, help='Number of shots when training')
    parser.add_argument('--lambda1', type=float, default=0.0,  help='weight for L1-norm')
    parser.add_argument('--reg_pos', type=str, default='Wcom_last', help='position of regularization')

    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--dec_lr', type=int, default=1000, help='Decreasing the learning rate every x iterations')
    parser.add_argument('--batch_size', type=int, default=20, help='Size of batch')
    parser.add_argument('--batch_size_test', type=int, default=10, help='Size of batch')
    parser.add_argument('--test_iter', type=int, default=1, help='Number of repeats for test')
    parser.add_argument('--iterations', type=int, default=5000, help='Number of epochs for train')
    parser.add_argument('--log_interval', type=int, default=5,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_interval', type=int, default=3000,
                        help='how many batches between each model saving')
    parser.add_argument('--test_interval', type=int, default=20,
                        help='how many batches between each test')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--active_random', type=int, default=0)

    return parser.parse_args()


class Processor:
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.load_data()
        self.train_writer = SummaryWriter(os.path.join(self.work_dir, 'train'), 'train')
        self.test_writer = SummaryWriter(os.path.join(self.work_dir, 'test'), 'test')
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.best_acc = 0

    def save_arg(self):
        assert self.arg.label_num == 3
        assert self.arg.batch_size == self.arg.batch_size_test
        assert self.arg.train_N_way == self.arg.test_N_way
        self.data_path = os.path.join(self.arg.data_root, 'nat', self.arg.data_name + '_0.8_').rstrip('/')
        losses = 'N{:d}b{:d}dec{:d}lr{:g}lam{:g}s{:d}'.format(
            self.arg.train_N_way, self.arg.batch_size, self.arg.dec_lr, self.arg.lr, self.arg.lambda1, self.arg.seed)
        self.arg.exp_name = '__'.join([losses, self.arg.exp_name])
        self.model_name = 't{}__{}'.format(time.strftime('%Y%m%d%H%M%S'), self.arg.exp_name)
        self.work_dir = os.path.join(self.arg.save_dir, 'Ours', self.arg.data_name, f'rand{self.arg.rand}',
                                     f'fold{self.arg.fold}', self.model_name)
        os.makedirs(self.work_dir, exist_ok=True)
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.print_log(',\t'.join([self.arg.exp_name]))

        # copy all files
        pwd = os.path.dirname(os.path.realpath(__file__))
        copytree(pwd, os.path.join(self.work_dir, 'code'), symlinks=False, ignore=ignore_patterns('__pycache__'))
        arg_dict = vars(self.arg)
        with open(os.path.join(self.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(arg_dict, f)

    def load_data(self):
        self.data_loader = dict()
        self.data_loader['train'] = loader.Load(
            partition='train', data_path=self.data_path, seed=self.arg.rand, fold=self.arg.fold
        )
        self.data_loader['test'] = loader.Load(
            partition='test', data_path=self.data_path, seed=self.arg.rand, fold=self.arg.fold
        )
        assert self.data_loader['train'].label_num == self.arg.label_num

    def load_model(self):
        self.model = model.MetricNN(
            emb_size=self.data_loader['train'].feature_num, label_num=self.arg.label_num
        ).cuda(self.output_device)

        if self.arg.lambda1 > 0:
            self.reg_loss = regularization.Regularization(
                model=self.model, weight_decay=self.arg.lambda1, position=self.arg.reg_pos, p=1,
            ).cuda(self.output_device)
            self.print_log(self.reg_loss.weight_names, print_time=False)
        else:
            self.print_log("no regularization", print_time=False)

    def load_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.arg.lr)

    def start(self):
        self.counter = 0
        self.total_loss = 0
        self.test_acc = 0
        df_test, df_train = [], []
        pred_test, pred_train = [], []
        for epoch in range(self.arg.iterations):  # 100000
            self.train(epoch)

            if (epoch + 1) % self.arg.test_interval == 0:
                with torch.no_grad():
                    # dfi_train, true_tr, pred_tr = self.test(epoch, 'train')
                    # df_train.append(dfi_train)
                    # pred_train.extend(pred_tr.T)

                    dfi_test, true_te, pred_te = self.test(epoch, 'test')
                    df_test.append(dfi_test)
                    pred_test.extend(pred_te.T)

        # save
        all_pred_te = pd.DataFrame(np.hstack((np.array([true_te]).T, np.array(pred_test).T)))
        all_pred_te.to_csv(os.path.join(self.work_dir, 'test_predictions.csv'), index=False)
        df_test = pd.concat(df_test, axis=0)
        df_test.to_csv(os.path.join(self.work_dir, 'test_indicators.csv'), index=False)

    def train(self, epoch):
        self.model.train()
        loader = self.data_loader['train']
        data = loader.get_task_batch_eq(
            batch_size=self.arg.batch_size, n_way=self.arg.train_N_way, num_shots=self.arg.train_N_shots
        )
        [batch_x, label_x, _, _, batches_xi, labels_yi] = self.cast_variable(self.cast_cuda(data))

        out_metric, out_logits = self.model(inputs=[batch_x, batches_xi, labels_yi])

        # Loss
        label_x_numpy = label_x.cpu().data.numpy()
        formatted_label_x = np.argmax(label_x_numpy, axis=1)
        formatted_label_x = Variable(torch.LongTensor(formatted_label_x)).cuda(self.output_device)
        loss = F.nll_loss(out_logits, formatted_label_x)

        if self.arg.lambda1 > 0:
            loss = loss + self.reg_loss(self.model)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.adjust_learning_rate(iter=epoch)

        self.total_loss += loss.data
        self.counter += 1
        if epoch % self.arg.log_interval == 0:
            self.print_log('Training epoch: {}'.format(epoch))
            self.train_writer.add_scalar(self.model_name + 'Loss', self.total_loss / self.counter, epoch)
            self.print_log('\tMean training loss: {}/{}={}.'.format(self.total_loss, self.counter, self.total_loss / self.counter))
            self.counter = 0
            self.total_loss = 0

    def test(self, epoch, ln):
        self.model.eval()
        loader = self.data_loader[ln]
        test_iter = 1 if ln == 'train' else self.arg.test_iter
        correct, total = 0, 0
        pred, true = np.ones([1, self.arg.label_num]), np.ones([1, self.arg.label_num])
        for i in range(int(loader.sample_num * test_iter / self.arg.batch_size_test)+1):
            data, seti = loader.get_whole_eq(
                batch_size=self.arg.batch_size_test, n_way=self.arg.test_N_way, num_shots=self.arg.test_N_shots,
                set0=i * self.arg.batch_size_test
            )
            if seti == 0:
                break
            [x, labels_x, _, _, xi_s, labels_yi] = self.cast_variable(self.cast_cuda(data))

            with torch.no_grad():
                output, out_logits = self.model(inputs=[x, xi_s, labels_yi])
            y_pred0 = np.exp(out_logits.data.cpu().numpy())
            y_pred = np.argmax(y_pred0, axis=1)
            labels_x_cpuo = labels_x.data.cpu().numpy()
            labels_x_cpu = np.argmax(labels_x_cpuo, axis=1)

            for row_i in range(y_pred.shape[0]):
                if y_pred[row_i] == labels_x_cpu[row_i]:
                    correct += 1
                total += 1

            true = np.vstack((true, labels_x_cpuo))
            pred = np.vstack((pred, y_pred0))

        self.print_log('Eval epoch: {}'.format(epoch + 1))
        acc = correct / total
        self.print_log('{}:\tAccuracy={}/{}={}'.format(ln, correct, total, acc))
        dfi = self.evaluate(pred[1:], true[1:])

        if ln == 'train':
            self.train_writer.add_scalar(self.model_name + '/acc', acc, epoch)
        elif ln == 'test':
            self.test_writer.add_scalar(self.model_name + '/acc', acc, epoch)
            if acc is not None and acc >= self.test_acc:
                self.test_acc = acc
            self.print_log("Best test accuracy {:.4f} \n".format(self.test_acc))

        return dfi, np.argmax(true[1:], axis=1), pred[1:]

    def evaluate(self, prob, true_onehot):
        assert self.arg.label_num == 3
        true = np.argmax(true_onehot, axis=1)
        metrics = ['acc', 'pre', 'sen', 'spe', 'f1', 'auc']
        ave = 'macro'
        auc, _, _ = get_auc(true=true, prob=prob)
        evals = eval_metric(true=true, prob=prob)
        df = pd.DataFrame([[evals[ave][kk] for kk in metrics[:-1]] + [auc[ave]]],
            columns=[kk if kk[:3] == 'acc' else f'{kk}_{ave}' for kk in metrics], index=[0])
        return df

    def cast_cuda(self, input):
        if isinstance(input, type([])):
            for i in range(len(input)):
                input[i] = self.cast_cuda(input[i])
        else:
            return input.cuda(self.output_device)
        return input

    def cast_variable(self, input):
        if isinstance(input, type([])):
            for i in range(len(input)):
                input[i] = self.cast_variable(input[i])
        else:
            return Variable(input)
        return input

    def adjust_learning_rate(self, iter):
        new_lr = self.arg.lr * (0.5 ** (int(iter / self.arg.dec_lr)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, string, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            string = f'[ {localtime} ] {string}'
        print(string)
        with open(os.path.join(self.work_dir, 'log.txt'), 'a') as f:
            print(string, file=f)


if __name__ == "__main__":
    arg = get_parser()
    seed_torch(arg.seed)
    processor = Processor(arg)
    processor.start()

