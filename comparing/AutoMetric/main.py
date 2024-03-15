"""
This repository holds the PyTorch code of our JBHI paper :
*Auto-Metric Graph Neural Network Based on a Meta-learning Strategy
for the Diagnosis of Alzheimer's disease*.

All the materials released in this library can **ONLY** be used for
**RESEARCH** purposes and not for commercial use.

The authors' institution (**Biomedical Image and Health Informatics
Lab,School of Biomedical Engineering, Shanghai Jiao Tong University**)
preserve the copyright and all legal rights of these codes."""

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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import net.gnn_model_w_change as models
from utils import seed_torch, eval_metric, get_auc


parser = argparse.ArgumentParser(description='AMGNN')
parser.add_argument('--metric_network', type=str, default='gnn', metavar='N',
                    help='gnn')
parser.add_argument('--dataset', type=str, default='AD', metavar='N',
                    help='AD')
parser.add_argument('--exp_name', type=str, default='', help='Name of the experiment')
parser.add_argument('--save_dir', type=str, default='./../../results/')
parser.add_argument('--data_root', type=str, default='./../../data/')
parser.add_argument('--data_name', type=str, default='nat1_dropFeature')
parser.add_argument('--rand', type=int, default=1, metavar='N')
parser.add_argument('--fold', type=int, default=0, metavar='N')
parser.add_argument('--test_N_way', type=int, default=3, metavar='N')
parser.add_argument('--train_N_way', type=int, default=3, metavar='N')
parser.add_argument('--test_N_shots', type=int, default=10, metavar='N')
parser.add_argument('--train_N_shots', type=int, default=10, metavar='N')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--feature_num', type=int, default=638, metavar='N',
                    help='feature number of one sample')
parser.add_argument('--clinical_feature_num', type=int, default=4, metavar='N',
                    help='clinical feature number of one sample')
parser.add_argument('--w_feature_num', type=int, default=634, metavar='N',
                    help='feature number for w computation')
# parser.add_argument('--w_feature_list', type=int, default=5, metavar='N',
#                     help='feature list for w computation')
# 0-4,1-9，2-5,3-13,4-9，5-14,6-18
# 0-4,1-9，2-10,3-13,4-14，5-19,6-23
parser.add_argument('--iterations', type=int, default=500, metavar='N',
                    help='number of epochs to train ')
parser.add_argument('--dec_lr', type=int, default=300, metavar='N',
                    help='Decreasing the learning rate every x iterations')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--batch_size', type=int, default=64, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--unlabeled_extra', type=int, default=0, metavar='N',
                    help='Number of shots when training')
parser.add_argument('--test_interval', type=int, default=200, metavar='N',
                    help='how many batches between each test')
parser.add_argument('--seed', type=int, default=2019, metavar='N')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print('GPU:', args.cuda)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def adjust_learning_rate(optimizers, lr, iter):
    new_lr = lr * (0.5 ** (int(iter / args.dec_lr)))

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class Generator(DataLoader):
    def __init__(self, root, keys=[0, 1, 2]):
        with open(root, 'rb') as load_data:
            data_dict = pickle.load(load_data)
        data_ = {}
        for i in range(len(keys)):
            data_[i] = data_dict[keys[i]]
        self.data = data_
        self.channal = 1
        self.feature_shape = np.array(self.data[1][0]).shape
        # print(self.feature_shape)
        self.sample_num = np.sum(np.array([len(self.data[i]) for i in range(3)]))
        # print(self.sample_num)

    def cast_cuda(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_cuda(input[i])
        else:
            return input.cuda()
        return input

    def get_task_batch(self, batch_size=5, n_way=4, num_shots=10, unlabeled_extra=0, cuda=False, variable=False):
        # init
        batch_x = np.zeros((batch_size, self.channal, self.feature_shape[0], self.feature_shape[1]), dtype='float32')  # features
        # print(batch_x.shape)
        labels_x = np.zeros((batch_size, n_way), dtype='float32')  # labels
        labels_x_global = np.zeros(batch_size, dtype='int64')
        numeric_labels = []
        batches_xi, labels_yi, oracles_yi = [], [], []
        for i in range(n_way * num_shots):
            batches_xi.append(
                np.zeros((batch_size, self.channal, self.feature_shape[0], self.feature_shape[1]), dtype='float32'))
            labels_yi.append(np.zeros((batch_size, n_way), dtype='float32'))
            oracles_yi.append((np.zeros((batch_size, n_way), dtype='float32')))

        # feed data

        for batch_counter in range(batch_size):
            pre_class = random.randint(0, n_way - 1)
            indexes_perm = np.random.permutation(n_way * num_shots)
            counter = 0
            for class_num in range(0, n_way):
                if class_num == pre_class:
                    # We take num_shots + one sample for one class
                    samples = random.sample(self.data[class_num], num_shots + 1)
                    # Test sample

                    batch_x[batch_counter, 0, :, :] = samples[0]
                    labels_x[batch_counter, class_num] = 1  # one hot
                    samples = samples[1::]
                else:
                    samples = random.sample(self.data[class_num], num_shots)
                for samples_num in range(len(samples)):
                    try:
                        batches_xi[indexes_perm[counter]][batch_counter, :] = samples[samples_num]
                    except:
                        print(samples[samples_num])

                    labels_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    oracles_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    # target_distances[batch_counter, indexes_perm[counter]] = 0
                    counter += 1

            numeric_labels.append(pre_class)

        batches_xi = [torch.from_numpy(batch_xi) for batch_xi in batches_xi]
        labels_yi = [torch.from_numpy(label_yi) for label_yi in labels_yi]
        oracles_yi = [torch.from_numpy(oracle_yi) for oracle_yi in oracles_yi]

        labels_x_scalar = np.argmax(labels_x, 1)

        return_arr = [torch.from_numpy(batch_x), torch.from_numpy(labels_x), torch.from_numpy(labels_x_scalar),
                      torch.from_numpy(labels_x_global), batches_xi, labels_yi, oracles_yi]
        if cuda:
            return_arr = self.cast_cuda(return_arr)
        if variable:
            return_arr = self.cast_variable(return_arr)
        return return_arr


def compute_adj(batch_x, batches_xi):
    x = torch.squeeze(batch_x)
    xi_s = [torch.squeeze(batch_xi) for batch_xi in batches_xi]

    nodes = [x] + xi_s
    nodes = [node.unsqueeze(1) for node in nodes]
    nodes = torch.cat(nodes, 1)
    age = nodes.narrow(2, 0, 1)
    age = age.cpu().numpy()
    gender = nodes.narrow(2, 1, 1)
    gendre = gender.cpu().numpy()
    apoe = nodes.narrow(2, 3, 1)
    apoe = apoe.cpu().numpy()
    edu = nodes.narrow(2, 2, 1)
    edu = edu.cpu().numpy()
    adj = np.ones(
        (args.batch_size, args.train_N_way * args.train_N_shots + 1, args.train_N_way * args.train_N_shots + 1, 1),
        dtype='float32') + 4

    for batch_num in range(args.batch_size):
        for i in range(args.train_N_way * args.train_N_shots + 1):
            for j in range(i + 1, args.train_N_way * args.train_N_shots + 1):
                if np.abs(age[batch_num, i, 0] - age[batch_num, j, 0]) <= 0.06:
                    adj[batch_num, i, j, 0] -= 1
                    adj[batch_num, j, i, 0] -= 1
                if np.abs(edu[batch_num, i, 0] - edu[batch_num, j, 0]) <= 0.14:
                    adj[batch_num, i, j, 0] -= 1
                    adj[batch_num, j, i, 0] -= 1
                if gendre[batch_num, i, 0] == gendre[batch_num, j, 0]:
                    adj[batch_num, i, j, 0] -= 1
                    adj[batch_num, j, i, 0] -= 1
                if apoe[batch_num, i, 0] == apoe[batch_num, j, 0]:
                    adj[batch_num, i, j, 0] -= 1
                    adj[batch_num, j, i, 0] -= 1
    adj = 1 / adj
    adj = torch.from_numpy(adj)
    return adj.cuda()


def train_batch(model, data):
    [amgnn, softmax_module] = model
    [batch_x, label_x, batches_xi, labels_yi, oracles_yi] = data
    z_clinical, z_mri_feature = batch_x[:, 0, 0, 0:args.clinical_feature_num], batch_x[:, :, :,
                                                                               args.clinical_feature_num:]
    zi_s_clinical = [batch_xi[:, 0, 0, 0:args.clinical_feature_num] for batch_xi in batches_xi]
    zi_s_mri_feature = [batch_xi[:, :, :, args.clinical_feature_num:] for batch_xi in batches_xi]

    adj = compute_adj(z_clinical, zi_s_clinical)

    out_metric, out_logits = amgnn(
        inputs=[z_clinical, z_mri_feature, zi_s_clinical, zi_s_mri_feature, labels_yi, oracles_yi, adj])
    logsoft_prob = softmax_module.forward(out_logits)

    # Loss
    label_x_numpy = label_x.cpu().data.numpy()
    formatted_label_x = np.argmax(label_x_numpy, axis=1)
    formatted_label_x = Variable(torch.LongTensor(formatted_label_x))
    if args.cuda:
        formatted_label_x = formatted_label_x.cuda()
    loss = F.nll_loss(logsoft_prob, formatted_label_x)
    loss.backward()

    return loss


def test_one_shot(model, save_path='log.txt'):
    print_log(save_path, '\n**** TESTING BEGIN ***')
    root = os.path.join(args.data_root, 'nat', args.data_name + '_', f'divide_{args.rand}', f'test{args.fold}.pkl')
    loader = Generator(root, keys=['CN', 'MCI', 'AD'])
    test_samples = loader.sample_num
    [amgnn, softmax_module] = model
    amgnn.eval()
    correct = 0
    total = 0
    iterations = int(test_samples / args.batch_size)
    # add
    pred, true = np.ones([1, 3]), np.ones([1, 3])
    for i in range(iterations):
        data = loader.get_task_batch(batch_size=args.batch_size, n_way=args.test_N_way,
                                     num_shots=args.test_N_shots, unlabeled_extra=args.unlabeled_extra, cuda=args.cuda)
        [x_t, labels_x_cpu_t, _, _, xi_s, labels_yi_cpu, oracles_yi] = data

        z_clinical, z_mri_feature = x_t[:, 0, 0, 0:args.clinical_feature_num], x_t[:, :, :, args.clinical_feature_num:]
        zi_s_clinical = [batch_xi[:, 0, 0, 0:args.clinical_feature_num] for batch_xi in xi_s]
        zi_s_mri_feature = [batch_xi[:, :, :, args.clinical_feature_num:] for batch_xi in xi_s]

        adj = compute_adj(z_clinical, zi_s_clinical)

        x = x_t
        labels_x_cpu = labels_x_cpu_t

        if args.cuda:
            xi_s = [batch_xi.cuda() for batch_xi in zi_s_mri_feature]
            labels_yi = [label_yi.cuda() for label_yi in labels_yi_cpu]
            oracles_yi = [oracle_yi.cuda() for oracle_yi in oracles_yi]
            x = x.cuda()
        else:
            labels_yi = labels_yi_cpu

        xi_s = [Variable(batch_xi) for batch_xi in zi_s_mri_feature]
        labels_yi = [Variable(label_yi) for label_yi in labels_yi]
        oracles_yi = [Variable(oracle_yi) for oracle_yi in oracles_yi]
        z_mri_feature = Variable(z_mri_feature)

        # Compute metric from embeddings
        output, out_logits = amgnn(inputs=[z_clinical, z_mri_feature, zi_s_clinical, xi_s, labels_yi, oracles_yi, adj])
        # output由logits经过sigmoid
        Y = softmax_module.forward(out_logits)  # 经过log_softmax

        y_pred0 = np.exp(Y.data.cpu().numpy())  # 预测概率值，即只经过softmax，抵消log
        y_pred = np.argmax(y_pred0, axis=1)  # 预测数值标签
        labels_x_cpuo = labels_x_cpu.data.cpu().numpy()  # 实际onehot标签
        labels_x_cpu = np.argmax(labels_x_cpuo, axis=1)  # 实际数值标签
        for row_i in range(y_pred.shape[0]):  # batch_size_test
            if y_pred[row_i] == labels_x_cpu[row_i]:
                correct += 1  # 正确数
            total += 1  # 样本数，一直在累加，算上之前的iter一起

        true = np.vstack((true, labels_x_cpuo))
        pred = np.vstack((pred, y_pred0))

    acc = correct / total
    print_log(save_path, '{} correct from {} \tAccuracy: {:.3f}%'.format(correct, total, 100.0 * acc))
    print_log(save_path, '*** TEST FINISHED ***\n')
    dfi = evaluate(pred[1:], true[1:])

    amgnn.train()

    return acc, dfi, np.argmax(true[1:], axis=1), pred[1:]


def print_log(work_dir, string, print_time=True):
    if print_time:
        localtime = time.asctime(time.localtime(time.time()))
        string = f'[ {localtime} ] {string}'
    print(string)
    with open(os.path.join(work_dir, 'log.txt'), 'a') as f:
        print(string, file=f)


def evaluate(prob, true_onehot):
    true = np.argmax(true_onehot, axis=1)
    metrics = ['acc', 'pre', 'sen', 'spe', 'f1', 'auc']
    ave = 'macro'
    auc, _, _ = get_auc(true=true, prob=prob)
    evals = eval_metric(true=true, prob=prob)
    df = pd.DataFrame([[evals[ave][kk] for kk in metrics[:-1]] + [auc[ave]]],
        columns=[kk if kk[:3] == 'acc' else f'{kk}_{ave}' for kk in metrics], index=[0])
    return df


def main():
    ################################################################
    print(time.strftime("%F"))
    seed_torch(args.seed)
    assert args.train_N_way == args.test_N_way
    assert args.train_N_shots == args.test_N_shots
    losses = 'N{:d}s{:d}b{:d}dec{:d}lr{:g}it{:d}se{:d}'.format(
        args.train_N_way, args.train_N_shots, args.batch_size, args.dec_lr, args.lr, args.iterations, args.seed)
    exp_name = '__'.join([losses, args.exp_name])
    model_name = 't{}__{}'.format(time.strftime('%Y%m%d%H%M%S'), exp_name)
    save_path = os.path.join(args.save_dir, 'AutoMetric', args.data_name, f'rand{args.rand}', f'fold{args.fold}', model_name)
    os.makedirs(save_path, exist_ok=True)

    print_log(save_path, exp_name)
    train_writer = SummaryWriter(os.path.join(save_path, 'train'), 'train')
    test_writer = SummaryWriter(os.path.join(save_path, 'test'), 'test')
    # csv2pkl(args.data_dir, args.rand, args.fold, 'train')
    # csv2pkl(args.data_dir, args.rand, args.fold, 'test')

    # copy all files
    pwd = os.path.dirname(os.path.realpath(__file__))
    copytree(pwd, os.path.join(save_path, 'code'), symlinks=False, ignore=ignore_patterns('__pycache__'))
    arg_dict = vars(args)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(arg_dict, f)

    amgnn = models.create_models(args, cnn_dim1=2)
    print_log(save_path, str(amgnn))
    softmax_module = models.SoftmaxModule()
    if args.cuda:
        amgnn.cuda()

    weight_decay = 0

    opt_amgnn = optim.Adam(amgnn.parameters(), lr=args.lr, weight_decay=weight_decay)
    amgnn.train()
    counter = 0
    total_loss = 0
    val_acc, val_acc_aux = 0, 0
    test_acc = 0
    df_test = []
    true_test, pred_test = [], []
    for batch_idx in range(args.iterations):

        root = os.path.join(args.data_root, 'nat', args.data_name + '_', f'divide_{args.rand}', f'train{args.fold}.pkl')
        da = Generator(root, keys=['CN', 'MCI', 'AD'])
        data = da.get_task_batch(batch_size=args.batch_size, n_way=args.train_N_way,
                                 num_shots=args.train_N_shots, unlabeled_extra=args.unlabeled_extra, cuda=args.cuda)
        [batch_x, label_x, _, _, batches_xi, labels_yi, oracles_yi] = data

        opt_amgnn.zero_grad()

        loss_d_metric = train_batch(model=[amgnn, softmax_module],
                                    data=[batch_x, label_x, batches_xi, labels_yi, oracles_yi])
        opt_amgnn.step()

        adjust_learning_rate(optimizers=[opt_amgnn], lr=args.lr, iter=batch_idx)

        ####################
        # Display
        ####################
        counter += 1
        total_loss += loss_d_metric.item()
        if batch_idx % args.log_interval == 0:  # 每隔一定的iter（20），打印loss
            display_str = 'Train Iter: {}'.format(batch_idx)
            display_str += '\tLoss_d_metric: {:.6f}'.format(total_loss / counter)
            print_log(save_path, display_str)
            train_writer.add_scalar(model_name + 'Loss', total_loss / counter, batch_idx)
            counter = 0
            total_loss = 0

        ####################
        # Test
        ####################
        if (batch_idx + 1) % args.log_interval == 0:

            # test_samples = 320
            test_acc_aux, df0, true0, pred0 = test_one_shot(model=[amgnn, softmax_module], save_path=save_path)
            df_test.append(df0)
            # print(true0)
            # if len(true_test) > 0:
            #     assert (true0 == true_test).all()
            # true_test = true0
            prob = np.hstack([np.array([true0]).T, pred0])
            pred_test.append(prob)
            amgnn.train()

            test_writer.add_scalar(model_name + '/acc', test_acc_aux, batch_idx)
            # if test_acc_aux is not None and test_acc_aux >= test_acc:
            #     test_acc = test_acc_aux
            #     print_log(save_path, "Best test accuracy {:.4f} \n".format(test_acc))  # 目前迭代次数为止最好的准确率
            #     os.system('rm ' + os.path.join(save_path, f'amgnn_best_model_*.pkl'))
            #     torch.save(amgnn, os.path.join(save_path, f'amgnn_best_model_{batch_idx}.pkl'))

    all_pred_te = pd.DataFrame(np.hstack(pred_test))
    all_pred_te.to_csv(os.path.join(save_path, 'test_predictions.csv'), index=False)
    df_test = pd.concat(df_test, axis=0)
    df_test.to_csv(os.path.join(save_path, 'test_indicators.csv'), index=False)


if __name__ == '__main__':
    main()
