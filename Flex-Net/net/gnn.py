import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from net import da_att

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_l = torch.cuda.LongTensor


def gmul(input):  # Wi:(300,6,6,2),x:(300,6,69or117or165),input=[Wi,x]
    W, x = input
    # x is a tensor of size (bs, N, num_features)
    # W is a tensor of size (bs, N, N, J)
    x_size = x.size()  # [300,6,69]
    W_size = W.size()  # [300,6,6,2]
    N = W_size[-2]  # 6
    W = W.split(1, 3)  # tuple[2](300,6,6,1)
    # W is now a tensor of size (bs, J*N, N)
    W = torch.cat(W, 1).squeeze(3)
    # output has size (bs, J*N, num_features)
    output = torch.bmm(W, x)
    output = output.split(N, 1)
    # output has size (bs, N, J*num_features)
    output = torch.cat(output, 2)
    return output


class Gconv(nn.Module):
    def __init__(self, nf_input, nf_output, J, bn_bool=True):
        super(Gconv, self).__init__()
        self.J = J
        self.num_inputs = J * nf_input
        self.num_outputs = nf_output
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

        self.bn_bool = bn_bool
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(self.num_outputs)

    def forward(self, input):  # Wi:(300,6,6,2),x:(300,6,69),input=[Wi,x]
        W = input[0]  # Wi (300,6,6,2)
        x = gmul(input)  # out has size (bs, N, num_inputs) num_inputs = J * nf_input
        # if self.J == 1:
        #    x = torch.abs(x)
        x_size = x.size()  # [300,6,138]
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        x = self.fc(x)  # has size (bs*N, num_outputs)

        if self.bn_bool:
            x = self.bn(x)

        x = x.view(*x_size[:-1], self.num_outputs)  # x_size[:-1]=[300, 6]
        return W, x  # W:(300,6,6,2) x:(300,6,48)


class Wcompute(nn.Module):
    def __init__(self, input_features, nf, operator='J2', activation='softmax',
                 ratio=[2, 2, 1, 1], num_operators=1, drop=False):
        # eg. input_features=30, nf=96 or 117
        super(Wcompute, self).__init__()
        self.num_features = nf
        self.operator = operator
        self.att = da_att.CAM_Module()
        self.conv2d_1 = nn.Conv2d(input_features, int(nf * ratio[0]), 1, stride=1)

        self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]))
        self.drop = drop
        if self.drop:
            self.dropout = nn.Dropout(0.3)
        self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
        self.bn_2 = nn.BatchNorm2d(int(nf * ratio[1]))  # (10,192,6,6)
        self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), nf * ratio[2], 1, stride=1)
        self.bn_3 = nn.BatchNorm2d(nf * ratio[2])  # (10,96,6,6)
        self.conv2d_4 = nn.Conv2d(nf * ratio[2], nf * ratio[3], 1, stride=1)
        self.bn_4 = nn.BatchNorm2d(nf * ratio[3])  # (10,96,6,6)

        self.conv2d_last = nn.Conv2d(nf, num_operators, 1, stride=1)  # (10,1,6,6)
        self.activation = activation

    def forward(self, x, W_id):  # x:(20,6,30), W_id:(20,6,6,1), ratio=[2, 1.5, 1, 1]
        W1 = x.unsqueeze(2)
        # size: bs x N x N x num_features
        W2 = torch.transpose(W1, 1, 2)
        # size: bs x N x N x num_features
        W_new = torch.abs(W1 - W2)
        # size: bs x num_features x N x N
        W_new = torch.transpose(W_new, 1, 3)

        W_new = self.conv2d_1(W_new)
        W_new = self.bn_1(W_new)
        W_new = F.leaky_relu(W_new)  # (,69,6,6)->(,138,6,6)
        if self.drop:
            W_new = self.dropout(W_new)

        W_new = self.conv2d_2(W_new)
        W_new = self.bn_2(W_new)
        W_new = F.leaky_relu(W_new)  # (,103,6,6) 69*1.5=103.5

        W_new = self.conv2d_3(W_new)
        W_new = self.bn_3(W_new)  # (,69,6,6)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_4(W_new)
        W_new = self.bn_4(W_new)  # (,69,6,6)
        W_new = F.leaky_relu(W_new)

        # ---att--- #######################################
        W_new = self.att(W_new)
        
        W_new = self.conv2d_last(W_new)  # (,1,6,6)
        # size: bs x N x N x 1
        W_new = torch.transpose(W_new, 1, 3)  # (300,6,6,1)

        if self.activation == 'softmax':
            W_new = W_new - W_id.expand_as(W_new) * 1e8
            W_new = torch.transpose(W_new, 2, 3)  # (,6,1,6)
            # Applying Softmax
            W_new = W_new.contiguous()
            W_new_size = W_new.size()  # [300,6,1,6]
            W_new = W_new.view(-1, W_new.size(3))
            W_new = F.softmax(W_new, dim=1)
            W_new = W_new.view(W_new_size)
            # Softmax applied
            W_new = torch.transpose(W_new, 2, 3)  # (,6,6,1)

        elif self.activation == 'sigmoid':
            W_new = torch.sigmoid(W_new)
            W_new *= (1 - W_id)
        elif self.activation == 'none':
            W_new *= (1 - W_id)
        else:
            raise NotImplementedError

        if self.operator == 'laplace':
            W_new = W_id - W_new
        elif self.operator == 'J2':
            W_new = torch.cat([W_id, W_new], 3)
        else:
            raise NotImplementedError

        return W_new


class GNN_nl(nn.Module):
    def __init__(self, input_features, label_num, nf, J):
        super(GNN_nl, self).__init__()
        self.input_features = input_features  # 69
        self.label_num = label_num
        self.nf = nf  # 96
        self.J = J  # 2

        self.num_layers = 2
        for i in range(self.num_layers):
            module_w = Wcompute(self.input_features + int(nf / 2) * i,  # 30+96/2*1=117
                                self.input_features + int(nf / 2) * i,
                                operator='J2', activation='softmax', ratio=[2, 1.5, 1, 1], drop=False)
            module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)  # 96/2=48
            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        self.w_comp_last = Wcompute(self.input_features + int(self.nf / 2) * self.num_layers,
                                    self.input_features + int(self.nf / 2) * (self.num_layers - 1),
                                    operator='J2', activation='softmax', ratio=[2, 1.5, 1, 1], drop=True)
        self.layer_last = Gconv(self.input_features + int(self.nf / 2) * self.num_layers, self.label_num, 2)  # 69+48*2=165, 5, 2

    def forward(self, x):  # x=nodes:(300,6,69)
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3)).cuda(x.device)

        for i in range(self.num_layers):  # num_layers=2
            Wi = self._modules['layer_w{}'.format(i)](x, W_init)
            x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])
            x = torch.cat([x, x_new], 2)  # (300,6,69+48=117or117+48=165)
        Wl = self.w_comp_last(x, W_init)
        out = self.layer_last([Wl, x])[1]

        return out[:, 0, :]  # (300,5)

