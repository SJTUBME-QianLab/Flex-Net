import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from net import gnn


class MetricNN(nn.Module):
    def __init__(self, emb_size, label_num):
        super(MetricNN, self).__init__()

        self.emb_size = emb_size  # 64
        self.label_num = label_num
        num_inputs = self.emb_size + self.label_num  # num_inputs=28+2=30
        self.gnn_obj = gnn.GNN_nl(input_features=num_inputs, label_num=self.label_num, nf=96, J=1)

    def forward(self, inputs):
        """input: [batch_x, [batches_xi], [labels_yi]]"""
        [z, zi_s, labels_yi] = inputs

        # Creating WW matrix
        # z：Tensor(20,28)，zi_s：list[5](20,28)，labels_yi:list[5](20,2)
        zero_pad = Variable(torch.zeros(labels_yi[0].size())).cuda(z.device)

        labels_yi = [zero_pad] + labels_yi
        zi_s = [z] + zi_s  # z for test, zi_s for train, [6](20,28)

        # zi in zi_s, label_yi in labels_yi
        nodes = [torch.cat([zi, label_yi], 1) for zi, label_yi in zip(zi_s, labels_yi)]
        nodes = [node.unsqueeze(1) for node in nodes]
        nodes = torch.cat(nodes, 1)

        logits = self.gnn_obj(nodes).squeeze(-1)
        outputs = torch.sigmoid(logits)

        return outputs, F.log_softmax(logits, dim=1)


def load_model(model_name, args, io):
    try:
        model = torch.load('checkpoints%s/%s/models/%s.t7' % (str(args.fold), args.exp_name, model_name))
        io.cprint('Loading Parameters from the last trained %s Model' % model_name)
        return model
    except BaseException:
        io.cprint('Initiallize new Network Weights for %s' % model_name)
        pass
    return None


def create_models(args, emb_size):
    print(args.data_name)
    return MetricNN(args, emb_size)  # -> line9
