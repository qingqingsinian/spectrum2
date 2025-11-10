import argparse
import numpy as np
import os
from .net import *
import time
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

def sr_cnn(data_path: str, model_path: str, win_size: int, lr: float, epochs: int, batch: int, num_worker: int, device: str, load_path: str = None):
    def adjust_lr(optimizer: optim.Optimizer, epoch: int):
        base_lr = lr
        cur_lr = base_lr * (0.5 ** ((epoch + 10) // 10))
        for param in optimizer.param_groups:
            param['lr'] = cur_lr

    def Var(x):
        return Variable(x.to(device))

    def loss_function(x, lb):
        l2_reg = 0.
        l2_weight = 0.
        for W in net.parameters():
            l2_reg = l2_reg + W.norm(2)
        kpiweight = torch.ones(lb.shape)
        kpiweight[lb == 1] = win_size // 100
        kpiweight = kpiweight.cuda()
        BCE = F.binary_cross_entropy(x, lb, weight=kpiweight, reduction='sum')
        return l2_reg * l2_weight + BCE

    def calc(pred, true):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for pre, gt in zip(pred, true):
            if gt == 1:
                if pre == 1:
                    TP += 1
                else:
                    FN += 1
            if gt == 0:
                if pre == 1:
                    FP += 1
                else:
                    TN += 1
        print('TP=%d FP=%d TN=%d FN=%d' % (TP, FP, TN, FN))
        return TP, FP, TN, FN

    def train(epoch, net: SRCNN, gen_set):
        train_loader = DataLoader(dataset=gen_set, shuffle=True, num_workers=num_worker, batch_size=batch,
                                       pin_memory=True)
        net.train()
        train_loss = 0
        totTP, totFP, totTN, totFN = 0, 0, 0, 0
        threshold = 0.5
        for batch_idx, (inputs, lb) in enumerate(tqdm(train_loader, desc="Iteration")):
            optimizer.zero_grad()
            inputs = inputs.float()
            lb = lb.float()
            valueseq = Var(inputs)
            lb = Var(lb)
            output = net(valueseq)
            if epoch > 110:
                aa = output.detach().cpu().numpy().reshape(-1)
                res = np.zeros(aa.shape, np.int64)
                res[aa > threshold] = 1
                bb = lb.detach().cpu().numpy().reshape(-1)
                TP, FP, TN, FN = calc(res, bb)
                totTP += TP
                totFP += FP
                totTN += TN
                totFN += FN
                if batch_idx % 100 == 0:
                    print('TP=%d FP=%d TN=%d FN=%d' % (TP, FP, TN, FN))
            loss1 = loss_function(output, lb)
            loss1.backward()
            train_loss += loss1.item()
            optimizer.step()
            torch.nn.utils.clip_grad_norm(net.parameters(), 5.0)
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss1.item() / len(inputs)))

    model = SRCNN(win_size)
    net = model.to(device)
    print(net)
    base_lr = lr
    bp_parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.SGD(bp_parameters, lr=base_lr, momentum=0.9, weight_decay=0.0)

    if load_path != None:
        net = load_model(model, load_path)
        print("model loaded")

    gen_data = gen_set(win_size, data_path)
    for epoch in range(1, epochs + 1):
        print('epoch :', epoch)
        train(epoch, net, gen_data)
        adjust_lr(optimizer, epoch)
        if epoch % 5 == 0:
            save_model(model, model_path + 'srcnn_retry' + str(epoch) + '_' + str(win_size) + '.bin')
    return

import argparse
import os
import time
import torch
import time
import torch
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SRCNN Training')
    parser.add_argument('--dataset', required=True, help='Root directory of the dataset')
    parser.add_argument('--window', type=int, default=32, help='Input window size')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device for training (auto-selects GPU if available)')
    parser.add_argument('--seed', type=int, default=54321, help='Random seed for reproducibility')
    parser.add_argument('--load', default=None, help='Path to pre-trained model for resuming training')
    parser.add_argument('--save', default='snapshot', help='Directory to save trained models')
    parser.add_argument('--epoch', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loader worker processes')

    args = parser.parse_args()

    # 设置可重复性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith('cuda'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 创建模型保存目录
    os.makedirs(args.save, exist_ok=True)
    
    # 构建数据路径
    train_data_path = f"{args.dataset}_{args.window}_train.json"
    if not os.path.exists(train_data_path):
        train_data_path = os.path.join(os.getcwd(), train_data_path)

    # 记录训练时间
    start_time = time.time()
    
    sr_cnn(
        train_data_path,
        save_dir=args.save,
        window_size=args.window,
        lr=args.lr,
        epochs=args.epoch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        load_path=args.load
    )
    
    total_time = time.time() - start_time
    print(f'Training completed in {total_time:.2f} seconds')