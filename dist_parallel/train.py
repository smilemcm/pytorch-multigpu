import os
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


# 분산 학습할때는 아래 코드가 추가된다.
# 아래 코드에서는 _distributed_c10d를 통해서 다른 노드와 통신을 한다.
import torch.distributed as dist
import torch.multiprocessing as mp
# 아래 코드에서는 DistributedSampler가 있음.
import torch.utils.data.distributed

from model import pyramidnet
import argparse
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', type=int, default=100, help='')
parser.add_argument('--num_workers', type=int, default=4, help='')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")


# 분산학습할때는 아래 코드가 더 추가된다.
# gpu부분도 main_worker()에서 다루니까..여기서는 그대로 두자.
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://192.168.0.179:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
# rank부분도 main_worker()에서 다루니까..여기서는 그대로 두자.
parser.add_argument('--rank', default=0, type=int, help='')
# 아래 world_size는 main코드에서 main()에서 값을 계산해서 넣으니까..여기서는 건드리지 말자.
parser.add_argument('--world_size', default=1, type=int, help='')
# distributed 파라메터는 사용하지도 않는데, 왜 있는지 모르겠음.
parser.add_argument('--distributed', action='store_true', help='')
args = parser.parse_args()

gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices


def main():
    args = parser.parse_args()

    # 분산 학습할때는 아래 코드가 추가가 되는데, 다른 노드에 있는 gpu개수를 확인을 못할것 같은데,
    # 내가 가지고 있는 노드의 gpu개수만 확인이 가능할것으로 추정
    ngpus_per_node = torch.cuda.device_count()

    # world_size는 모든 노드에 있는 gpus개수를 의미하는데, 입력개수로 받을때도 있어서 햇갈릴수있음.
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        
        
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()    
    print("Use GPU: {} for training".format(args.gpu))
        
    args.rank = args.rank * ngpus_per_node + gpu    
    dist.init_process_group(backend=args.dist_backend
                            ,init_method=args.dist_url
                            ,world_size=args.world_size
                            ,rank=args.rank)

    print('==> Making model..')
    net = pyramidnet()
    torch.cuda.set_device(args.gpu)
    net.cuda(args.gpu)


    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.num_workers = int(args.num_workers / ngpus_per_node)

    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])


    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    print('==> Preparing data..')
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    dataset_train = CIFAR10(root='../data',
                            train=True,
                            download=True,
                            transform=transforms_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = DataLoader(dataset_train,
                              batch_size=args.batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=args.num_workers,
                              sampler=train_sampler)

    # there are 10 classes so the dataset name is cifar-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=1e-4)
    
    train(net, criterion, optimizer, train_loader, args.gpu)
            

def train(net, criterion, optimizer, train_loader, device):
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    
    epoch_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = time.time()
        
        inputs = inputs.cuda(device)
        targets = targets.cuda(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100 * correct / total
        
        batch_time = time.time() - start
        
        if batch_idx % 20 == 0:
            print('Epoch: [{}/{}]| loss: {:.3f} | acc: {:.3f} | batch time: {:.3f}s '.format(
                batch_idx, len(train_loader), train_loss/(batch_idx+1), acc, batch_time))
    
    elapse_time = time.time() - epoch_start
    elapse_time = datetime.timedelta(seconds=elapse_time)
    print("Training time {}".format(elapse_time))
    

if __name__=='__main__':
    main()