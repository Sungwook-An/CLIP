"""
Evaluate on ImageNet. Note that at the moment, training is not implemented (I am working on it).
that being said, evaluation is working.
"""

import argparse
import os
import random
import shutil
import time
import warnings
import PIL

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import wandb

from torch.utils.data import DataLoader, random_split
from examples.imagenet.cosine_annealing_with_warmup import CosineAnnealingWarmUpRestarts

sweep_configuration = {
    'method': 'bayes',
    'name': 'lgd2024_EfficientNet_lr',
    'metric': {'goal': 'maximize', 'name': 'val top1 acc avg'},
    'parameters': 
    {
        'lr': {'max': 0.5, 'min': 0.001},
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project='lgd2024_EfficientNet')

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch import EfficientNet_ # EfficientNet for No BlurPool

# For AutoAugment
from autoaugment import ImageNetPolicy
# For AdvProp training
from attacker import PGDAttacker, NoOpAttacker

# lr, wd, seed - wandb sweep으로 가능
# Optimizer 변경 - Adam, etc ...

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')
parser.add_argument('--advprop', default=False, action='store_true',
                    help='use advprop or not')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# argument 추가
parser.add_argument('-tw', '--trained_weight', default=None, type=str, help='path of the model_best_pth.tar file (default: none)')
parser.add_argument('-bp', '--blurpool', default=False, action='store_true', help='BlurPool T/F')
parser.add_argument('-bw', '--blurpooled_model_weight', default=None, type=str, help='path of the model_best_pth.tar file (default: none)')
parser.add_argument('-s', '--sweep', default=False, action='store_true', help='Sweep T/F')

# 폐기
parser.add_argument('-n', '--noisy_student', default=None, type=str, help='path of the noisy_student.pth file (default: none)')
parser.add_argument('-aa', '--auto_augment', default=False, action='store_true', help='Auto augment T/F')
parser.add_argument('-at', '--advprop_train', default=False, action='store_true', help='Advprop T/F')

best_acc1 = 0


def main():
    wandb.init(project='lgd2024_EfficientNet')

    args = parser.parse_args()
    
    if args.sweep:
        lr = wandb.config.lr
        args.lr = lr

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print("args.distributed = ", args.distributed)
    
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    print(args)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    # create model
    if 'efficientnet' in args.arch:  # NEW
        if args.pretrained:
            # blurpooled efficient net을 사용할 때와 일반 efficient net 사용할 때 구분
            if args.blurpooled_model_weight != None:
                # weights_path를 입력
                model = EfficientNet.from_pretrained(args.arch, weights_path=args.blurpooled_model_weight, advprop=args.advprop)
                print("=> using blurpooled pre-trained model '{}'".format(args.arch))
            elif args.noisy_student != None:
                if args.blurpool:
                    model = EfficientNet.from_pretrained(args.arch, weights_path=args.noisy_student, advprop=args.advprop)
                else:
                    model = EfficientNet.from_pretrained(args.arch, weights_path=args.noisy_student, advprop=args.advprop)
                print("=> using pre-trained model 'noisy student {}'".format(args.arch))
            elif args.trained_weight != None:
                if args.blurpool:
                    model = EfficientNet.from_pretrained(args.arch, weights_path=args.trained_weight, advprop=args.advprop)
                else:
                    model = EfficientNet.from_pretrained(args.arch, weights_path=args.trained_weight, advprop=args.advprop)
                print("=> using pre-trained model '{}'".format(args.arch))
            else:
                if args.blurpool:
                    model = EfficientNet.from_pretrained(args.arch, advprop=args.advprop)
                else:
                    model = EfficientNet.from_pretrained(args.arch, advprop=args.advprop)
                print("=> using pre-trained model '{}'".format(args.arch))

        else:
            print("=> creating model '{}'".format(args.arch))
            if args.blurpool:
                model = EfficientNet.from_name(args.arch)
            else:
                model = EfficientNet.from_name(args.arch)

    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = torch.optim.RMSprop(model.parameters(), args.lr,
    #                                 momentum=args.momentum,
    #                                 weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1'].to('cuda:0')
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            if list(checkpoint['state_dict'].keys())[0].startswith('module'):
                new_state_dict = {}
                for k, v in checkpoint['state_dict'].items():
                    k = k[7:]  # remove 'module.' from the key
                    new_state_dict[k] = v
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    if args.advprop:
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    if 'efficientnet' in args.arch:
        if args.blurpool:
            image_size = EfficientNet.get_image_size(args.arch)
        else:
            image_size = EfficientNet.get_image_size(args.arch)
    else:
        image_size = args.image_size

    if args.auto_augment:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_transforms = transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    print('Using image size', image_size)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        res = validate(val_loader, model, criterion, args)
        with open('res.txt', 'w') as f:
            print(res, file=f)
        return

    wandb.log({'lr': args.lr, 'image_size': image_size})
    # LR scheduling 추가
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # LR scheduling 추가
        scheduler.step()
        #adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args).to('cuda:0')

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1).to('cuda:0')

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            if epoch == args.start_epoch:
                cur_time = time.localtime()
                timestamp = int(time.mktime(cur_time))
                
                year = time.strftime('%Y', time.localtime(timestamp))
                month = time.strftime('%m', time.localtime(timestamp))
                day = time.strftime('%d', time.localtime(timestamp))
                hour = time.strftime('%H', time.localtime(timestamp))
                minute = time.strftime('%M', time.localtime(timestamp))
                second = time.strftime('%S', time.localtime(timestamp))
                if int(hour) + 9 > 23:
                    hour = str(int(hour) + 9 - 24).zfill(2)
                    day = str(int(day) + 1).zfill(2)
                    if int(month) in [1, 3, 5, 7, 8, 10, 12]:
                        if int(day) > 31:
                            day = '01'
                            month = str(int(month)+1).zfill(2)
                    elif int(month) in [4, 6, 9, 11]:
                        if int(day) > 30:
                            day = '01'
                            month = str(int(month)+1).zfill(2)
                    elif int(month) == 2:
                        if int(year) % 4 == 0:
                            if int(day) > 29:
                                day = '01'
                                month = str(int(month) + 1).zfill(2)
                        else:
                            if int(day) > 28:
                                day = '01'
                                month = str(int(month) + 1).zfill(2)
                    if int(month) > 12:
                        month = '01'
                        year = str(int(year) + 1).zfill(2)
                else:
                    hour = str(int(hour) + 9).zfill(2)
                timestamp = f"{year[-2:]}{month}{day}_{hour}{minute}{second}"
                
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                checkpoint = f'./checkpoints/checkpoint_{timestamp}.pth.tar'
                print('timestamp = ', timestamp)
                wandb.log({'created': timestamp})
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args, checkpoint, timestamp)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        
        # AdvProp
        if args.advprop_train:
            aux_images, _ = PGDAttacker(1, 1, 1, 15, 0.0, translation=False, device='cuda:0').attack(images, target, model)
            images = torch.cat([images, aux_images], dim=0)
            target = torch.cat([target, target], dim=0)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)
            wandb.log({
                'train batch time': round(batch_time.val, 6), 
                'train batch time avg': round(batch_time.avg, 6), 
                'train data time': round(data_time.val, 6), 
                'train data time avg': round(data_time.avg, 6), 
                'train losses': round(losses.val, 6), 
                'train losses avg': round(losses.avg, 6), 
                'train top1 acc': top1.val, 
                'train top1 acc avg': top1.avg, 
                'train top5 acc': top5.val, 
                'train top5 acc avg': top5.avg
                }, step = int((epoch*len(train_loader)*args.batch_size + i*args.batch_size) / args.print_freq))

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        wandb.log({
                    'val batch time avg': round(batch_time.avg, 6), 
                    'val losses avg': round(losses.avg, 6), 
                    'val top1 acc avg': top1.avg, 
                    'val top5 acc avg': top5.avg
                    })

    return top1.avg


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar', timestamp=0):
    torch.save(state, filename)
    if is_best:
        if not os.path.exists('./model_best'):
            os.makedirs('./model_best')
        shutil.copyfile(filename, f'./model_best/model_best_{timestamp}.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    args = parser.parse_args()
    
    if args.sweep:
        wandb.agent(sweep_id, function=main)
    else:
        main()
    