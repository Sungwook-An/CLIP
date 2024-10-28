import argparse
import os
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
assert cvd is not None, "Error: CUDA_VISIBLE_DEVICES must be set!"
import sys
import random
import shutil
import warnings
import PIL
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.data import distributed, DataLoader, Dataset
import torchvision.transforms as transforms

import wandb
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# pip install transformers
from transformers import AutoTokenizer, CLIPTextModel
from imagenet_class import CLASSNAMES

from efficientnet_pytorch import EfficientNet

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='efficientnet-b1',
                    help='model architecture (default: efficientnet-b1)')
parser.add_argument('--imagenet', default=False, action='store_true',
                    help='ImageNet val set path')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lp_epochs', default=0, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--ft_epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--drop_out', default=0.1, type=float, help='drop out')
parser.add_argument('--warmup_epochs', default=0, type=int, help='warmup epochs')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--image_size', default=240, type=int,
                    help='image size')
parser.add_argument('--advprop', default=False, action='store_true',
                    help='use advprop or not')

MODEL = 'openai/clip-vit-large-patch14'
BEST_ACC1 = 0


import open_clip

# 원하는 text encoder 모델 이름과 해당 모델의 가중치(체크포인트) 이름을 지정
model_name = "ViT-B-32"
pretrained = "openai"

# 모델과 Preprocessor 로드
model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)

# dataset class 정의
class ImageTextDataset(Dataset):
    def __init__(self, df, transform=None):
        self.data_frame = df
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx, 0]
        images = Image.open(img_name).convert('RGB')
        
        if self.transform:
            images = self.transform(images)
        
        text = self.data_frame.iloc[idx, 1]
        text_inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        labels = self.data_frame.iloc[idx, 2]

        return images, text_inputs, labels
    
class ImgProjectionHead(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(ImgProjectionHead, self).__init__()
        dropout_prob = args.drop_out
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x, self.logit_scale.exp()
    
class TxtProjectionHead(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(TxtProjectionHead, self).__init__()
        dropout_prob = args.drop_out
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]
    
    
def load_and_split_data(args, csv_file, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_file)
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    
    return train_df, val_df
    
def collate_fn(batch):
    images, text_inputs, labels = zip(*batch)
    
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    input_ids = pad_sequence([item['input_ids'].squeeze(0) for item in text_inputs], batch_first=True)
    attention_mask = pad_sequence([item['attention_mask'].squeeze(0) for item in text_inputs], batch_first=True)
    
    text_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }
    
    return images, text_inputs, labels

# Contrastive loss function 정의
def contrastive_loss(args, image_embeds, text_embeds, labels, logit_scale):
    # ########## Old loss
    logits_per_image = logit_scale * image_embeds @ text_embeds.t()
    logits_per_text = logits_per_image.t()
    # logits_per_image = torch.matmul(logit_scale * image_embeds, text_embeds.t())
    # logits_per_text = torch.matmul(text_embeds, (logit_scale * image_embeds).t())
    labels = torch.arange(len(image_embeds)).to(logits_per_image.device)
    # print(labels.size())
    labels = labels.cuda(args.gpu)
    
    image_loss = nn.CrossEntropyLoss()(logits_per_image, labels)
    text_loss = nn.CrossEntropyLoss()(logits_per_text, labels)
    loss = (image_loss + text_loss) / 2
    
    # ########## New loss
    # image_embeds = F.normalize(image_embeds, dim=1)
    # text_embeds = F.normalize(text_embeds, dim=1)
    # # similarity = image_embeds @ text_embeds.t()
    
    # labels = (labels.unsqueeze(1) == labels.unsqueeze(0)).cuda(args.gpu)
    # labels = labels.float()
    
    # distances = torch.cdist(image_embeds, text_embeds, p=2)
    
    # positive_pairs = labels * torch.pow(distances, 2)
    # negative_pairs = (1 - labels) * torch.pow(torch.clamp(1 - distances, min=0.0), 2)
    
    # loss = torch.sum(positive_pairs + negative_pairs)
    
    return loss

# def linear_probing(args, image_encoder, text_encoder, image_projection, text_projection, train_loader, train_sampler, epoch, scheduler, optimizer):
#     image_projection.train()
#     text_projection.train()
    
#     if args.distributed:
#         train_sampler.set_epoch(epoch)
#     scheduler.step()
    
#     print('lp epoch=', epoch)
#     print('train')
#     for images, texts, labels in tqdm(train_loader):
#         images = images.cuda(args.gpu, non_blocking=True)
#         labels = labels.cuda(args.gpu)

#         # Image와 text를 encoding
#         image_features = image_encoder(images).cuda(args.gpu)
#         text_inputs = {k: v.squeeze(1).cuda(args.gpu) for k, v in texts.items()}
#         text_features = text_encoder(**text_inputs).pooler_output

#         # Feature embedding
#         image_embeds = image_projection(image_features)
#         text_embeds = text_projection(text_features)
#         logit_scale = image_projection.module.logit_scale
#         # logit_scale = 1

#         # Loss 계산 및 back prop
#         lp_ft_loss = contrastive_loss(args, image_embeds, text_embeds, labels, logit_scale)
        
#         similarities = torch.matmul(image_embeds, text_embeds.T)

#         top1_pred_indices = similarities.argmax(dim=1)
#         top1_pred_labels = labels[top1_pred_indices]

#         # Compute top-1 accuracy
#         correct = (top1_pred_labels == labels).sum().item()
#         total = labels.size(0)
#         top1_accuracy = correct / total

#         if args.local_rank == 0:
#             wandb.log({'lp_ft_top1_accuracy': top1_accuracy*100, 'lp_ft_loss': lp_ft_loss})
        
#         optimizer.zero_grad()
#         lp_ft_loss.backward()
#         optimizer.step()
#         strFormat = '%-6s%-5s%-20s%-10s%-11s%-8s\n'
#         strOut = strFormat % ('epoch=', epoch, 'lp_ft_top1_accuracy=', round(top1_accuracy*100, 4), 'lp_ft_loss=', round(lp_ft_loss.item(), 4))
#         # print(strOut, end="")


def ft_one_epoch(args, image_encoder, text_encoder, image_projection, text_projection, epoch, images, texts, labels):
    images = images.cuda(args.gpu, non_blocking=True)
    labels = labels.cuda(args.gpu)

    # Image와 text를 encoding
    image_features = image_encoder(images).cuda(args.gpu)
    text_inputs = {k: v.squeeze(1).cuda(args.gpu) for k, v in texts.items()}
    text_features = text_encoder(**text_inputs).pooler_output

    # Feature embedding
    image_embeds, logit_scale = image_projection(image_features)
    text_embeds = text_projection(text_features)
    # logit_scale = 1

    # Loss 계산 및 back prop
    lp_ft_loss = contrastive_loss(args, image_embeds, text_embeds, labels, logit_scale)

    similarities = logit_scale * torch.matmul(image_embeds, text_embeds.T)

    top1_pred_indices = similarities.argmax(dim=1)
    top1_pred_labels = labels[top1_pred_indices]

    # Compute top-1 accuracy
    correct = (top1_pred_labels == labels).sum().item()
    total = labels.size(0)
    top1_accuracy = correct / total
    
    # strFormat = '%-6s%-5s%-20s%-10s%-11s%-8s\n'
    # strOut = strFormat % ('epoch=', epoch, 'lp_ft_top1_accuracy=', round(top1_accuracy*100, 4), 'lp_ft_loss=', round(lp_ft_loss.item(), 4))
    # print(strOut, end="")
    
    return top1_accuracy, lp_ft_loss

def full_fine_tuning(args, image_encoder, text_encoder, image_projection, text_projection, train_loader, train_sampler, epoch, scheduler, optimizer):
    image_encoder.eval()
    text_encoder.eval()
    image_projection.train()
    text_projection.train()
    
    # image_projection._set_static_graph()
    
    if args.distributed:
        train_sampler.set_epoch(epoch)
    scheduler.step()
    
    if args.local_rank == 0:
        print("full fine tuning")
        for images, texts, labels in tqdm(train_loader):
            top1_accuracy, lp_ft_loss = ft_one_epoch(args, image_encoder, text_encoder, image_projection, text_projection, epoch, images, texts, labels)
            
            optimizer.zero_grad()
            lp_ft_loss.backward()
            optimizer.step()
            
            wandb.log({'lp_ft_top1_accuracy': top1_accuracy*100, 'lp_ft_loss': lp_ft_loss})
    else:
        for images, texts, labels in train_loader:
            top1_accuracy, lp_ft_loss = ft_one_epoch(args, image_encoder, text_encoder, image_projection, text_projection, epoch, images, texts, labels)
            
            optimizer.zero_grad()
            lp_ft_loss.backward()
            optimizer.step()

def val_one_epoch(args, image_encoder, text_encoder, image_projection, text_projection, image_embeddings_all, text_embeddings_all, labels_all, images, texts, labels):
    images = images.cuda(args.gpu)
    labels = labels.cuda(args.gpu)
    
    image_features = image_encoder(images)
    text_inputs = {k: v.squeeze(1).cuda(args.gpu) for k, v in texts.items()}
    text_features = text_encoder(**text_inputs).pooler_output

    # Feature embedding
    image_embeds, logit_scale = image_projection(image_features)
    text_embeds = text_projection(text_features)
    
    image_embeddings_all.append(image_embeds)
    text_embeddings_all.append(text_embeds)
    labels_all.append(labels)
    
    return image_embeddings_all, text_embeddings_all, labels_all, logit_scale

def get_metric(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()

        metric = np.mean(preds < 1)
        # metrics[f"{name}_mean_rank"] = preds.mean() + 1
        # metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        # for k in [1, 5, 10]:
        #     metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metric

def validation(args, image_encoder, text_encoder, image_projection, text_projection, val_loader, epoch, scheduler, optimizer):
    global BEST_ACC1
    
    image_encoder.eval()
    text_encoder.eval()
    image_projection.eval()
    text_projection.eval()
    
    image_embeddings_all = []
    text_embeddings_all = []
    labels_all = []
    
    with torch.no_grad():
        tokenizer_new = AutoTokenizer.from_pretrained(MODEL)
        if args.local_rank == 0:
            print('validation')
            for images, texts, labels in tqdm(val_loader):
                image_embeddings_all, text_embeddings_all, labels_all, logit_scale = val_one_epoch(args, image_encoder, text_encoder, image_projection, text_projection, image_embeddings_all, text_embeddings_all, labels_all, images, texts, labels)
        else:
            for images, texts, labels in val_loader:
                image_embeddings_all, text_embeddings_all, labels_all, logit_scale = val_one_epoch(args, image_encoder, text_encoder, image_projection, text_projection, image_embeddings_all, text_embeddings_all, labels_all, images, texts, labels)
    
        image_embeddings_all = torch.cat(image_embeddings_all)
        text_embeddings_all = torch.cat(text_embeddings_all)
        labels_all = torch.cat(labels_all)

        text_embeddings_new = []

        for c in tqdm(CLASSNAMES):
            texts = ["A photo of a {}.".format(c)]
            texts = tokenizer_new(texts, return_tensors='pt', padding=True, truncation=True)
            text_inputs = {k: v.squeeze(1).cuda(args.gpu) for k, v in texts.items()}
            text_features = text_encoder(**text_inputs).pooler_output

            # Feature embedding
            text_embeds = text_projection(text_features)
            text_embeddings_new.append(text_embeds)
        
        text_embeddings_new = torch.cat(text_embeddings_new)
        print("text_embeddings_new :", text_embeddings_new.size())
        print("image_embeddings_all :", image_embeddings_all.size())
        print("labels_all :", labels_all.size())

# #####EDit######
        # # Normalize embeddings
        # image_embeddings_all = F.normalize(image_embeddings_all, dim=1)
        # text_embeddings_new = F.normalize(text_embeddings_new, dim=1)
        
        similarities = logit_scale * torch.matmul(image_embeddings_all, text_embeddings_new.T)
        top1_pred_indices = ((similarities.argmax(dim=1))).to(torch.int)
        
        print(f"top1_pred_indices: {top1_pred_indices[:100]}")
        print(f"labels_all: {labels_all[:100]}")
        print(f"top1_pred_indices size: {top1_pred_indices.size()}, {top1_pred_indices.dtype}")
        print(f"labels_all size: {labels_all.size()}, {labels_all.dtype}")
        
        # top1_pred_labels = labels_all[top1_pred_indices]

        # Compute top-1 accuracy
        correct = (top1_pred_indices == labels_all).sum().item()

        total = labels_all.size(0)
        top1_accuracy = correct / total
######EDit######
        # image_embeddings_all = F.normalize(image_embeddings_all, dim=1)
        # text_embeddings_all = F.normalize(text_embeddings_all, dim=1)

        # similarities = torch.matmul(image_embeddings_all, text_embeddings_all.t())
        # print(similarities.size())
        # print("similarities :", similarities)
        # similarities = similarities.argmax(dim=1)
        
        # correct = (similarities == torch.arange(similarities.size(0)).cuda()).sum().item()

        # total = text_embeddings_all.size(0)
        # top1_accuracy = correct / total

        # top1_accuracy = get_metric(image_embeddings_all, text_embeddings_all, logit_scale)
        
        if args.local_rank == 0:
            print('top1_accuracy :', top1_accuracy*100, end=" ")
            print('%')
            wandb.log({'top1_accuracy': top1_accuracy*100})
            
        is_best = top1_accuracy > BEST_ACC1
        BEST_ACC1 = max(top1_accuracy, BEST_ACC1)

        if not args.distributed or (args.distributed and args.rank == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'image_encoder_state_dict': image_encoder.state_dict(),
                'text_encoder_state_dict': text_encoder.state_dict(),
                'image_projection_state_dict': image_projection.state_dict(),
                'text_projection_state_dict': text_projection.state_dict(),
                'best_acc1': BEST_ACC1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best)
            
    torch.cuda.empty_cache()
    
    return top1_accuracy

def save_checkpoint(state, is_best, filename='/home/mango/LGD-CLIP/clip_ckpt/last.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '/home/mango/LGD-CLIP/clip_ckpt/model_best.pth.tar')
    
# def zeroshot_classifier(args, image_encoder, text_encoder, image_projection, text_projection, val_loader):
#     global MODEL
    
#     image_encoder.eval()
#     text_encoder.eval()
#     image_projection.eval()
#     text_projection.eval()
    
#     image_embeddings_all = []
#     text_embeddings_all = []
#     labels_all = []
    
#     with torch.no_grad():
#         tokenizer = AutoTokenizer.from_pretrained(MODEL)
#         for c in tqdm(CLASSNAMES):
#             texts = [t.format(c) for t in TEMPLATES]
#             texts = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
#             text_inputs = {k: v.squeeze(1).cuda(args.gpu) for k, v in texts.items()}
#             text_features = text_encoder(**text_inputs).pooler_output

#             # Feature embedding
#             text_embeds = text_projection(text_features)
#             text_embeddings_all.append(text_embeds)
        
#         text_embeddings_all = torch.cat(text_embeddings_all)
#         # text_embeddings_all : 80000x512
    
#     with torch.no_grad():
#         for images, labels in tqdm(val_loader):
#             images = images.cuda(args.gpu)
#             labels = labels.cuda(args.gpu)
            
#             image_features = image_encoder(images)

#             # Feature embedding
#             image_embeds = image_projection(image_features)
            
#             image_embeddings_all.append(image_embeds)
#             labels_all.append(labels)
    
#     image_embeddings_all = torch.cat(image_embeddings_all)
#     labels_all = torch.cat(labels_all)
#     # image_embedding_all : 10000x512
#     # labels_all : 10000x512
    
#     # Normalize embeddings
#     # image_embeddings_all = F.normalize(image_embeddings_all, dim=1)
#     # text_embeddings_all = F.normalize(text_embeddings_all, dim=1)
    
#     similarities = torch.matmul(image_embeddings_all, text_embeddings_all.T)
#     top1_pred_indices = ((similarities.argmax(dim=1))/len(TEMPLATES)).to(torch.int)
    
#     print(f"top1_pred_indices: {top1_pred_indices[:100]}")
#     print(f"labels_all: {labels_all[:100]}")
#     print(f"top1_pred_indices size: {top1_pred_indices.size()}, {top1_pred_indices.dtype}")
#     print(f"labels_all size: {labels_all.size()}, {labels_all.dtype}")
    
#     top1_pred_labels = labels_all[top1_pred_indices]

#     # Compute top-1 accuracy
#     correct = (top1_pred_labels == labels_all).sum().item()
#     print("top1_pred_labels :", top1_pred_labels)
#     print("labels_all :", labels_all)
#     total = labels_all.size(0)
#     top1_accuracy = correct / total

#     print('top1_accuracy :', top1_accuracy*100, end=" ")
#     print('%')
#     if args.local_rank == 0:
#         wandb.log({'top1_accuracy': top1_accuracy*100})

   
def main():    
    args = parser.parse_args()
    torch.set_num_threads(args.workers)
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # if args.gpu is not None:
    #     warnings.warn('You have chosen a specific GPU. This will completely '
    #                   'disable data parallelism.')

    cvd = list(os.environ["CUDA_VISIBLE_DEVICES"])
    args.distributed = False
    if len(cvd) > 1:
        args.distributed = True
        args.world_size = int(os.environ["WORLD_SIZE"])
    print("args.distributed = ", args.distributed)
    print("args.world_size = ", args.world_size)
    
    ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        main_worker(args)
        # mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args)


def main_worker(args):
    global MODEL
    torch.autograd.set_detect_anomaly(True)
    
    args.rank = int(os.environ.get("RANK", 0))
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.gpu = args.local_rank
    print('args:', args)
    print(f'GPU: {args.gpu}, Local rank: {args.local_rank}, Rank: {args.rank}')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        torch.cuda.set_device(args.gpu)
        torch.distributed.barrier()
    
    if args.local_rank == 0:
        wandb.init(project='lgd2024_clip')
        
    # 사전 학습된 모델 로드
    # Image encoder는 EfficientNet_-b1 사용, text encoder는 Transformer 사용
    image_encoder = EfficientNet.from_pretrained(args.arch, weights_path='/home/mango/LGD-CLIP/model_best_blurpool_78_528.pth.tar', advprop=args.advprop)
    
    if hasattr(image_encoder, '_fc'):
        image_encoder._fc = torch.nn.Identity()

    text_encoder = CLIPTextModel.from_pretrained(MODEL)

    image_projection = ImgProjectionHead(args, input_dim=1280, output_dim=512)
    text_projection = TxtProjectionHead(args, input_dim=768, output_dim=512)
    
    if args.distributed:
        image_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(image_encoder)
        text_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(text_encoder)
        image_projection = torch.nn.SyncBatchNorm.convert_sync_batchnorm(image_projection)
        text_projection = torch.nn.SyncBatchNorm.convert_sync_batchnorm(text_projection)
        
        image_encoder = image_encoder.cuda(args.gpu)
        image_encoder = torch.nn.parallel.DistributedDataParallel(image_encoder, device_ids=[args.gpu], find_unused_parameters=True)

        text_encoder = text_encoder.cuda(args.gpu)
        text_encoder = torch.nn.parallel.DistributedDataParallel(text_encoder, device_ids=[args.gpu], find_unused_parameters=True)

        image_projection = image_projection.cuda(args.gpu)
        image_projection = torch.nn.parallel.DistributedDataParallel(image_projection, device_ids=[args.gpu], find_unused_parameters=True)

        text_projection = text_projection.cuda(args.gpu)
        text_projection = torch.nn.parallel.DistributedDataParallel(text_projection, device_ids=[args.gpu], find_unused_parameters=True)
    else:
        image_encoder = image_encoder.cuda(args.gpu)
        text_encoder = text_encoder.cuda(args.gpu)
        image_projection = image_projection.cuda(args.gpu)
        text_projection = text_projection.cuda(args.gpu)

    cudnn.benchmark = True
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            # if args.gpu is not None:
            #     # best_acc1 may be from a checkpoint from a different GPU
            #     best_acc1 = best_acc1.to(args.gpu)
            image_encoder.load_state_dict(checkpoint['image_encoder_state_dict'])
            text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
            image_projection.load_state_dict(checkpoint['image_projection_state_dict'])
            text_projection.load_state_dict(checkpoint['text_projection_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_size = args.image_size
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])
    val_transforms = transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    
    train_csv_file = '/home/mango/LGD-CLIP/imagenet_caption_train_with_labels.csv'
    val_csv_file = '/home/mango/LGD-CLIP/imagenet_caption_val_with_labels.csv'
    train_df = pd.read_csv(train_csv_file)
    val_df = pd.read_csv(val_csv_file)
    
    train_dataset = ImageTextDataset(train_df, train_transforms)
    val_dataset = ImageTextDataset(val_df, val_transforms)
    
    if args.distributed:
        train_sampler = distributed.DistributedSampler(dataset=train_dataset, shuffle=True)
        val_sampler = distributed.DistributedSampler(dataset=val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        
    # if args.distributed:
    #     train_loader = DataLoader(train_dataset, batch_size=args.batch_size//args.world_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True, sampler=train_sampler)
    #     val_loader = DataLoader(val_dataset, batch_size=args.batch_size//args.world_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True, sampler=val_sampler)
    # else:
    #     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    #     val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    
    # # Learning loop
    # optimizer = optim.Adam(list(image_projection.parameters()) + list(text_projection.parameters()), lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.lp_epochs)
    
    # prev_top1_accuracy = 0
    # count = 0
    # for param in image_encoder.parameters():
    #     param.requires_grad = False
    # for param in text_encoder.parameters():
    #     param.requires_grad = False
    # for epoch in range(args.lp_epochs):
    #     if args.distributed:
    #         train_sampler.set_epoch(epoch)
    #     linear_probing(args, image_encoder, text_encoder, image_projection, text_projection, train_loader, train_sampler, epoch, scheduler, optimizer)
    #     top1_accuracy = validation(args, image_encoder, text_encoder, image_projection, text_projection, val_loader, epoch, scheduler, optimizer)
        # if prev_top1_accuracy < top1_accuracy:
        #     prev_top1_accuracy = top1_accuracy
        #     count = 0
        # else:
        #     count += 1
        #     if count > 20:
        #         break
    
    # args.lr = args.lr*0.1
    # args.ft_batch_size = int(args.batch_size/4)
    args.ft_batch_size = args.batch_size
    
    optimizer = optim.Adam(list(image_projection.parameters()) + list(text_projection.parameters()), lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.ft_epochs)
    # scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.ft_epochs+args.warmup_epochs, min_lr=args.lr/10)
    
    if args.distributed:
        train_loader = DataLoader(train_dataset, batch_size=args.ft_batch_size//args.world_size, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=args.ft_batch_size//args.world_size, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True, sampler=val_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.ft_batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.ft_batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    
    prev_top1_accuracy = 0
    count = 0
    
    image_projection._set_static_graph()
    
    for param in image_encoder.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False
    for epoch in range(args.ft_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        full_fine_tuning(args, image_encoder, text_encoder, image_projection, text_projection, train_loader, train_sampler, epoch, scheduler, optimizer)
        top1_acc = validation(args, image_encoder, text_encoder, image_projection, text_projection, val_loader, epoch, scheduler, optimizer)
        print(f"top1_acc: {top1_acc}")

if __name__ == '__main__':
    main()
