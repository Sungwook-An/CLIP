import argparse
import os
import sys
import random
import shutil
import warnings
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.nn.utils.rnn import pad_sequence
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.data import distributed, DataLoader, Dataset
import torchvision.transforms as transforms

import pandas as pd
from PIL import Image
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
parser.add_argument('--base_path', metavar='DIR', default='./',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='efficientnet-b1',
                    help='model architecture (default: efficientnet-b1)')
parser.add_argument('--imagenet', default=False, action='store_true',
                    help='ImageNet val set path')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--eta_min', default=0.0, type=float,
                    metavar='ETA MIN', help='eta_min for Cosine Annealing LR')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--drop_out', default=0.1, type=float, help='drop out')
parser.add_argument('--warmup_epochs', default=10, type=int, help='warmup epochs')
parser.add_argument('--projection_dim', default=256, type=int, help='projection dimension')

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
parser.add_argument('--image_size', default=256, type=int,
                    help='image size')

parser.add_argument('--validate', default=False, action='store_true',
                    help='only validate')
parser.add_argument('--classification', default=False, action='store_true',
                    help='only classification')

parser.add_argument('--imagenet_pretrained', default='ImageNet_clip_ckpt/model_best.pth.tar', type=str, help='imagenet pretrained model')
parser.add_argument('--save_folder', default=None, type=str, help='save folder')
parser.add_argument('--save_last_file', default='last.pth.tar',type=str, help='last checkpoint file')
parser.add_argument('--save_best_file', default='model_best.pth.tar',type=str, help='best checkpoint file')
parser.add_argument('--img_encoder_path', default='/data001/dlpusers/kangx80/PROJECT/AI_SHARE/SOGANG-LGD_SHARE/SCL_domain_specific/save/SupCon/lgd_dataset_models/LP20_FT80_NoAug_blurpool_SupCon_lgd_dataset_efficientnet_b1_lr_0.1_decay_0.0001_bsz_128_temp_0.07_trial_0_cosine/ckpt_epoch_100.pth', type=str, help='best checkpoint file')
parser.add_argument('--train_csv_file', default='csv/lgd_dataset_train.csv', type=str, help='train csv file')
parser.add_argument('--val_csv_file', default='csv/lgd_dataset_val.csv', type=str, help='val csv file')

MODEL = 'openai/clip-vit-large-patch14'
BEST_ACC1 = 0

# dataset class
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

        return images, text_inputs, labels, text
    
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
        self.tau = nn.Parameter(torch.tensor(0.07))
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x, self.tau

class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr, last_epoch=-1):
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
    
def collate_fn(batch):
    images, text_inputs, labels, str_texts = zip(*batch)
    
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    input_ids = pad_sequence([item['input_ids'].squeeze(0) for item in text_inputs], batch_first=True)
    attention_mask = pad_sequence([item['attention_mask'].squeeze(0) for item in text_inputs], batch_first=True)
    
    text_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }
    
    str_texts = list(str_texts)
    
    return images, text_inputs, labels, str_texts

# Contrastive loss function
def contrastive_loss(args, image_embeds, text_embeds, labels, logit_scale, tau):
    # ########## Old loss
    # logits_per_image = logit_scale * image_embeds @ text_embeds.T
    # logits_per_text = logits_per_image.T
    # labels = torch.arange(len(image_embeds)).to(logits_per_image.device)
    # labels = labels.cuda(args.gpu)
    
    # image_loss = nn.CrossEntropyLoss()(logits_per_image, labels)
    # text_loss = nn.CrossEntropyLoss()(logits_per_text, labels)
    # loss = (image_loss + text_loss) / 2
    

    ########## New loss
    # similarity_matrix = logit_scale * image_embeds @ text_embeds.T
    
    # labels = (labels.unsqueeze(1) == labels.unsqueeze(0)).cuda(args.gpu)
    # labels = labels.float()
    
    # positive_pairs = labels * torch.pow(1 - similarity_matrix, 2)
    # negative_pairs = (1 - labels) * torch.pow(torch.clamp(1 + similarity_matrix), 2)
    
    # loss = torch.sum(positive_pairs + negative_pairs)
    
    # Class-wise contrastive loss
    similarity_matrix = logit_scale * image_embeds @ text_embeds.T
    
    labels = (labels.unsqueeze(1) == labels.unsqueeze(0)).cuda(args.gpu)
    positive_mask = labels.float()
    
    scaled_similarity_matrix = similarity_matrix / tau
    exp_similarity = torch.exp(scaled_similarity_matrix)
    
    masked_exp_similarity = exp_similarity * (1 - positive_mask)
    denom = masked_exp_similarity.sum(dim=1, keepdim=True) + exp_similarity.sum(dim=1, keepdim=True)
    probs = exp_similarity / denom
    
    log_probs = (probs * positive_mask).sum(dim=1)
    loss_img_to_txt = -torch.log(log_probs).mean()
    
    scaled_similarity_matrix_T = similarity_matrix.T / tau
    exp_similarity_T = torch.exp(scaled_similarity_matrix_T)
    
    masked_exp_similarity_T = exp_similarity_T * (1 - positive_mask.T)
    denom_T = masked_exp_similarity_T.sum(dim=1, keepdim=True) + exp_similarity_T.sum(dim=1, keepdim=True)
    probs_T = exp_similarity_T / denom_T
    
    log_probs_T = (probs_T * positive_mask.T).sum(dim=1)
    loss_txt_to_img = -torch.log(log_probs_T).mean()
    
    loss = (loss_img_to_txt + loss_txt_to_img) / 2
    
    return loss

def save_checkpoint(args, state, is_best):
    folder = os.path.join(args.base_path, args.save_folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, args.save_last_file)
    model_best_pth_tar = os.path.join(folder, args.save_best_file)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_best_pth_tar)

def train_one_epoch(args, image_encoder, text_encoder, text_projection, epoch, images, texts, labels, logit_scale):
    images = images.cuda(args.gpu, non_blocking=True)
    labels = labels.cuda(args.gpu)

    # Image text encoding
    with torch.no_grad():
        image_embeds = image_encoder(images).cuda(args.gpu)
    text_inputs = {k: v.squeeze(1).cuda(args.gpu) for k, v in texts.items()}
    text_features = text_encoder(**text_inputs).pooler_output

    text_embeds, tau = text_projection(text_features)

    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    lp_ft_loss = contrastive_loss(args, image_embeds, text_embeds, labels, logit_scale, tau)

    similarities = logit_scale * image_embeds @ text_embeds.T

    top1_pred_indices = similarities.argmax(dim=1)
    top1_pred_labels = labels[top1_pred_indices]

    correct = (top1_pred_labels == labels).sum().item()
    total = labels.size(0)
    top1_accuracy = correct / total
    
    return top1_accuracy, lp_ft_loss

def train(args, image_encoder, text_encoder, text_projection, train_loader, train_sampler, epoch, scheduler, optimizer, logit_scale):
    image_encoder.eval()
    text_encoder.train()
    text_projection.train()
    
    if args.distributed:
        train_sampler.set_epoch(epoch)
    scheduler.step()
    
    for images, texts, labels, _ in tqdm(train_loader, ncols=100):
        top1_accuracy, lp_ft_loss = train_one_epoch(args, image_encoder, text_encoder, text_projection, epoch, images, texts, labels, logit_scale)
        
        optimizer.zero_grad()
        lp_ft_loss.backward()
        optimizer.step()

def val_one_epoch(args, image_encoder, text_encoder, text_projection, image_embeddings_all, text_embeddings_all, labels_all, images, texts, labels, logit_scale):
    images = images.cuda(args.gpu)
    labels = labels.cuda(args.gpu)
    
    image_embeds = image_encoder(images)
    text_inputs = {k: v.squeeze(1).cuda(args.gpu) for k, v in texts.items()}
    text_features = text_encoder(**text_inputs).pooler_output

    text_embeds, _ = text_projection(text_features)
    
    image_embeddings_all.append(image_embeds)
    text_embeddings_all.append(text_embeds)
    labels_all.append(labels)
    
    return image_embeddings_all, text_embeddings_all, labels_all, logit_scale
    
def validation_all(args, image_encoder, text_encoder, text_projection, val_loader, epoch, scheduler, optimizer, logit_scale):
    global BEST_ACC1
    
    image_encoder.eval()
    text_encoder.eval()
    text_projection.eval()
    
    image_embeddings_all = []
    text_embeddings_all = []
    labels_all = []
    # texts_all = []
    
    with torch.no_grad():
        for images, texts, labels, str_texts in tqdm(val_loader, ncols=100):
            image_embeddings_all, text_embeddings_all, labels_all, logit_scale = val_one_epoch(args, image_encoder, text_encoder, text_projection, image_embeddings_all, text_embeddings_all, labels_all, images, texts, labels, logit_scale)
            # texts_all.extend(str_texts)
    
        image_embeddings_all = torch.cat(image_embeddings_all)
        text_embeddings_all = torch.cat(text_embeddings_all)
        labels_all = torch.cat(labels_all)

        image_embeddings_all = image_embeddings_all / image_embeddings_all.norm(dim=-1, keepdim=True)
        text_embeddings_all  = text_embeddings_all / text_embeddings_all.norm(dim=-1, keepdim=True)

        similarities = logit_scale * image_embeddings_all @ text_embeddings_all.T

        top1_pred_indices = similarities.topk(1, dim=1).indices.squeeze(1)
        top1_pred_labels = labels_all[top1_pred_indices]
        correct_top1 = (top1_pred_labels == labels_all).sum().item()
        top1_accuracy = correct_top1 / text_embeddings_all.size(0)

        # top5_pred_indices = similarities.topk(5, dim=1).indices
        # top5_pred_labels = labels_all[top5_pred_indices]
        # correct_top5 = (top5_pred_labels == labels_all.unsqueeze(1)).any(dim=1).sum().item()
        # top5_accuracy = correct_top5 / text_embeddings_all.size(0)
        
        # incorrect_top1_captions = []
        # incorrect_indices = (top1_pred_labels != labels_all).nonzero(as_tuple=True)[0]
        
        # for i in incorrect_indices:
        #     correct_caption = texts_all[i]
        #     predicted_captions = [texts_all[idx] for idx in top5_pred_indices[i]]
        #     incorrect_top1_captions.append({
        #         'correct_caption': correct_caption,
        #         'predicted_top5_captions': predicted_captions
        #     })
                
        # output_file = f'incorrect_top1_captions_{args.local_rank}.json'
        # with open(output_file, 'w', encoding='utf-8') as f:
        #     json.dump(incorrect_top1_captions, f, ensure_ascii=False, indent=4)
        # print(f"Saved incorrect top1 captions to {output_file}")
        

        print('top1_accuracy :', top1_accuracy*100, end=" ")
        print('%')
        # print('top5_accuracy :', top5_accuracy*100, end=" ")
        # print('%')
            
        if not args.validate:
            print("Updating best accuracy and saving checkpoint")
            is_best = top1_accuracy > BEST_ACC1
            BEST_ACC1 = max(top1_accuracy, BEST_ACC1)
            
            if not args.distributed or (args.distributed and args.rank == 0):
                save_checkpoint(args, {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'image_encoder_state_dict': image_encoder.state_dict(),
                    'logit_scale': logit_scale.detach().cpu().item(),
                    'text_encoder_state_dict': text_encoder.state_dict(),
                    'text_projection_state_dict': text_projection.state_dict(),
                    'best_acc1': BEST_ACC1,
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict()
                }, is_best)
            
    torch.cuda.empty_cache()
    
    return top1_accuracy

def validation(args, image_encoder, text_encoder, text_projection, val_loader, epoch, scheduler, optimizer, logit_scale):
    global BEST_ACC1
    
    image_encoder.eval()
    text_encoder.eval()
    text_projection.eval()
    
    top1_accuracy_all = []
    with torch.no_grad():
        for images, texts, labels, _ in val_loader:
            images = images.cuda(args.gpu)
            labels = labels.cuda(args.gpu)
            
            image_embeds = image_encoder(images)
            text_inputs = {k: v.squeeze(1).cuda(args.gpu) for k, v in texts.items()}
            text_features = text_encoder(**text_inputs).pooler_output
            
            text_embeds, _ = text_projection(text_features)
            
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds  = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            similarities = logit_scale * image_embeds @ text_embeds.T
            
            top1_pred_indices = ((similarities.argmax(dim=1))).to(torch.int)
            top1_pred_labels = labels[top1_pred_indices]
            
            correct = (top1_pred_labels == labels).sum().item()
            total = text_embeds.size(0)
            top1_accuracy = correct / total
            top1_accuracy_all.append(top1_accuracy)
    
        top1_accuracy = sum(top1_accuracy_all) / len(top1_accuracy_all)

        print('top1_accuracy :', top1_accuracy*100, end=" ")
        print('%')

        if not args.validate:
            if args.local_rank == 0:
                print("Updating best accuracy and saving checkpoint")
            is_best = top1_accuracy > BEST_ACC1
            BEST_ACC1 = max(top1_accuracy, BEST_ACC1)
            
            if not args.distributed or (args.distributed and args.rank == 0):
                save_checkpoint(args, {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'image_encoder_state_dict': image_encoder.state_dict(),
                    'logit_scale': logit_scale.detach().cpu().item(),
                    'text_encoder_state_dict': text_encoder.state_dict(),
                    'text_projection_state_dict': text_projection.state_dict(),
                    'best_acc1': BEST_ACC1,
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict()
                }, is_best)
            
    torch.cuda.empty_cache()
    
    return top1_accuracy

def classification(args, image_encoder, text_encoder, text_projection, val_loader, epoch, scheduler, optimizer, logit_scale):
    image_encoder.eval()
    text_encoder.eval()
    text_projection.eval()
    
    image_embeddings_all = []
    text_embeddings_all = []
    labels_all = []
    
    with torch.no_grad():
        tokenizer_new = AutoTokenizer.from_pretrained(MODEL)
        for images, texts, labels, _ in val_loader:
            image_embeddings_all, _, labels_all, logit_scale = val_one_epoch(args, image_encoder, text_encoder, text_projection, image_embeddings_all, text_embeddings_all, labels_all, images, texts, labels, logit_scale)
    
        image_embeddings_all = torch.cat(image_embeddings_all)
        labels_all = torch.cat(labels_all)

        text_embeddings_new = []

        for c in tqdm(CLASSNAMES, ncols=100):
            texts = ["A photo of a {}.".format(c)]
            texts = tokenizer_new(texts, return_tensors='pt', padding=True, truncation=True)
            text_inputs = {k: v.squeeze(1).cuda(args.gpu) for k, v in texts.items()}
            text_features = text_encoder(**text_inputs).pooler_output

            text_embeds, _ = text_projection(text_features)
            text_embeddings_new.append(text_embeds)
        
        text_embeddings_new = torch.cat(text_embeddings_new)
        print("text_embeddings_new :", text_embeddings_new.size())
        print("image_embeddings_all :", image_embeddings_all.size())
        print("labels_all :", labels_all.size())

        image_embeddings_all = image_embeddings_all / image_embeddings_all.norm(dim=-1, keepdim=True)
        text_embeddings_new  = text_embeddings_new / text_embeddings_new.norm(dim=-1, keepdim=True)
        
        similarities = logit_scale * image_embeddings_all @ text_embeddings_new.T
        top1_pred_indices = ((similarities.argmax(dim=1))).to(torch.int)
        
        correct = (top1_pred_indices == labels_all).sum().item()

        total = labels_all.size(0)
        top1_accuracy = correct / total

        if args.local_rank == 0:
            print('Classification top1_accuracy :', top1_accuracy*100, end=" ")
            print('%')
            
    torch.cuda.empty_cache()
    
    return top1_accuracy
  
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

    cvd = list(os.environ["CUDA_VISIBLE_DEVICES"])
    args.distributed = False
    if len(cvd) > 1:
        args.distributed = True
        args.world_size = int(os.environ["WORLD_SIZE"])
    print("args.distributed = ", args.distributed)
    print("args.world_size = ", args.world_size)
    
    ngpus_per_node = torch.cuda.device_count()
    main_worker(args)

def main_worker(args):
    global MODEL

    train_csv_file = os.path.join(args.base_path, args.train_csv_file)
    val_csv_file = os.path.join(args.base_path, args.val_csv_file)
    
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
    
    image_encoder = EfficientNet.from_pretrained(args.arch, weights_path=args.img_encoder_path, num_classes=args.projection_dim)
    num_features = image_encoder._fc.in_features
    image_encoder._fc = nn.Linear(num_features, args.projection_dim)

    image_projection = ImgProjectionHead(args, input_dim=1280, output_dim=512)
    text_encoder = CLIPTextModel.from_pretrained(MODEL)
    text_projection = TxtProjectionHead(args, input_dim=768, output_dim=args.projection_dim)
    
    if args.distributed:
        image_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(image_encoder)
        image_projection = torch.nn.SyncBatchNorm.convert_sync_batchnorm(image_projection)
        text_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(text_encoder)
        text_projection = torch.nn.SyncBatchNorm.convert_sync_batchnorm(text_projection)
        
        image_encoder = image_encoder.cuda(args.gpu)
        image_encoder = torch.nn.parallel.DistributedDataParallel(image_encoder, device_ids=[args.gpu], find_unused_parameters=True)
        image_projection = image_projection.cuda(args.gpu)
        image_projection = torch.nn.parallel.DistributedDataParallel(image_projection, device_ids=[args.gpu], find_unused_parameters=True)
        text_encoder = text_encoder.cuda(args.gpu)
        text_encoder = torch.nn.parallel.DistributedDataParallel(text_encoder, device_ids=[args.gpu], find_unused_parameters=True)
        text_projection = text_projection.cuda(args.gpu)
        text_projection = torch.nn.parallel.DistributedDataParallel(text_projection, device_ids=[args.gpu], find_unused_parameters=True)
    else:
        image_encoder = image_encoder.cuda(args.gpu)
        image_projection = image_projection.cuda(args.gpu)
        text_encoder = text_encoder.cuda(args.gpu)
        text_projection = text_projection.cuda(args.gpu)

    cudnn.benchmark = True
    
    pretrained_path = os.path.join(args.base_path, args.imagenet_pretrained)
    pretrained_model = torch.load(pretrained_path)
    
    state_dict = pretrained_model['image_projection_state_dict']
    new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    image_projection.load_state_dict(new_state_dict)
    
    logit_scale = image_projection.logit_scale
    
    state_dict = pretrained_model['text_encoder_state_dict']
    new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    text_encoder.load_state_dict(new_state_dict)
    
    normalize = transforms.Normalize(mean=[0.5057, 0.4387, 0.4286], std=[0.2143, 0.2160, 0.1986])
    image_size = args.image_size
    
    train_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
        ])
    val_transforms = transforms.Compose([
        # transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        # transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    
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

    print("Setting optimizer for Text Encoder Fine-Tuning")
    optimizer = optim.Adam(list(text_encoder.parameters()) + list(text_projection.parameters()) + [logit_scale], lr=args.lr, betas=(0.9, 0.98), eps=1e-6)

    if args.warmup_epochs > 0:
        scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.epochs+args.warmup_epochs, min_lr=args.eta_min)
        args.epochs += args.warmup_epochs
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.eta_min)
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
            text_projection.load_state_dict(checkpoint['text_projection_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    if args.distributed:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size//args.world_size, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size//args.world_size, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True, sampler=val_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    for param in image_encoder.parameters():
        param.requires_grad = False
    for param in image_projection.parameters():
        param.requires_grad = False

    for param in text_encoder.parameters():
        param.requires_grad = True
    for param in text_projection.parameters():
        param.requires_grad = True

    logit_scale.requires_grad = True
        
    # for group in optimizer.param_groups:
    #     for param in group['params']:
    #         print(param.shape, param.requires_grad)

    if args.validate:
        top1_acc = validation_all(args, image_encoder, text_encoder, text_projection, val_loader, 0, scheduler, optimizer, logit_scale)
        return
    elif args.classification:
        top1_acc = classification(args, image_encoder, text_encoder, text_projection, val_loader, 0, scheduler, optimizer, logit_scale)
        print(f"Classification top1_acc: {top1_acc}")
        return
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(args, image_encoder, text_encoder, text_projection, train_loader, train_sampler, epoch, scheduler, optimizer, logit_scale)
        top1_acc = validation_all(args, image_encoder, text_encoder, text_projection, val_loader, epoch, scheduler, optimizer, logit_scale)

if __name__ == '__main__':
    main()
