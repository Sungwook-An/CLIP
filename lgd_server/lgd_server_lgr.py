import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import argparse
import os
import sys
import shutil
from tqdm import tqdm
import pandas as pd
import pickle
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from efficientnet_pytorch import EfficientNet

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

parser = argparse.ArgumentParser(description='PyTorch Fine-Tuning')
parser.add_argument('--data', metavar='DIR', default='/', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='efficientnet-b1', help='model architecture (default: efficientnet-b1)')
parser.add_argument('--base_path', metavar='DIR', default='./', help='path to dataset')
parser.add_argument('--train_csv_file', default='csv/ft_lgr/lgd_dataset_train_ft.csv', type=str, help='train csv file')
parser.add_argument('--val_csv_file', default='csv/ft_lgr/lgd_dataset_val_ft.csv', type=str, help='validation csv file')
parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--validate', default=False, action='store_true', help='only validate')

parser.add_argument('--gpu', default=0, type=str, help='GPU ID')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--num_classes', default=3, type=int, help='number of classes')
parser.add_argument('--embed_dim', default=768, type=int, help='embedding dimension')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--eta_min', default=1e-6, type=float, help='minimum learning rate for cosine annealing scheduler')
parser.add_argument('--init_tau', default=0.07, type=float, help='initial temperature parameter')

parser.add_argument('--full', default=False, action='store_true', help='fine-tune full model')
parser.add_argument('--ft_img_encoder_path', default='/data001/dlpusers/kangx80/PROJECT/AI_SHARE/SOGANG-LGD_SHARE/SCL_domain_specific/save/SupCE/final_dataset_models/CE_final_final_dataset_efficientnet_b1_initLR_0.005_eta_min_1e-06_epochs_100_bsz_10_cosine_blurpool_graph_nsl_k_2_lambdagraph_0.1/ckpt_model_best.pth', type=str, help='best checkpoint file')
parser.add_argument('--text_embed_path', default='pkl/text_embeddings.pkl', help='path to text embeddings')
parser.add_argument('--save_folder', default=None, type=str, help='folder to save models')
parser.add_argument('--save_last_file', default='ckpt_last.pth', type=str, help='last checkpoint file')
parser.add_argument('--save_best_file', default='ckpt_best.pth', type=str, help='best checkpoint file')

# dataset class
class FTDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        Initialize the dataset.
       
        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = df
        self.transform = transform

    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Get the item at the specified index.
        
        Args:
            idx (int): Index of the item.
            
        Returns:
            tuple: Tuple containing the image and label.
        """
        img_name = self.data_frame.iloc[idx, 0]
        images = Image.open(img_name).convert('RGB')
        
        if self.transform:
            images = self.transform(images)
        
        labels = self.data_frame.iloc[idx, 1]

        return images, labels

class LanguageGuidedRecognitionHead(torch.nn.Module):
    def __init__(self, embed_dim, num_classes, init_tau=0.07):
        """
        Initialize Language-Guided Recognition (LGR) Head
        
        Args:
            embed_dim (int): Embedding dimension of the model (D).
            num_classes (int): Number of classes (C).
            init_tau (float): Temperature parameter for attention and classification.
        """
        super(LanguageGuidedRecognitionHead, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Learnable temperature parameter
        self.tau = torch.nn.Parameter(torch.tensor(init_tau).log())
        
        # Linear transformations for query (Q) and key (K)
        self.query_proj = torch.nn.Linear(embed_dim, embed_dim) 
        self.key_proj = torch.nn.Linear(embed_dim, embed_dim)
        
        # MLP (FCx2) for visual classification
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, image_embed, text_embed):
        """
        Forward pass of the LGR head.
        
        Args:
            image_embed (torch.Tensor): Image embedding tensor (B, D).
            text_embed (torch.Tensor): Text embedding tensor (C, D).
        
        Returns:
            torch.Tensor: Classification logits (B, C).
        """
        B, D = image_embed.size()
        C, D = text_embed.size()
        M = 1 # Number of sentences per class
        
        # Normalize the embeddings
        image_embed = F.layer_norm(image_embed, [D])    # Shape: (B, D)
        text_embed = F.layer_norm(text_embed, [D])      # Shape: (C, D)
        
        # Linear transformations for query (Q) and key (K)
        Q = self.query_proj(image_embed)                # Shape: (B, D)
        K = self.key_proj(text_embed)                   # Shape: (C, D)
        V = text_embed.unsqueeze(1)                     # Shape: (C, M, D)
        
        # Attention mechanism
        attn_scores = Q @ K.T                           # Shape: (B, C)
        attn_probs = F.softmax(attn_scores, dim=1)      # Shape: (B, C)
        attn_probs = attn_probs.view(B, C, M)           # Shape: (B, C, M)
        G = torch.einsum('bcm,cmd->bcd', attn_probs, V) # Shape: (B, C, D)
        
        # Classification probabilities                    
        P_I = F.softmax(self.mlp(image_embed), dim=-1)
        image_embed = F.normalize(image_embed, dim=-1)  # Shape: (B, D)
        G = F.normalize(G, dim=-1)                      # Shape: (B, C, D)
        cosine_sim = torch.einsum('bd,bcd->bc', image_embed, G)
        P_T = F.softmax(cosine_sim / self.tau, dim=-1)
        
        return P_I + P_T                                # Shape: (B, C)
    
def save_checkpoint(args, state, is_best):
    """
    Save the model checkpoint.
    Args:
        args: Arguments containing configuration and training settings.
        state (dict): State dictionary containing model, optimizer, and scheduler states.
        is_best (bool): True if the current model is the best model.
    """
    
    folder = os.path.join(args.base_path, args.save_folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, args.save_last_file)
    model_best_pth_tar = os.path.join(folder, args.save_best_file)
    torch.save(state, filename)
    if is_best:
        print("Saving best model")
        shutil.copyfile(filename, model_best_pth_tar)
        
def save_args_to_txt(args, filename):
    folder = os.path.join(args.base_path, args.save_folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(filename, 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    
def train_one_epoch(args, image_encoder, lgr_head, text_embed, optimizer, scheduler, train_loader, epoch):
    """
    Train the model for one epoch.
    
    Args:
        args: Arguments containing configuration and training settings.
        image_encoder (nn.Module): Pre-trained image encoder.
        lgr_head (LanguageGuidedRecognitionHead): Language-Guided Recognition head.
        text_embed (torch.Tensor): Text embeddings tensor.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        train_loader (DataLoader): Training data loader.
        epoch (int): Current epoch number.
        
    Returns:
        tuple: Tuple containing average loss and accuracy for the epoch.
    """
    image_encoder.train()
    lgr_head.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Iterate over the training data
    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", ncols=100):
        # Move inputs and labels to GPU
        images = images.cuda(args.gpu)
        labels = labels.cuda(args.gpu)
        
        # Extract image embeddings using the pre-trained image encoder
        image_embed = image_encoder(images)
        # Forward pass through the LGR head
        logits = lgr_head(image_embed, text_embed)
        
        # Compute loss using cross-entropy
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    # Step the scheduler after all batches
    scheduler.step()
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total * 100
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return avg_loss, accuracy

def validate(args, image_encoder, lgr_head, text_embed, val_loader, epoch):
    """
    Validate the model on the validation set.
    
    Args:
        args: Arguments containing configuration and valdiation settings.
        image_encoder (nn.Module): Pre-trained image encoder.
        lgr_head (LanguageGuidedRecognitionHead): Language-Guided Recognition head.
        text_embed (torch.Tensor): Text embeddings tensor.
        val_loader (DataLoader): Validation data loader.
        epoch (int): Current epoch number.
        
    Returns:
        tuple: Tuple containing average loss and accuracy for the epoch.                       
    """
    image_encoder.eval()
    lgr_head.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # For confusion matrix
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        # Iterate over the validation data
        for images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", ncols=100):
            # Move inputs and labels to GPU
            images = images.cuda(args.gpu)
            labels = labels.cuda(args.gpu)
            
            # Extract image embeddings using the pre-trained image encoder
            image_embed = image_encoder(images)
            
            # Forward pass through the LGR head
            logits = lgr_head(image_embed, text_embed)
            
            # Compute loss using cross-entropy
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total * 100
    print(f"Validation Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    if args.validate:
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Save confusion matrix plot
        plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Class1','Class2','Class3'], yticklabels=['Class1','Class2','Class3'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        save_folder_path = os.path.dirname(args.resume)
        plt.savefig(os.path.join(args.base_path, save_folder_path, f'confusion_matrix.png'))
        plt.close()
        
        print("Confusion matrix saved")

    torch.cuda.empty_cache()
    
    return avg_loss, accuracy

def main():
    """
    Main function to initialize the model and training/validation loops.
    """
    args = parser.parse_args()
    
    if args.resume is not None and 'full' in args.resume:
        args.full = True
    
    # Load text embeddings
    text_embed_path = os.path.join(args.base_path, args.text_embed_path)
    print(f"Loading text embeddings from {text_embed_path}")
    with open(args.text_embed_path, "rb") as f:
        text_embeddings = pickle.load(f)
    text_embeddings = text_embeddings.cuda(args.gpu)
    
    # Initialize image encoder
    image_encoder = EfficientNet.from_pretrained(args.arch, weights_path=args.ft_img_encoder_path, num_classes=3)
    num_features = image_encoder._fc.in_features
    image_encoder._fc = nn.Linear(num_features, args.embed_dim)
    image_encoder = image_encoder.cuda(args.gpu)
    
    # Initialize LGR head
    lgr_head = LanguageGuidedRecognitionHead(args.embed_dim, args.num_classes, args.init_tau)
    lgr_head = lgr_head.cuda(args.gpu)
    
    # Define optimizer
    if args.full:
        optimizer = optim.Adam(list(image_encoder.parameters()) + list(lgr_head.parameters()), lr=args.lr)
    else:
        optimizer = optim.Adam(list(image_encoder._fc.parameters()) + list(lgr_head.parameters()), lr=args.lr)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.eta_min)
    
    # Load datasets and DataLoaders
    normalize = transforms.Normalize(mean=[0.4112, 0.3806, 0.4069], std=[0.2050, 0.1862, 0.1844])
    train_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(image_size),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
        ])
    val_transforms = transforms.Compose([
        # transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        # transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    
    train_csv_file = os.path.join(args.base_path, args.train_csv_file)
    val_csv_file = os.path.join(args.base_path, args.val_csv_file)
    
    train_df = pd.read_csv(train_csv_file)
    val_df = pd.read_csv(val_csv_file)
    
    train_dataset = FTDataset(train_df, train_transforms)
    val_dataset = FTDataset(val_df, val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    # Freeze all parameters in the image_encoder
    if args.full:
        for param in image_encoder.parameters():
            param.requires_grad = True
    else:
        for param in image_encoder.parameters():
            param.requires_grad = False
        for param in image_encoder._fc.parameters():
            param.requires_grad = True
    
    # Resume training
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                print("No GPU ID provided")
                exit(0)
            elif torch.cuda.is_available():
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            image_encoder.load_state_dict(checkpoint['image_encoder_state_dict'])
            lgr_head.load_state_dict(checkpoint['lgr_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit(0)
    
    if args.validate:
        test_loss, test_acc = validate(args, image_encoder, lgr_head, text_embeddings, val_loader, 0)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        return
    
    save_args_to_txt(args, os.path.join(args.base_path, args.save_folder, 'args.txt'))
    
    # Training loop
    best_acc1 = 0.0
    best_epoch = 0
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(args, image_encoder, lgr_head, text_embeddings, optimizer, scheduler, train_loader, epoch)
        val_loss, val_acc = validate(args, image_encoder, lgr_head, text_embeddings, val_loader, epoch)
        
        # Save the best model
        if not args.validate:
            print("Saving checkpoint")
            is_best = val_acc >= best_acc1
            if val_acc >= best_acc1:
                best_acc1 = val_acc
                best_epoch = epoch

            save_checkpoint(args, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'best_acc1': best_acc1,
                'image_encoder_state_dict': image_encoder.state_dict(),
                'lgr_state_dict': lgr_head.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best)
            
        print(f"Epoch {epoch+1} completed: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("")
    print(f"Best validation accuracy: {best_acc1:.2f}% at epoch {best_epoch+1}")

if __name__ == '__main__':
    main()