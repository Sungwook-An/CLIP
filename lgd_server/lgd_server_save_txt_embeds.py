import argparse
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
import pickle

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

parser = argparse.ArgumentParser(description='Text Embedding Extraction')
parser.add_argument('--data', metavar='DIR', default='/', help='path to dataset')
parser.add_argument('--base_path', metavar='DIR', default='./', help='path to dataset')
parser.add_argument('--ft_pretrained', metavar='DIR', default='save_models_ft/classwise_30_lr1e-7~1e-9_epochs10_projDim512_94.667/model_best.pth.tar', help='path to dataset')
parser.add_argument('--save_name', metavar='DIR', default='text_embeddings_cw30_lr7~9_ep10_pd512.pkl', help='path to dataset')

args = parser.parse_args()

MODEL = 'openai/clip-vit-large-patch14'

# Define tokenizer and Transformer model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
text_encoder = CLIPTextModel.from_pretrained(MODEL)

# Load pretrained model
pretrained_path = os.path.join(args.base_path, args.ft_pretrained)
pretrained_model = torch.load(pretrained_path)

# Load state_dict
state_dict = pretrained_model['text_encoder_state_dict']
text_encoder.load_state_dict(state_dict)

# Freeze model
text_encoder.eval()

# Class texts
class_texts = ["A picture of first class", "A picture of second class", "A picture of third class"]

# Text tokenize & GPU load (option)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_encoder = text_encoder.to(device)
tokenized = tokenizer(class_texts, padding=True, truncation=True, return_tensors="pt")
tokenized = {key: val.to(device) for key, val in tokenized.items()}

# Generate text embeddings
with torch.no_grad():
    transformer_outputs = text_encoder(**tokenized)
    text_embeddings = transformer_outputs.last_hidden_state.mean(dim=1)  # Shape: (3, D)

# Save text embeddings
text_embeddings = text_embeddings.cpu()
output_dir = os.path.join(DIR, "pkl")

os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, args.save_name)

with open(output_path, "wb") as f:
    pickle.dump(text_embeddings, f)

print("")
print("")
print("Text embeddings saved successfully!")
