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
parser.add_argument('--second_pretrained', metavar='DIR', default='save_models/classwise_128_lr1e-7~1e-9_epochs10_projDim512_99.115/model_best.pth.tar', help='path to dataset')

args = parser.parse_args()

MODEL = 'openai/clip-vit-large-patch14'

# Define tokenizer and Transformer model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
text_encoder = CLIPTextModel.from_pretrained(MODEL)

# Load pretrained model
pretrained_path = os.path.join(args.base_path, args.second_pretrained)
pretrained_model = torch.load(pretrained_path)

# Load state_dict
state_dict = pretrained_model['text_encoder_state_dict']
text_encoder.load_state_dict(state_dict)

# Freeze model
text_encoder.eval()

# Class texts
class_texts = ["A picture of class1", "A picture of class2", "A picture of class3"]

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
with open("text_embeddings.pkl", "wb") as f:
    pickle.dump(text_embeddings, f)

print("")
print("")
print("Text embeddings saved successfully!")
