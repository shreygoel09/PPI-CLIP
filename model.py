# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import config
import data_utils

class CLIPLoss(nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()
    
    def forward(self, logits, labels):
        loss = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.transpose(-2, -1), labels)
        return (loss + loss_t) / 2

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.loss = nn.CosineEmbeddingLoss()
    
    def forward(self, x1, x2, targets):
        loss = self.loss(x1, x2, targets)
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, margin, x1, x2, label):
        """
        Computes Contrastive Loss
        """
        dist = torch.nn.functional.pairwise_distance(x1, x2)
        loss = (1 - label) * torch.pow(dist, 2) \
            + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
        loss = torch.mean(loss)
        return loss

# First projection of ESM embeddings is with MLP
class ProteinMLP(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_layers, dropout, output_relu):
        super(ProteinMLP, self).__init__()
        layers_list = [nn.Linear(input_dim, embedding_dim)]
        for _ in range(num_layers-1):
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Dropout(p=dropout))
            layers_list.append(nn.Linear(embedding_dim, embedding_dim))
        
        if output_relu:
            layers_list.append(nn.LeakyReLU())
        
        self.layers = nn.Sequential(*layers_list)

    def forward(self, input_embedding, padding_mask=None):
        if padding_mask is not None:
            non_pads = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
            input_embedding = input_embedding.sum(dim=1) / non_pads
            
            if torch.isnan(input_embedding).any() or torch.isinf(input_embedding).any():
                print("nan or inf present after padding mask transformation")
                sys.stdout.flush()

        return self.layers(input_embedding)


class ProteinCLIP(nn.Module):
    def __init__(self):
        super(ProteinCLIP, self).__init__()
        # Project input ESM embeddings with MLP
        self.seq1_mlp = ProteinMLP(
            input_dim=1280,
            embedding_dim=1280,
            num_layers=config.MLP_LAYERS,
            dropout=config.MLP_DROPOUT,
            output_relu=config.OUTPUT_RELU
        )
        self.seq2_mlp = ProteinMLP(
            input_dim=1280,
            embedding_dim=1280,
            num_layers=config.MLP_LAYERS,
            dropout=config.MLP_DROPOUT,
            output_relu=config.OUTPUT_RELU
        )

        # Pass MLP outputs to transformer
        protein_transformer_layer = nn.TransformerEncoderLayer(
            d_model=1280,
            nhead=config.TRANSFORMER_HEADS,
            dropout=config.TRANSFORMER_DROPOUT,
            batch_first=True
        )
        self.protein_transformer = nn.TransformerEncoder(protein_transformer_layer, num_layers=config.TRANSFORMER_LAYERS)

        # Transformer to MLP once more
        self.embedding1_mlp = ProteinMLP(
            input_dim=1280,
            embedding_dim=1280,
            num_layers=config.MLP_LAYERS,
            dropout=config.MLP_DROPOUT,
            output_relu=config.OUTPUT_RELU
        )
        self.embedding2_mlp = ProteinMLP(
            input_dim=1280,
            embedding_dim=1280,
            num_layers=config.MLP_LAYERS,
            dropout=config.MLP_DROPOUT,
            output_relu=config.OUTPUT_RELU
        )

        # Learn temperature parameter
        temp = torch.full((1,), float(config.INIT_TEMP))
        self.temperature = nn.Parameter(temp, requires_grad=True)

    
    def forward(self, seq1_input, seq2_input, seq1_attn_mask, seq2_attn_mask):
        # MLP to inputs for nonlinearity
        seq1_embed = self.seq1_mlp(seq1_input)
        seq2_embed = self.seq2_mlp(seq2_input)

        # Transformer encoder layers
        seq1_embed = self.protein_transformer(seq1_input, src_key_padding_mask=seq1_attn_mask)
        seq2_embed = self.protein_transformer(seq2_input, src_key_padding_mask=seq2_attn_mask)

        # MLP on transformer outputs again
        seq1_embed = self.embedding1_mlp(seq1_embed, padding_mask=seq1_attn_mask)
        seq2_embed = self.embedding2_mlp(seq2_embed, padding_mask=seq2_attn_mask)

        # Final CLIP layer (normalization), add eps factor to prevent div by 0
        seq1_embed = F.normalize(seq1_embed + 1e-8, dim=1)
        seq2_embed = F.normalize(seq2_embed + 1e-8, dim=1)

        # Pairwise cosine similarity scaled by temperature parameter
        logits = torch.matmul(seq1_embed, seq2_embed.transpose(-2, -1)) * torch.exp(self.temperature)

        return logits

# Train function
def train(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0

    total_steps = len(train_loader)
    update_interval = total_steps // 4
    progbar = tqdm(total=total_steps, leave=True, file=sys.stdout)

    for step, batch in enumerate(train_loader):
        seq1_latents_padded, seq2_latents_padded, seq1_attn_mask_padded, seq2_attn_mask_padded, labels = batch
        labels = labels.type(torch.LongTensor)
        seq1_latents_padded, seq2_latents_padded, seq1_attn_mask_padded, seq2_attn_mask_padded, labels = seq1_latents_padded.to(device), seq2_latents_padded.to(device), seq1_attn_mask_padded.to(device), seq2_attn_mask_padded.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast():
            logits = model(seq1_latents_padded, seq2_latents_padded, seq1_attn_mask_padded, seq2_attn_mask_padded)
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if step % update_interval == 0 or step == total_steps:
            progbar.update(update_interval)
            sys.stdout.flush()
        
    progbar.close()
    avg_train_loss = total_loss / len(train_loader)
    return avg_train_loss


def validation(model, criterion, val_loader, device):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(val_loader):
            seq1_latents_padded, seq2_latents_padded, seq1_attn_mask_padded, seq2_attn_mask_padded, labels = batch
            seq1_latents_padded, seq2_latents_padded, seq1_attn_mask_padded, seq2_attn_mask_padded, labels = seq1_latents_padded.to(device), seq2_latents_padded.to(device), seq1_attn_mask_padded.to(device), seq2_attn_mask_padded.to(device), labels.to(device)
            logits = model(seq1_latents_padded, seq2_latents_padded, seq1_attn_mask_padded, seq2_attn_mask_padded)
            val_loss += criterion(logits, labels).item()
    
    avg_val_loss = val_loss / len(val_loader)
    model.train()
    return avg_val_loss


# Test function
def evaluate(model, test_loader, device):
    model.eval()
    preds, true_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            embeddings, labels = batch
            embeddings, labels = embeddings.to(device), labels.to(device)
            logits = model(embeddings)
            preds.append(logits.cpu().numpy())
            labels.append(logits.cpu().numpy())
        
    return preds, true_labels


# Metrics function
def calc_metrics(preds, labels, threshold):
    flat_binary_preds, flat_prob_preds, flat_labels = [], [], []

    for pred, label in zip(preds, labels):
        flat_binary_preds.extend((pred > threshold).astype(int).flatten())
        flat_prob_preds.extend(pred.flatten())
        flat_labels.extend(label.flatten())

    flat_binary_preds = np.array(flat_binary_preds)
    flat_prob_preds = np.array(flat_prob_preds)
    flat_labels = np.array(flat_labels)

    accuracy = accuracy_score(flat_labels, flat_binary_preds)
    precision = precision_score(flat_labels, flat_binary_preds)
    recall = recall_score(flat_labels, flat_binary_preds)
    f1 = f1_score(flat_labels, flat_binary_preds)
    roc_auc = roc_auc_score(flat_labels, flat_prob_preds)

    return accuracy, precision, recall, f1, roc_auc