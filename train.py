import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import ProteinCLIP, ContrastiveLoss, CosineSimilarityLoss, CLIPLoss
from model import train, validation
from data_utils import get_dataloaders
import config
import sys
import os

def train_and_val(train_loader, val_loader, criterion, device, model):
    best_val_loss = 0
    for epoch in range(num_epochs):
        # Train the model
        print(f"EPOCH {epoch+1}/{num_epochs}")
        train_loss = train(model, optimizer, criterion, train_loader, device)
        print(f"TRAIN LOSS: {train_loss}")
        sys.stdout.flush()

        # Save checkpoints at every epoch
        curr_ckpt_path = os.path.join(config.CKPT_DIR, f'ckpt_{epoch+1}.pth')
        torch.save(model.state_dict(), curr_ckpt_path)

        # Evaluate trained model on validation dataset
        val_loss = validation(model, criterion, val_loader, device)
        print(f"VALIDATION LOSS: {val_loss}")
        sys.stdout.flush()

        # Save model if it is more performant on validation dataset
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(config.PATH, 'best_model.pth')
            torch.save(model.state_dict(), best_ckpt_path)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProteinCLIP().to(device)
    model = nn.DataParallel(model)
    optimizer = optim.Adam(params=model.parameters(), lr=config.LR)
    num_epochs = config.EPOCHS

    train_loader, val_loader, _ = get_dataloaders(config)

    if config.LOSS_TYPE == "contrastive":
        criterion = ContrastiveLoss(margin=1.0)
    elif config.LOSS_TYPE == "cosine":
        criterion = CosineSimilarityLoss()
    elif config.LOSS_TYPE == "bce_avg":
        criterion = CLIPLoss()
    else:
        raise ValueError("Loss must be either 'contrastive', 'cosine', or 'bce_avg'")

    torch.cuda.empty_cache()
    train_and_val(train_loader, val_loader, criterion, device, model)

