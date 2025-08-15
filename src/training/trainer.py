import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging
import yaml
import joblib
import pathlib as pl
import glob
import subprocess

def train_model(model, dloader, num_epochs=25, lr=1e-4, checkpoint_path="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in tqdm(dloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.8f}")

        # Save best model checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, checkpoint_path)
            print(f"✅ Saved new best model at epoch {epoch+1} with loss {best_loss:.8f}")

    print("Training complete.")
    return model

def evaluate_model(model, tloader, device='cpu'):
    """
    Evaluate the model using a PyTorch DataLoader.
    
    Args:
        model: trained model
        tloader: DataLoader object yielding (input, placeholder_target)
        device: device to run evaluation on ('cuda' or 'cpu')
    
    Returns:
        preds: torch.Tensor of shape (N, C, H, W) where N is number of sequences
    """
    model.eval()
    model.to(device)

    all_preds = []

    with torch.no_grad():
        for xb, _ in tqdm(tloader):
            xb = xb.to(device)  # xb shape: (B=1, T, C, H, W)
            
            output = model(xb)  # should return (B=1, T=1, C, H, W)
            if isinstance(output, (list, tuple)):
                output = output[0]  # output[0] if model returns a tuple/list
            
            output = output[:, -1] 
            all_preds.append(output.cpu())

    preds = torch.cat(all_preds, dim=0)  # shape: (N, C, H, W)
    return preds
