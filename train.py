import argparse
from scheduler import cosine_schedule, optimizer, base_lrs
from dataloader import train_loader, val_loader
from model.shvit_enhancement import SHViTEnhanced
import time
import torch
import torch.nn as nn
import os

def parse():
    parser = argparse.ArgumentParser(description="Train SHVIT on COCO dataset")
    parser.add_argument(
        "--epochs", type=int, default=150, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="shvit_model.pth",
        help="Path to save the trained model",
    )
    return parser.parse_args()  


print ("✅Argument parser for training initialized ! ")
args = parse()
print ("✅Argument parser for training done ! ")
# --------------------------
# Training one epoch function

def train_one_epoch(model, dataloader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        # Move labels to device if necessary
        # Assuming labels is a list of lists, we keep it on CPU for loss calculation
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch {epoch}, Training Loss: {epoch_loss:.4f}")
    return epoch_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            # Move labels to device if necessary
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0) 
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss = val_running_loss / len(val_loader)
    val_acc = 100 * correct / total
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
    return val_loss, val_acc


def main ():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SHViTEnhanced().to(device)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Adjust learning rate
        current_lrs = cosine_schedule(epoch, optimizer, base_lrs, warmup_epochs=10, max_epochs=args.epochs)
        print(f"Epoch {epoch}, Learning Rates: {current_lrs}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds.")
        
        print(f"Epoch {epoch} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_save_path)
            print(f"Model saved at epoch {epoch} with validation loss {val_loss:.4f}")  
    print("Training completed.")
    
if __name__ == "__main__":
    main()
      