"""
Femtech Privacy Policy Deception Detection - BERT Classification
================================================================
This module implements BERT-based classification for privacy policy sentences
from reproductive health applications. Addresses class imbalance through
inverse frequency weighting.

Author: Alexis Prieto
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
CONFIG = {
    'model_name': 'bert-base-uncased',
    'max_length': 128,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'epochs': 3,
    'train_split': 0.8,
    'random_seed': 42,
    'num_labels': 9,
    'label_names': [
        'Data Collection', 'Data Sharing', 'Data Retention', 'User Rights',
        'Security Measures', 'Third Party', 'Policy Changes', 'Contact Info',
        'Legal Compliance'
    ]
}

class PrivacyPolicyDataset(Dataset):
    """Custom Dataset for privacy policy sentences."""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def compute_class_weights(labels, num_classes):
    """
    Compute inverse frequency class weights to handle imbalance.
    
    The femtech dataset has severe imbalance:
    - Data Collection: 563 samples (37.1%)
    - Legal Compliance: 20 samples (1.3%)
    - Ratio: 9.8x imbalance
    
    This weighting penalizes errors on minority classes more heavily.
    """
    label_counts = Counter(labels)
    total_samples = len(labels)
    
    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float)


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy, predictions, true_labels


def plot_confusion_matrix(true_labels, predictions, label_names, save_path):
    """Generate and save confusion matrix visualization."""
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('BERT Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path):
    """Plot training and validation metrics over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, train_accs, 'b-', label='Training Accuracy')
    axes[1].plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training history saved to {save_path}")


def main(data_path, output_dir='./results'):
    """
    Main training pipeline.
    
    Args:
        data_path: Path to CSV with 'sentence' and 'label' columns
        output_dir: Directory for saving results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    texts = df['sentence'].tolist()
    labels = df['label'].tolist()
    
    print(f"Total samples: {len(texts)}")
    print(f"Label distribution: {Counter(labels)}")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, 
        test_size=1-CONFIG['train_split'],
        random_state=CONFIG['random_seed'],
        stratify=labels
    )
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    
    # Initialize tokenizer and model
    print("Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])
    model = BertForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=CONFIG['num_labels']
    )
    model.to(device)
    
    # Create datasets
    train_dataset = PrivacyPolicyDataset(
        train_texts, train_labels, tokenizer, CONFIG['max_length']
    )
    val_dataset = PrivacyPolicyDataset(
        val_texts, val_labels, tokenizer, CONFIG['max_length']
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'], shuffle=False
    )
    
    # Compute class weights for handling imbalance
    class_weights = compute_class_weights(train_labels, CONFIG['num_labels'])
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights}")
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    total_steps = len(train_loader) * CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    print("\nStarting training...")
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            print(f"New best model saved with accuracy: {val_acc:.4f}")
    
    # Final evaluation
    print("\nFinal Evaluation...")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))
    _, final_acc, predictions, true_labels = evaluate(
        model, val_loader, criterion, device
    )
    
    print(f"\nFinal Accuracy: {final_acc:.4f} ({final_acc*100:.1f}%)")
    print("\nClassification Report:")
    print(classification_report(
        true_labels, predictions, 
        target_names=CONFIG['label_names'],
        zero_division=0
    ))
    
    # Generate visualizations
    plot_confusion_matrix(
        true_labels, predictions, CONFIG['label_names'],
        os.path.join(output_dir, 'confusion_matrix.png')
    )
    plot_training_history(
        train_losses, train_accs, val_losses, val_accs,
        os.path.join(output_dir, 'training_history.png')
    )
    
    # Save results summary
    results = {
        'final_accuracy': final_acc,
        'best_val_accuracy': best_val_acc,
        'train_samples': len(train_texts),
        'val_samples': len(val_texts),
        'config': CONFIG
    }
    
    with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nResults saved to {output_dir}")
    return model, final_acc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train BERT classifier for privacy policies')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--output', type=str, default='./results', help='Output directory')
    args = parser.parse_args()
    
    main(args.data, args.output)
