import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from Model.IGLCFN import IGLCFN
from dataset import IC9600Dataset


def calculate_metrics(labels, scores):
    score = np.array(scores)
    label = np.array(labels)

    MAE = np.sqrt(np.abs(score - label).mean())
    MSE = np.sqrt(np.mean(np.abs(score - label) ** 2))
    Pearson = np.float64(pearsonr(label, score)[0])
    Spearmanr = np.float64(spearmanr(label, score)[0])
    tx = score
    ty = label
    ccc = (2 * Pearson * tx.std() * ty.std()) / (np.power(tx.mean() - ty.mean(), 2) + tx.var() + ty.var())
    R2 = r2_score(label, score)

    return {
        "MSE": MSE,
        "MAE": MAE,
        "PCC (Pearson)": Pearson,
        "SRCC (Spearman)": Spearmanr,
        "CCC": ccc,
        "R2": R2
    }


def main_test(weights_path: str = './checkpoint/ck_best'):
    # --- 1. Setup Device and Paths ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_ROOT = './IC9600/'
    BATCH_SIZE = 16
    NUM_WORKERS = 0

    print(f"Using device: {device}")

    # --- 2. Prepare Dataset and DataLoader ---
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = IC9600Dataset(
        data_root=DATA_ROOT,
        list_file_name='test.txt',
        transform=test_transform
    )

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=BATCH_SIZE,
                                 num_workers=NUM_WORKERS,
                                 shuffle=False)

    # --- 3. Instantiate Model and Load Weights ---
    model = IGLCFN()
    model.to(device)

    if os.path.exists(weights_path):
        print(f"Loading weights file: {weights_path}")
        try:
            state_dict = torch.load(weights_path, map_location=device)

            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            model.load_state_dict(state_dict, strict=False)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load weights, using randomly initialized model. Error: {e}")
    else:
        print(f"Warning: Weights file {weights_path} not found. Testing with untrained model.")

    model.eval()

    # --- 4. Prediction Loop ---
    all_scores = []
    all_labels = []

    print("\nStarting prediction on the test set...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_dataloader):
            images = images.to(device)

            score= model(images)

            all_scores.extend(score.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (i + 1) % 10 == 0:
                print(f"Processed {((i + 1) * BATCH_SIZE)}/{len(test_dataset)} samples")

    print("Prediction complete.")

    # --- 5. Calculate and Print Metrics ---
    if len(all_scores) > 1:
        metrics = calculate_metrics(all_labels, all_scores)

        print("\n========================================")
        print("📊 Model Prediction Evaluation Metrics")
        print("========================================")
        print(f"Total Test Samples: {len(all_labels)}")
        print(f"  MSE (Mean Squared Error):       {metrics['MSE']:.4f}")
        print(f"  MAE (Mean Absolute Error):   {metrics['MAE']:.4f}")
        print(f"  PCC (Pearson Correlation):     {metrics['PCC (Pearson)']:.4f}")
        print(f"  SRCC (Spearman Correlation):  {metrics['SRCC (Spearman)']:.4f}")
        print(f"  CCC (Concordance Correlation):     {metrics['CCC']:.4f}")
        print(f"  R2 ($R^2$):           {metrics['R2']:.4f}")
        print("========================================")
    else:
        print("Error: Insufficient number of samples to calculate metrics.")


if __name__ == "__main__":
    main_test(weights_path='./checkpoints/IGLCFN.pth')