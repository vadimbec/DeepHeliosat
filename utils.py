import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
import random
import gc
import os
import re
import warnings
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

warnings.filterwarnings('ignore', category=FutureWarning)
sns.set(style="whitegrid", palette=sns.color_palette("hls", 10), font_scale=1.2)


# ── Training ──────────────────────────────────────────────────────────────────

def move_inputs_to_device(inputs, device):
    sat_data, tab_data = inputs
    return sat_data.to(device), tab_data.to(device)


def train_norm(model, dataloader, X_train_max, y_train_max, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_ghi_loss = 0.0
    running_ghi_norm_loss = 0.0
    for data in tqdm(dataloader, desc="Training"):
        inputs, labels, GHI_norm = data[:2], data[2], data[3]
        inputs = move_inputs_to_device(inputs, device)
        labels = labels.to(device).float()
        GHI_norm = GHI_norm.to(device).float()
        GHI = GHI_norm * y_train_max['GHI_corrected']

        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)
        ghi_outputs = (outputs * y_train_max['kc_corrected']) * (inputs[1][:, 6] * X_train_max['ghi_clear'])
        ghi_outputs_norm = ghi_outputs / y_train_max['GHI_corrected']

        loss = criterion(outputs * y_train_max['kc_corrected'], labels)
        ghi_norm_loss = criterion(ghi_outputs_norm, GHI_norm)
        ghi_loss = criterion(ghi_outputs, GHI)

        ghi_norm_loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(labels)
        running_ghi_loss += ghi_loss.item() * len(GHI)
        running_ghi_norm_loss += ghi_norm_loss.item() * len(GHI_norm)
    n = len(dataloader.dataset)
    return running_loss / n, running_ghi_loss / n, running_ghi_norm_loss / n


def validate_norm(model, dataloader, X_train_max, y_train_max, criterion, device):
    model.eval()
    running_loss = 0.0
    running_ghi_loss = 0.0
    running_ghi_norm_loss = 0.0
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Validating"):
            inputs, labels, GHI_norm = data[:2], data[2], data[3]
            inputs = move_inputs_to_device(inputs, device)
            labels = labels.to(device).float()
            GHI_norm = GHI_norm.to(device).float()
            GHI = GHI_norm * y_train_max['GHI_corrected']

            outputs = model(inputs).squeeze(1)
            ghi_outputs = (outputs * y_train_max['kc_corrected']) * (inputs[1][:, 6] * X_train_max['ghi_clear'])
            ghi_outputs_norm = ghi_outputs / y_train_max['GHI_corrected']

            loss = criterion(outputs, labels)
            ghi_norm_loss = criterion(ghi_outputs_norm, GHI_norm)
            ghi_loss = criterion(ghi_outputs, GHI)

            running_loss += loss.item() * len(labels)
            running_ghi_loss += ghi_loss.item() * len(GHI)
            running_ghi_norm_loss += ghi_norm_loss.item() * len(GHI_norm)
    n = len(dataloader.dataset)
    return running_loss / n, running_ghi_loss / n, running_ghi_norm_loss / n


def evaluate_model_maxnorm(model, dataloader, X_train_max, y_train_max, device):
    model.eval()
    all_preds, all_labels = [], []
    all_preds_kc, all_labels_kc = [], []
    with torch.no_grad():
        for data in dataloader:
            inputs, labels, GHI_norm = data[:2], data[2], data[3]
            inputs = move_inputs_to_device(inputs, device)
            labels = labels.to(device)
            GHI_norm = GHI_norm.to(device).float()
            GHI = GHI_norm * y_train_max['GHI_corrected']
            kc_labels = labels.float() * y_train_max['kc_corrected']
            kc_outputs = model(inputs).squeeze(1) * y_train_max['kc_corrected']
            ghi_outputs = kc_outputs * (inputs[1][:, 6] * X_train_max['ghi_clear'])

            all_preds.append(ghi_outputs)
            all_labels.append(GHI)
            all_preds_kc.append(kc_outputs)
            all_labels_kc.append(kc_labels)
    return (torch.cat(all_preds), torch.cat(all_labels),
            torch.cat(all_preds_kc), torch.cat(all_labels_kc))


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.stop = False

    def __call__(self, val_loss, model, model_path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(model, model_path)
        elif score < self.best_score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self._save(model, model_path)
            self.counter = 0

    def _save(self, model, model_path):
        torch.save(model.state_dict(), model_path)
        print("Saved new checkpoint.")


# ── Evaluation metrics ────────────────────────────────────────────────────────

def compute_station_metrics(group, true_col, pred_col, psm3_col, station_id):
    errors = group[pred_col] - group[true_col]
    errors_psm3 = group[psm3_col] - group[true_col]
    mean_true = np.mean(group[true_col])

    corr = group[[true_col, pred_col]].corr().iloc[0, 1]
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.abs(errors).mean()
    mbe = np.array(errors).mean()
    sde = np.std(errors)

    corr_psm3 = group[[true_col, psm3_col]].corr().iloc[0, 1]
    rmse_psm3 = np.sqrt(np.mean(errors_psm3**2))
    mae_psm3 = np.abs(errors_psm3).mean()
    mbe_psm3 = np.array(errors_psm3).mean()
    sde_psm3 = np.std(errors_psm3)

    return {
        'station_id': station_id,
        'Correlation': corr,
        'RMSE': rmse, 'nRMSE': 100 * rmse / mean_true,
        'MAE': mae,  'nMAE':  100 * mae  / mean_true,
        'MBE': mbe,  'nMBE':  100 * mbe  / mean_true,
        'Abs(MBE)': abs(mbe), 'Abs(nMBE)': abs(100 * mbe / mean_true),
        'SDE': sde,  'nSDE':  100 * sde  / mean_true,
        'Skill score RMSE': 100 * (rmse_psm3 - rmse) / rmse_psm3,
        'Skill score MAE':  100 * (mae_psm3  - mae)  / mae_psm3,
        'Skill score Abs(MBE)': 100 * (abs(mbe_psm3) - abs(mbe)) / abs(mbe_psm3),
        'Correlation_PSM3': corr_psm3,
        'RMSE_PSM3': rmse_psm3, 'nRMSE_PSM3': 100 * rmse_psm3 / mean_true,
        'MAE_PSM3':  mae_psm3,  'nMAE_PSM3':  100 * mae_psm3  / mean_true,
        'MBE_PSM3':  mbe_psm3,  'nMBE_PSM3':  100 * mbe_psm3  / mean_true,
        'Abs(MBE)_PSM3': abs(mbe_psm3), 'Abs(nMBE)_PSM3': abs(100 * mbe_psm3 / mean_true),
        'SDE_PSM3':  sde_psm3,  'nSDE_PSM3':  100 * sde_psm3  / mean_true,
    }


def compute_metrics(df, true_col, pred_col, psm3_col):
    """Compute RMSE/MAE/MBE/SDE overall and per station, with skill scores vs PSM3."""
    errors = df[pred_col] - df[true_col]
    errors_psm3 = df[psm3_col] - df[true_col]
    mean_true = np.mean(df[true_col])

    corr = df[[true_col, pred_col]].corr().iloc[0, 1]
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.abs(errors).mean()
    mbe = np.array(errors).mean()
    sde = np.std(errors)

    rmse_psm3 = np.sqrt(np.mean(errors_psm3**2))
    mae_psm3 = np.abs(errors_psm3).mean()
    mbe_psm3 = np.array(errors_psm3).mean()

    overall = {
        'station_id': 'Overall',
        'Correlation': corr,
        'RMSE': rmse, 'nRMSE': 100 * rmse / mean_true,
        'MAE': mae,  'nMAE':  100 * mae  / mean_true,
        'MBE': mbe,  'nMBE':  100 * mbe  / mean_true,
        'Abs(MBE)': abs(mbe), 'Abs(nMBE)': abs(100 * mbe / mean_true),
        'SDE': sde,  'nSDE':  100 * sde  / mean_true,
        'Skill score RMSE': 100 * (rmse_psm3 - rmse) / rmse_psm3,
        'Skill score MAE':  100 * (mae_psm3  - mae)  / mae_psm3,
        'Skill score Abs(MBE)': 100 * (abs(mbe_psm3) - abs(mbe)) / abs(mbe_psm3),
        'RMSE_PSM3': rmse_psm3, 'nRMSE_PSM3': 100 * rmse_psm3 / mean_true,
        'MAE_PSM3':  mae_psm3,  'nMAE_PSM3':  100 * mae_psm3  / mean_true,
        'MBE_PSM3':  mbe_psm3,  'nMBE_PSM3':  100 * mbe_psm3  / mean_true,
        'Abs(MBE)_PSM3': abs(mbe_psm3),
        'SDE_PSM3': np.std(errors_psm3), 'nSDE_PSM3': 100 * np.std(errors_psm3) / mean_true,
        'Correlation_PSM3': df[[true_col, psm3_col]].corr().iloc[0, 1],
    }

    metrics_data = [overall]
    if len(df['station_id'].unique()) > 1:
        for station, group in df.groupby('station_id'):
            metrics_data.append(compute_station_metrics(group, true_col, pred_col, psm3_col, station))
    else:
        metrics_data[0]['station_id'] = df['station_id'].unique()[0]

    return pd.DataFrame(metrics_data)


# ── Visualization ─────────────────────────────────────────────────────────────

def scatterplot_single(ax, df, target_column, pred_column, title, values=[0, 800], kc=False, vmin=None, vmax=None):
    if kc:
        values = [0, 1.5]
    hexbin = ax.hexbin(df[target_column], df[pred_column], bins="log", gridsize=100,
                       cmap='YlGnBu', linewidths=0.5, vmin=vmin, vmax=vmax)
    ax.plot(values, values, linestyle='--', color='red', linewidth=2)

    errors = df[pred_column] - df[target_column]
    corr = df[[target_column, pred_column]].corr().iloc[0, 1]
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.abs(errors).mean()
    mbe = np.mean(errors)
    sde = np.std(errors)
    mean_true = np.mean(df[target_column])

    nrmse = 100 * rmse / mean_true
    nmae  = 100 * mae  / mean_true
    nmbe  = 100 * mbe  / mean_true
    nsde  = 100 * sde  / mean_true

    ax.set_title(title, fontweight='bold')
    if kc:
        ax.text(0.03, 0.85,
                f'Corr = {corr:.2f}\nnRMSE = {nrmse:.2f}%\nnMAE = {nmae:.2f}%\n'
                f'nMBE = {nmbe:.2f}%\nnSDE = {nsde:.2f}%\nN = {len(df)}',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='round,pad=0.3'), fontsize=10)
    else:
        ax.text(0.03, 0.85,
                f'Corr = {corr:.2f}\nRMSE = {rmse:.2f} W/m²\nMAE = {mae:.2f} W/m²\n'
                f'MBE = {mbe:.2f} W/m²\nSDE = {sde:.2f} W/m²\nN = {len(df)}',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='round,pad=0.3'), fontsize=10)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    return hexbin
