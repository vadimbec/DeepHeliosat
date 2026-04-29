"""
Training script for the FCN baseline model.

Fixed crop size (33×33 pixels, ~50 km footprint at GOES-16 resolution).
Runs 5 independent training runs to assess variability.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from models import FCN, RMSELoss
from dataset import SatelliteDataset
from utils import train_norm, validate_norm, EarlyStopping

# ── Data ──────────────────────────────────────────────────────────────────────

df = pd.read_csv('/mnt/m0/vadim.becquet/Code/Dataset/FullDataset_all_corrected_roundagg.csv',
                 parse_dates=True, index_col=0)
df['file_path_small'] = df['file_path'].str.replace('Images', 'Images_small')
df = df[(df['QC'] == False) | (df['station_id'].str.startswith('SOLRAD'))]
df = df[df['solar_zenith'] <= np.deg2rad(80)]
df = df[df['GHI'] > 0]
df = df[df['GHI'] <= df['TOA_HI']]
df.drop(columns=['QC_NC'], inplace=True)

albedo_df = pd.read_csv('central_albedo_values.csv', parse_dates=['timestamp'])
df.reset_index(inplace=True)
if 'index' in df.columns:
    df.rename(columns={'index': 'timestamp'}, inplace=True)
df = pd.merge(df, albedo_df, how='left', on=['station_id', 'timestamp'])
df.set_index('timestamp', inplace=True)
df.dropna(inplace=True)

df['BHI_corrected'] = df['BNI_corrected'] * np.cos(df['solar_zenith'])

features = [
    'latitude', 'longitude', 'elevation', 'solar_zenith', 'solar_azimuth',
    'TOA_HI', 'ghi_clear', 'dhi_clear', 'bhi_clear', 'dni_clear',
    'M6C01', 'M6C02', 'M6C03', 'M6C04', 'M6C05', 'M6C06', 'M6C13',
    'ws_albedo', 'bs_albedo',
    'M6C01_kappa0', 'M6C02_kappa0', 'M6C03_kappa0', 'M6C04_kappa0',
    'M6C05_kappa0', 'M6C06_kappa0',
    'M6C13_planck_fk1', 'M6C13_planck_fk2', 'M6C13_planck_bc1', 'M6C13_planck_bc2',
]
target   = ['GHI_corrected', 'kc_corrected']
channels = ['M6C01', 'M6C02', 'M6C03', 'M6C04', 'M6C05', 'M6C06', 'M6C13']

test_ids = ['NREL_MIDC-UOSMRL', 'NREL_MIDC-UTPASRL', 'BSRN-LRC',
            'NREL_MIDC-UAT', 'SOLRAD-BIS', 'SOLRAD-STE']

train_data = df[(~df['station_id'].isin(test_ids)) & (df.index.year <= 2021)]
test_data  = df[(df['station_id'].isin(test_ids))  & (df.index.year == 2022)]

print('Train stations:', train_data['station_id'].unique().tolist())
print('Test  stations:', test_data['station_id'].unique().tolist())

# Normalisation
X_train      = train_data[features]
y_train      = train_data[target]
X_train_max  = abs(X_train).max()
y_train_max  = abs(y_train).max()
X_train_norm = X_train / X_train_max
y_train_norm = y_train / y_train_max

X_train_tensor = torch.tensor(X_train_norm.values, dtype=torch.float32)

X_train_names = np.array(train_data.file_path_small.tolist())
orig_size     = 33

# Dataset & DataLoaders
augmentation = T.Compose([T.RandomRotation(degrees=180)])
train_dataset = SatelliteDataset(
    X_train_names, X_train_tensor, y_train_norm,
    img_norm=X_train_max[channels],
    transform=augmentation, orig_size=orig_size, num_workers=64
)
VALIDATION_SPLIT = 0.1
train_size = int((1 - VALIDATION_SPLIT) * len(train_dataset))
valid_size = len(train_dataset) - train_size
train_subset, valid_subset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

BATCH = 512
train_dl = DataLoader(train_subset, batch_size=BATCH, shuffle=True,  num_workers=16)
valid_dl = DataLoader(valid_subset, batch_size=BATCH, shuffle=False, num_workers=4)

print(f'Train: {len(train_subset)} samples | Valid: {len(valid_subset)} samples')

# ── Training loop ─────────────────────────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CROP   = orig_size   # FCN uses the full image (no crop reduction)

for run in range(5):
    print(f'\n=== Run {run} ===')
    MODEL_PATH = f'/mnt/m0/vadim.becquet/Code/models/fcn_crop{CROP}_run{run}.pth'

    model     = FCN(crop=CROP, n_tab=len(features), n_channels=9).float().to(DEVICE)
    criterion = RMSELoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    stopper   = EarlyStopping(patience=20)

    for epoch in range(60):
        tr_kc, tr_ghi, _ = train_norm(model, train_dl, X_train_max, y_train_max, criterion, optimizer, DEVICE)
        va_kc, va_ghi, _ = validate_norm(model, valid_dl, X_train_max, y_train_max, criterion, DEVICE)
        print(f'Epoch {epoch+1:3d} | train kc {tr_kc:.4f} ghi {tr_ghi:.2f} | '
              f'val kc {va_kc:.4f} ghi {va_ghi:.2f}')
        stopper(va_ghi, model, MODEL_PATH)
        if stopper.stop:
            print('Early stopping.')
            break
