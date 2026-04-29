import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from multiprocessing import Pool
from tqdm import tqdm


def load_file(filename):
    """Load GOES-16 satellite channels from a .npy file."""
    return np.load(filename, allow_pickle=True).item()['images']


def load_file_albedo(filename):
    """Load MODIS albedo data from a .npy file."""
    return np.load(filename, allow_pickle=True).item()


class SatelliteDataset(Dataset):
    """
    Dataset for GHI estimation from GOES-16 satellite imagery + MODIS albedo.

    Each sample loads:
      - 7 GOES-16 channels (M6C01–M6C06, M6C13) from Images_small/
      - 2 MODIS albedo channels (ws_albedo, bs_albedo) from Albedo/
      → concatenated into a 9 × orig_size × orig_size tensor

    Returns: (sat_tensor, tab_tensor, kc_corrected, GHI_corrected)

    Args:
        filenames:    Array of .npy file paths (Images_small/).
        features:     Normalised tabular features as a float32 tensor (N × n_tab).
        targets:      DataFrame with columns 'kc_corrected' and 'GHI_corrected'.
        img_norm:     Per-channel normalisation values (length 7); set to empty to skip.
        transform:    torchvision transform applied to the 9-channel satellite tensor.
        orig_size:    Spatial size to which every channel is resized (default 33).
        num_workers:  Number of parallel workers for pre-loading files into RAM.
    """

    def __init__(self, filenames, features, targets,
                 img_norm=None, transform=None, orig_size=33, num_workers=32):
        self.filenames = filenames
        self.features = features
        self.targets = targets
        self.transform = transform
        self.img_norm = img_norm if img_norm is not None else []
        self.orig_size = orig_size

        print("Loading satellite images into memory…")
        with Pool(num_workers) as p:
            self.loaded_files = list(
                tqdm(p.imap(load_file, filenames), total=len(filenames), desc="GOES-16"))

        albedo_paths = np.char.replace(filenames, 'Images_small', 'Albedo')
        print("Loading albedo maps into memory…")
        with Pool(num_workers) as p:
            self.loaded_albedo = list(
                tqdm(p.imap(load_file_albedo, albedo_paths), total=len(albedo_paths), desc="Albedo"))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        x_sat   = self.loaded_files[index]
        x_modis = self.loaded_albedo[index]
        x_tab   = self.features[index]

        def to_tensor(channel_dict):
            tensors = []
            for ch_data in channel_dict.values():
                if ch_data.shape[-1] != self.orig_size:
                    t = nn.functional.interpolate(
                        torch.tensor(ch_data).unsqueeze(0).unsqueeze(0).float(),
                        size=(self.orig_size, self.orig_size),
                        mode='bilinear', align_corners=True
                    ).squeeze(1)
                else:
                    t = torch.tensor(ch_data).unsqueeze(0)
                tensors.append(t)
            result = torch.cat(tensors, dim=0)
            if torch.isnan(result).any():
                result = torch.nan_to_num(result)
            return result

        sat_tensor   = to_tensor(x_sat)
        modis_tensor = to_tensor(x_modis).float()
        sat_tensor   = torch.cat((sat_tensor, modis_tensor), dim=0)

        if self.transform:
            sat_tensor = self.transform(sat_tensor)

        if len(self.img_norm):
            for i in range(7):
                sat_tensor[i] = sat_tensor[i] / self.img_norm[i]

        y_kc  = self.targets['kc_corrected'][index]
        y_ghi = self.targets['GHI_corrected'][index]
        return sat_tensor, x_tab, y_kc, y_ghi
