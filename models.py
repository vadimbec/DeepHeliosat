import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2 as T




# ══════════════════════════════════════════════════════════════════════════════
#  SHARED COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Two-layer residual block with optional downsampling (stride > 1)."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class RMSELoss(nn.Module):
    def forward(self, yhat, y):
        return torch.sqrt(nn.MSELoss()(yhat, y))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════

class YuanModel(nn.Module):
    """
    Three-stream fusion network for GHI estimation from satellite imagery.

    Architecture
    ────────────
    • Image stream   : 2 × ResidualBlock → GlobalAvgPool → FC(128)
    • Tabular stream : FC(128)
    • Cross stream   : concat(image_feat, tab_feat) → FC(128)
    • Output head    : concat(all 3) → FC(512) → FC(1)

    Inputs
    ──────
    x = (sat_tensor, tab_tensor)
      sat_tensor : (B, n_channels, H, W)  — 7 GOES-16 + 2 MODIS albedo channels
      tab_tensor : (B, n_tab)             — normalised tabular features

    Args:
        n_channels : number of satellite channels (default 9)
        n_tab      : number of tabular features   (default 17)
        crop       : if set, centre-crop images to (crop × crop) before forwarding
    """
    def __init__(self, n_channels=9, n_tab=17, crop=None):
        super().__init__()
        Nf1, Nf2 = 16, 32
        Nfc = 128
        drop_fc = 0.4
        self.crop = crop

        self.img_input = nn.Sequential(
            ResidualBlock(n_channels, Nf1),
            ResidualBlock(Nf1, Nf2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc_img = nn.Sequential(
            nn.Linear(Nf2, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop_fc)
        )
        self.fc_tab = nn.Sequential(
            nn.Linear(n_tab, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop_fc)
        )
        self.fc_mix = nn.Sequential(
            nn.Linear(n_tab + Nf2, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop_fc)
        )
        self.output = nn.Sequential(
            nn.Linear(3 * Nfc, 512), nn.ReLU(), nn.Dropout(drop_fc), nn.Linear(512, 1)
        )

    def forward(self, x):
        x_img, x_tab = x
        if self.crop:
            x_img = T.CenterCrop((self.crop, self.crop))(x_img)
        feat_img = self.img_input(x_img)
        h_img  = self.fc_img(feat_img)
        h_tab  = self.fc_tab(x_tab)
        h_mix  = self.fc_mix(torch.cat((feat_img, x_tab), 1))
        return self.output(torch.cat((h_tab, h_img, h_mix), 1))


class FCN(nn.Module):
    """
    Baseline fully-connected network for GHI estimation.

    Architecture
    ────────────
    Flatten(sat) + concat(tab) → FC(128) → FC(128) → FC(1)

    Inputs  same as YuanModel.

    Args:
        n_channels : number of satellite channels (default 9)
        n_tab      : number of tabular features   (default 27)
        crop       : if set, centre-crop images before forwarding
    """
    def __init__(self, n_channels=9, n_tab=27, crop=None):
        super().__init__()
        self.crop = crop
        self.n_channels = n_channels
        img_flat = (crop ** 2) * n_channels if crop else (33 ** 2) * n_channels

        self.net = nn.Sequential(
            nn.Linear(img_flat + n_tab, 128), nn.ReLU(),
            nn.Linear(128, 128),              nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x_img, x_tab = x
        if self.crop:
            x_img = T.CenterCrop((self.crop, self.crop))(x_img)
        x_flat = nn.Flatten()(x_img)
        return self.net(torch.cat((x_flat, x_tab), 1))


# ══════════════════════════════════════════════════════════════════════════════
#  DEPRECATED — kept for reference, not used in production
#
#  These architectures were explored during the thesis but superseded by
#  YuanModel and FCN above.  Names prefixed with Legacy_ had a name conflict
#  with the current main models.
# ══════════════════════════════════════════════════════════════════════════════

class MyCNN(nn.Module):
    """Early baseline: 2 conv blocks + FC fusion (7 channels, fixed n_tab=27)."""
    def __init__(self, crop=None):
        super().__init__()
        Nf1, Nf2 = 16, 32
        Nfc, drop = 128, 0.4
        self.crop = crop

        self.img_input = nn.Sequential(
            nn.Conv2d(7, Nf1, 3, padding=1), nn.BatchNorm2d(Nf1), nn.ReLU(), nn.Dropout2d(0.2),
            nn.Conv2d(Nf1, Nf2, 3, padding=1), nn.BatchNorm2d(Nf2), nn.ReLU(), nn.Dropout2d(0.2),
        )
        self.fc_img = nn.Sequential(
            nn.Flatten(),
            nn.Linear(Nf2 * crop * crop, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop)
        )
        self.tab_input = nn.Sequential(
            nn.Linear(27, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop)
        )
        self.output = nn.Linear(2 * Nfc, 1)

    def forward(self, x):
        x_img, x_tab = x
        if self.crop:
            x_img = T.CenterCrop((self.crop, self.crop))(x_img)
        return self.output(torch.cat((self.fc_img(self.img_input(x_img)), self.tab_input(x_tab)), 1))


class SmallResNet(nn.Module):
    """Early ResNet baseline (7 channels, fixed n_tab=27, GlobalAvgPool)."""
    def __init__(self, crop=None):
        super().__init__()
        Nf1, Nf2 = 16, 32
        Nfc, drop = 128, 0.4
        self.crop = crop

        self.img_input = nn.Sequential(
            ResidualBlock(7, Nf1),
            ResidualBlock(Nf1, Nf2, stride=2)
        )
        self.fc_img = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(Nf2, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop)
        )
        self.tab_input = nn.Sequential(
            nn.Linear(27, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(),
            nn.Linear(Nfc, Nfc), nn.Dropout(drop)
        )
        self.output = nn.Linear(2 * Nfc, 1)

    def forward(self, x):
        x_img, x_tab = x
        if self.crop:
            x_img = T.CenterCrop((self.crop, self.crop))(x_img)
        return self.output(torch.cat((self.fc_img(self.img_input(x_img)), self.tab_input(x_tab)), 1))


class Legacy_FCN(nn.Module):
    """Original FCN — 7 channels only, no albedo inputs. Superseded by FCN."""
    def __init__(self, crop=None, n_tab=27):
        super().__init__()
        self.crop = crop
        img_flat = (crop ** 2) * 7 if crop else (65 ** 2) * 7
        self.net = nn.Sequential(
            nn.Linear(img_flat + n_tab, 128), nn.ReLU(),
            nn.Linear(128, 128),              nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x_img, x_tab = x
        if self.crop:
            x_img = T.CenterCrop((self.crop, self.crop))(x_img)
        return self.net(torch.cat((nn.Flatten()(x_img), x_tab), 1))


class FCN_tab(nn.Module):
    """Tabular-only baseline (no satellite imagery)."""
    def __init__(self, n_tab=27):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_tab, 64), nn.ReLU(),
            nn.Linear(64, 64),    nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        _, x_tab = x
        return self.net(x_tab)


class Legacy_YuanModel(nn.Module):
    """Original three-stream model — 7 channels, hardcoded n_tab=27. Superseded by YuanModel."""
    def __init__(self, crop=None):
        super().__init__()
        Nf1, Nf2 = 16, 32
        Nfc, drop = 128, 0.4
        self.crop = crop

        self.img_input = nn.Sequential(
            ResidualBlock(7, Nf1), ResidualBlock(Nf1, Nf2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.fc_img = nn.Sequential(nn.Linear(Nf2, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.fc_tab = nn.Sequential(nn.Linear(27, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.fc_mix = nn.Sequential(nn.Linear(27 + Nf2, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.output = nn.Sequential(nn.Linear(3 * Nfc, 512), nn.ReLU(), nn.Dropout(drop), nn.Linear(512, 1))

    def forward(self, x):
        x_img, x_tab = x
        if self.crop:
            x_img = T.CenterCrop((self.crop, self.crop))(x_img)
        f = self.img_input(x_img)
        return self.output(torch.cat((self.fc_tab(x_tab), self.fc_img(f), self.fc_mix(torch.cat((f, x_tab), 1))), 1))


class YuanModel_Ntablen(nn.Module):
    """Three-stream model with variable n_tab, 7 channels. Intermediate step before YuanModel."""
    def __init__(self, n_tab=17, crop=None):
        super().__init__()
        Nf1, Nf2 = 16, 32
        Nfc, drop = 128, 0.4
        self.crop = crop

        self.img_input = nn.Sequential(
            ResidualBlock(7, Nf1), ResidualBlock(Nf1, Nf2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.fc_img = nn.Sequential(nn.Linear(Nf2, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.fc_tab = nn.Sequential(nn.Linear(n_tab, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.fc_mix = nn.Sequential(nn.Linear(n_tab + Nf2, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.output = nn.Sequential(nn.Linear(3 * Nfc, 512), nn.ReLU(), nn.Dropout(drop), nn.Linear(512, 1))

    def forward(self, x):
        x_img, x_tab = x
        if self.crop:
            x_img = T.CenterCrop((self.crop, self.crop))(x_img)
        f = self.img_input(x_img)
        return self.output(torch.cat((self.fc_tab(x_tab), self.fc_img(f), self.fc_mix(torch.cat((f, x_tab), 1))), 1))


class YuanModelFullFeat(nn.Module):
    """Three-stream model that keeps full spatial features (no GlobalAvgPool), 7 channels."""
    def __init__(self, crop=None):
        super().__init__()
        Nf1, Nf2 = 16, 32
        Nfc, drop = 128, 0.4
        self.crop = crop
        n_tab = 27

        self.img_input = nn.Sequential(
            ResidualBlock(7, Nf1), ResidualBlock(Nf1, Nf2, stride=2), nn.Flatten()
        )
        self._init_flat_dim(7)

        self.fc_img = nn.Sequential(nn.Linear(self.flat_dim, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.fc_tab = nn.Sequential(nn.Linear(n_tab, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.fc_mix = nn.Sequential(nn.Linear(n_tab + self.flat_dim, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.output = nn.Sequential(nn.Linear(3 * Nfc, 512), nn.ReLU(), nn.Dropout(drop), nn.Linear(512, 1))

    def _init_flat_dim(self, n_ch):
        with torch.no_grad():
            dummy = torch.zeros(1, n_ch, self.crop, self.crop) if self.crop else torch.zeros(1, n_ch, 35, 35)
            self.flat_dim = self.img_input(dummy).shape[1]

    def forward(self, x):
        x_img, x_tab = x
        if self.crop:
            x_img = T.CenterCrop((self.crop, self.crop))(x_img)
        f = self.img_input(x_img)
        return self.output(torch.cat((self.fc_tab(x_tab), self.fc_img(f), self.fc_mix(torch.cat((f, x_tab), 1))), 1))


class YuanModelFullFeatAlbedo(nn.Module):
    """Full-spatial-feature version of YuanModel with 9 channels. Experimental variant."""
    def __init__(self, n_tab=17, n_channels=9, crop=None):
        super().__init__()
        Nf1, Nf2 = 16, 32
        Nfc, drop = 128, 0.4
        self.crop = crop

        self.img_input = nn.Sequential(
            ResidualBlock(n_channels, Nf1), ResidualBlock(Nf1, Nf2, stride=2), nn.Flatten()
        )
        self._init_flat_dim(n_channels)

        self.fc_img = nn.Sequential(nn.Linear(self.flat_dim, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.fc_tab = nn.Sequential(nn.Linear(n_tab, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.fc_mix = nn.Sequential(nn.Linear(n_tab + self.flat_dim, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.output = nn.Sequential(nn.Linear(3 * Nfc, 512), nn.ReLU(), nn.Dropout(drop), nn.Linear(512, 1))

    def _init_flat_dim(self, n_ch):
        with torch.no_grad():
            dummy = torch.zeros(1, n_ch, self.crop, self.crop) if self.crop else torch.zeros(1, n_ch, 35, 35)
            self.flat_dim = self.img_input(dummy).shape[1]

    def forward(self, x):
        x_img, x_tab = x
        if self.crop:
            x_img = T.CenterCrop((self.crop, self.crop))(x_img)
        f = self.img_input(x_img)
        return self.output(torch.cat((self.fc_tab(x_tab), self.fc_img(f), self.fc_mix(torch.cat((f, x_tab), 1))), 1))


class YuanModelClassifier(nn.Module):
    """Three-stream model adapted for cloud-type classification (output = num_classes)."""
    def __init__(self, num_classes=3, crop=None):
        super().__init__()
        Nf1, Nf2 = 16, 32
        Nfc, drop = 128, 0.4
        self.crop = crop

        self.img_input = nn.Sequential(
            ResidualBlock(7, Nf1), ResidualBlock(Nf1, Nf2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.fc_img = nn.Sequential(nn.Linear(Nf2, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.fc_tab = nn.Sequential(nn.Linear(27, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.fc_mix = nn.Sequential(nn.Linear(27 + Nf2, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.output = nn.Sequential(nn.Linear(3 * Nfc, 512), nn.ReLU(), nn.Dropout(drop), nn.Linear(512, num_classes))

    def forward(self, x):
        x_img, x_tab = x
        if self.crop:
            x_img = T.CenterCrop((self.crop, self.crop))(x_img)
        f = self.img_input(x_img)
        return self.output(torch.cat((self.fc_tab(x_tab), self.fc_img(f), self.fc_mix(torch.cat((f, x_tab), 1))), 1))


class GlobalYuan(nn.Module):
    """Mixture-of-experts: routes each sample to a specialist YuanModel via a classifier."""
    def __init__(self, crop=None, num_classes=3, device=None):
        super().__init__()
        self.crop = crop
        self.device = device
        self.YuanClassifier = YuanModelClassifier(crop=crop, num_classes=num_classes)
        self.models = nn.ModuleList([Legacy_YuanModel(crop=crop) for _ in range(num_classes)])

    def forward(self, x):
        class_indices = self.YuanClassifier(x).argmax(dim=1)
        output = torch.zeros(len(class_indices), 1).to(self.device)
        for cid in class_indices.unique():
            idx = (class_indices == cid).nonzero(as_tuple=True)[0]
            output[idx] = self.models[cid]((x[0][idx], x[1][idx]))
        return output


class GlobalYuanLogits(nn.Module):
    """Soft mixture-of-experts: weighted sum of specialist outputs via softmax logits."""
    def __init__(self, crop=None, num_classes=3, device=None, classifier_grad=False):
        super().__init__()
        self.device = device
        self.YuanClassifier = YuanModelClassifier(crop=crop, num_classes=num_classes)
        self.models = nn.ModuleList([Legacy_YuanModel(crop=crop) for _ in range(num_classes)])
        for p in self.YuanClassifier.parameters():
            p.requires_grad = classifier_grad

    def forward(self, x):
        probs = F.softmax(self.YuanClassifier(x), dim=1)
        output = torch.zeros(probs.shape[0], 1).to(self.device)
        for i, model in enumerate(self.models):
            output += probs[:, i].view(-1, 1) * model(x)
        return output


class PhysYuanModel(nn.Module):
    """Physics-informed variant: predicts (BHI_norm, DHI_norm) instead of kc directly."""
    def __init__(self, crop=None):
        super().__init__()
        Nf1, Nf2 = 16, 32
        Nfc, drop = 128, 0.4
        self.crop = crop

        self.img_input = nn.Sequential(
            ResidualBlock(7, Nf1), ResidualBlock(Nf1, Nf2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.fc_img = nn.Sequential(nn.Linear(Nf2, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.fc_tab = nn.Sequential(nn.Linear(27, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.fc_mix = nn.Sequential(nn.Linear(27 + Nf2, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.output = nn.Sequential(nn.Linear(3 * Nfc, 512), nn.ReLU(), nn.Dropout(drop), nn.Linear(512, 2))

    def forward(self, x):
        x_img, x_tab = x
        if self.crop:
            x_img = T.CenterCrop((self.crop, self.crop))(x_img)
        f = self.img_input(x_img)
        return self.output(torch.cat((self.fc_tab(x_tab), self.fc_img(f), self.fc_mix(torch.cat((f, x_tab), 1))), 1))


# ── Attention modules ─────────────────────────────────────────────────────────

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio), nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid((self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))).unsqueeze(2).unsqueeze(3))
        return attn * x, attn


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1))
        attn = self.sigmoid(out)
        return attn * x, attn


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttentionModule(kernel_size)

    def forward(self, x):
        x, ch_attn = self.channel_attention(x)
        x, sp_attn = self.spatial_attention(x)
        return x, ch_attn, sp_attn


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.q = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.k = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, W, H = x.size()
        q = self.q(x).view(B, -1, W * H).permute(0, 2, 1)
        k = self.k(x).view(B, -1, W * H)
        v = self.v(x).view(B, -1, W * H)
        attn = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, W, H)
        return self.gamma * out + x


class YuanModelAttention(nn.Module):
    """YuanModel with CBAM attention after residual blocks (7 channels, GlobalAvgPool)."""
    def __init__(self, crop=None, output_attention=False):
        super().__init__()
        Nf1, Nf2 = 16, 32
        Nfc, drop = 128, 0.4
        self.crop = crop
        self.output_attention = output_attention
        n_tab = 27

        self.residual_blocks = nn.Sequential(ResidualBlock(7, Nf1), ResidualBlock(Nf1, Nf2, stride=2))
        self.cbam = CBAM(Nf2)
        self.img_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

        self.fc_img = nn.Sequential(nn.Linear(Nf2, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.fc_tab = nn.Sequential(nn.Linear(n_tab, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.fc_mix = nn.Sequential(nn.Linear(n_tab + Nf2, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.output = nn.Sequential(nn.Linear(3 * Nfc, 512), nn.ReLU(), nn.Dropout(drop), nn.Linear(512, 1))

    def forward(self, x):
        x_img, x_tab = x
        if self.crop:
            x_img = T.CenterCrop((self.crop, self.crop))(x_img)
        f = self.residual_blocks(x_img)
        f, ch_attn, sp_attn = self.cbam(f)
        f = self.img_pooling(f)
        out = self.output(torch.cat((self.fc_tab(x_tab), self.fc_img(f), self.fc_mix(torch.cat((f, x_tab), 1))), 1))
        if self.output_attention:
            return out, ch_attn, sp_attn
        return out


class YuanModelAttentionFullFeat(nn.Module):
    """YuanModel with CBAM + full spatial features (no GlobalAvgPool), 7 channels."""
    def __init__(self, crop=None, output_attention=False):
        super().__init__()
        Nf1, Nf2 = 16, 32
        Nfc, drop = 128, 0.4
        self.crop = crop
        self.output_attention = output_attention
        n_tab = 27

        self.residual_blocks = nn.Sequential(ResidualBlock(7, Nf1), ResidualBlock(Nf1, Nf2, stride=2))
        self.cbam = CBAM(Nf2)
        self.img_pooling = nn.Flatten()
        self._init_spatial_dim(7)

        flat = Nf2 * self.spatial_dim
        self.fc_img = nn.Sequential(nn.Linear(flat, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.fc_tab = nn.Sequential(nn.Linear(n_tab, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.fc_mix = nn.Sequential(nn.Linear(n_tab + flat, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.output = nn.Sequential(nn.Linear(3 * Nfc, 512), nn.ReLU(), nn.Dropout(drop), nn.Linear(512, 1))

    def _init_spatial_dim(self, n_ch):
        with torch.no_grad():
            dummy = torch.zeros(1, n_ch, self.crop, self.crop) if self.crop else torch.zeros(1, n_ch, 35, 35)
            out = self.residual_blocks(dummy)
            self.spatial_dim = out.shape[2] * out.shape[3]

    def forward(self, x):
        x_img, x_tab = x
        if self.crop:
            x_img = T.CenterCrop((self.crop, self.crop))(x_img)
        f = self.residual_blocks(x_img)
        f, ch_attn, sp_attn = self.cbam(f)
        f = self.img_pooling(f)
        out = self.output(torch.cat((self.fc_tab(x_tab), self.fc_img(f), self.fc_mix(torch.cat((f, x_tab), 1))), 1))
        if self.output_attention:
            return out, ch_attn, sp_attn
        return out


class YuanModelAttentionFullFeat_Ntablen(nn.Module):
    """YuanModelAttentionFullFeat with variable n_tab."""
    def __init__(self, n_tab=17, crop=None, output_attention=False):
        super().__init__()
        Nf1, Nf2 = 16, 32
        Nfc, drop = 128, 0.4
        self.crop = crop
        self.output_attention = output_attention

        self.residual_blocks = nn.Sequential(ResidualBlock(7, Nf1), ResidualBlock(Nf1, Nf2, stride=2))
        self.cbam = CBAM(Nf2)
        self.img_pooling = nn.Flatten()
        self._init_spatial_dim(7)

        flat = Nf2 * self.spatial_dim
        self.fc_img = nn.Sequential(nn.Linear(flat, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.fc_tab = nn.Sequential(nn.Linear(n_tab, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.fc_mix = nn.Sequential(nn.Linear(n_tab + flat, Nfc), nn.BatchNorm1d(Nfc), nn.ReLU(), nn.Dropout(drop))
        self.output = nn.Sequential(nn.Linear(3 * Nfc, 512), nn.ReLU(), nn.Dropout(drop), nn.Linear(512, 1))

    def _init_spatial_dim(self, n_ch):
        with torch.no_grad():
            dummy = torch.zeros(1, n_ch, self.crop, self.crop) if self.crop else torch.zeros(1, n_ch, 35, 35)
            out = self.residual_blocks(dummy)
            self.spatial_dim = out.shape[2] * out.shape[3]

    def forward(self, x):
        x_img, x_tab = x
        if self.crop:
            x_img = T.CenterCrop((self.crop, self.crop))(x_img)
        f = self.residual_blocks(x_img)
        f, ch_attn, sp_attn = self.cbam(f)
        f = self.img_pooling(f)
        out = self.output(torch.cat((self.fc_tab(x_tab), self.fc_img(f), self.fc_mix(torch.cat((f, x_tab), 1))), 1))
        if self.output_attention:
            return out, ch_attn, sp_attn
        return out


# NOTE: SunpathModel, SunpathPatchModel, MultiHeightFusionModel,
# MultiHeightPatchFusionModel, WeightedMultiHeightPatchFusionModel,
# CombinedWeightModel, and YuanModelSelfAttentionFullFeat are available in
# the original thesis codebase (Tez/models.py).  They are omitted here as
# they depend on pre-computed sunpath pixel sequences and cloud-height maps
# not part of the standard GOES-16 pipeline.
