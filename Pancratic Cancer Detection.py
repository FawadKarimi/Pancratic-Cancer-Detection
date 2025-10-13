"""
Production-Ready 3D Explainable Deep Learning Framework
for Early Pancreatic Cancer Detection Using CT Scans

Complete implementation with:
- Data preprocessing pipeline
- Model architecture
- Training & validation
- XAI metrics (IoU, Pointing Game Accuracy)
- Inference pipeline
- Model checkpointing & logging
- Production deployment utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import ndimage
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(log_dir: Path):
    """Setup comprehensive logging for training and inference"""
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'training_{datetime.now():%Y%m%d_%H%M%S}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ============================================================================
# DATA PREPROCESSING UTILITIES
# ============================================================================

class CTPreprocessor:
    """
    CT scan preprocessing following paper specifications:
    - HU normalization (-87 to 199)
    - Resampling to isotropic spacing
    - Volume cropping/padding
    """

    def __init__(self,
                 hu_min: float = -87.0,
                 hu_max: float = 199.0,
                 target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 target_size: Tuple[int, int, int] = (64, 128, 128)):
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.target_spacing = target_spacing
        self.target_size = target_size

    def normalize_hu(self, volume: np.ndarray) -> np.ndarray:
        """Clip and normalize HU values to [0, 1]"""
        volume = np.clip(volume, self.hu_min, self.hu_max)
        volume = (volume - self.hu_min) / (self.hu_max - self.hu_min)
        return volume.astype(np.float32)

    def resample_volume(self, volume: np.ndarray,
                        original_spacing: Tuple[float, float, float]) -> np.ndarray:
        """Resample volume to isotropic spacing"""
        original_shape = volume.shape
        resize_factor = np.array(original_spacing) / np.array(self.target_spacing)
        new_shape = np.round(original_shape * resize_factor).astype(int)

        # Use scipy for high-quality resampling
        zoom_factors = new_shape / original_shape
        resampled = ndimage.zoom(volume, zoom_factors, order=3, mode='nearest')

        return resampled

    def crop_or_pad(self, volume: np.ndarray) -> np.ndarray:
        """Crop or pad volume to target size"""
        current_shape = np.array(volume.shape)
        target_shape = np.array(self.target_size)

        # Calculate padding/cropping
        pad_width = []
        for i in range(3):
            if current_shape[i] < target_shape[i]:
                diff = target_shape[i] - current_shape[i]
                pad_before = diff // 2
                pad_after = diff - pad_before
                pad_width.append((pad_before, pad_after))
            else:
                pad_width.append((0, 0))

        # Pad if necessary
        if any(p[0] > 0 or p[1] > 0 for p in pad_width):
            volume = np.pad(volume, pad_width, mode='constant', constant_values=0)

        # Crop if necessary
        if any(current_shape > target_shape):
            start_idx = [(c - t) // 2 if c > t else 0
                         for c, t in zip(volume.shape, target_shape)]
            volume = volume[
                start_idx[0]:start_idx[0] + target_shape[0],
                start_idx[1]:start_idx[1] + target_shape[1],
                start_idx[2]:start_idx[2] + target_shape[2]
            ]

        return volume

    def preprocess(self, volume: np.ndarray,
                   spacing: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """Complete preprocessing pipeline"""
        # Normalize HU
        volume = self.normalize_hu(volume)

        # Resample if spacing provided
        if spacing is not None:
            volume = self.resample_volume(volume, spacing)

        # Crop or pad to target size
        volume = self.crop_or_pad(volume)

        return volume


class DataAugmentation3D:
    """3D data augmentation for CT volumes"""

    def __init__(self,
                 rotation_range: float = 15.0,
                 zoom_range: Tuple[float, float] = (0.9, 1.1),
                 noise_std: float = 0.05,
                 apply_prob: float = 0.5):
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.noise_std = noise_std
        self.apply_prob = apply_prob

    def random_rotation(self, volume: np.ndarray) -> np.ndarray:
        """Random 3D rotation"""
        if np.random.rand() > self.apply_prob:
            return volume

        angles = np.random.uniform(-self.rotation_range, self.rotation_range, 3)
        for axis in range(3):
            volume = ndimage.rotate(volume, angles[axis], axes=(axis, (axis + 1) % 3),
                                    reshape=False, order=3, mode='nearest')
        return volume

    def random_zoom(self, volume: np.ndarray) -> np.ndarray:
        """Random zoom"""
        if np.random.rand() > self.apply_prob:
            return volume

        zoom_factor = np.random.uniform(*self.zoom_range)
        zoomed = ndimage.zoom(volume, zoom_factor, order=3, mode='nearest')

        # Crop or pad back to original size
        original_shape = volume.shape
        if zoomed.shape != original_shape:
            # Calculate crop/pad indices
            start_idx = [(z - o) // 2 if z > o else 0
                         for z, o in zip(zoomed.shape, original_shape)]
            end_idx = [s + o for s, o in zip(start_idx, original_shape)]

            # Crop if larger
            if any(z > o for z, o in zip(zoomed.shape, original_shape)):
                zoomed = zoomed[
                    max(0, start_idx[0]):end_idx[0],
                    max(0, start_idx[1]):end_idx[1],
                    max(0, start_idx[2]):end_idx[2]
                ]

            # Pad if smaller
            if zoomed.shape != original_shape:
                pad_width = [(0, max(0, o - z))
                             for o, z in zip(original_shape, zoomed.shape)]
                zoomed = np.pad(zoomed, pad_width, mode='constant')

        return zoomed

    def add_gaussian_noise(self, volume: np.ndarray) -> np.ndarray:
        """Add Gaussian noise"""
        if np.random.rand() > self.apply_prob:
            return volume

        noise = np.random.normal(0, self.noise_std, volume.shape)
        return np.clip(volume + noise, 0, 1)

    def random_flip(self, volume: np.ndarray) -> np.ndarray:
        """Random flipping"""
        if np.random.rand() > self.apply_prob:
            return volume

        axes = [i for i in range(3) if np.random.rand() > 0.5]
        for axis in axes:
            volume = np.flip(volume, axis=axis).copy()
        return volume

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        """Apply all augmentations"""
        volume = self.random_rotation(volume)
        volume = self.random_zoom(volume)
        volume = self.random_flip(volume)
        volume = self.add_gaussian_noise(volume)
        return volume


# ============================================================================
# DATASET CLASS
# ============================================================================

class PancreaticCancerDataset(Dataset):
    """
    Dataset for pancreatic cancer CT scans with segmentation masks
    """

    def __init__(self,
                 data_dir: Path,
                 split: str = 'train',
                 preprocessor: Optional[CTPreprocessor] = None,
                 augmentation: Optional[DataAugmentation3D] = None):
        """
        Args:
            data_dir: Directory containing CT scans and masks
            split: 'train', 'val', or 'test'
            preprocessor: CTPreprocessor instance
            augmentation: DataAugmentation3D instance (only applied in training)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.preprocessor = preprocessor or CTPreprocessor()
        self.augmentation = augmentation if split == 'train' else None

        # Load data manifest
        manifest_file = self.data_dir / f'{split}_manifest.json'
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                self.manifest = json.load(f)
        else:
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

        self.samples = self.manifest['samples']

    def __len__(self) -> int:
        return len(self.samples)

    def load_nifti(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load NIfTI file and return volume + spacing"""
        nii = nib.load(filepath)
        volume = nii.get_fdata()
        spacing = nii.header.get_zooms()
        return volume, spacing

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load CT scan
        ct_path = self.data_dir / sample['ct_scan']
        ct_volume, spacing = self.load_nifti(ct_path)

        # Load segmentation mask
        seg_path = self.data_dir / sample['segmentation']
        seg_mask, _ = self.load_nifti(seg_path)

        # Load auxiliary masks if available
        duct_mask = vessel_mask = None
        if 'duct_mask' in sample:
            duct_path = self.data_dir / sample['duct_mask']
            duct_mask, _ = self.load_nifti(duct_path)

        if 'vessel_mask' in sample:
            vessel_path = self.data_dir / sample['vessel_mask']
            vessel_mask, _ = self.load_nifti(vessel_path)

        # Preprocess
        ct_volume = self.preprocessor.preprocess(ct_volume, spacing)
        seg_mask = self.preprocessor.crop_or_pad(seg_mask)

        if duct_mask is not None:
            duct_mask = self.preprocessor.crop_or_pad(duct_mask)
        if vessel_mask is not None:
            vessel_mask = self.preprocessor.crop_or_pad(vessel_mask)

        # Apply augmentation
        if self.augmentation is not None:
            ct_volume = self.augmentation(ct_volume)

        # Convert to tensors
        ct_tensor = torch.from_numpy(ct_volume).unsqueeze(0).float()
        seg_tensor = torch.from_numpy(seg_mask).long()

        result = {
            'image': ct_tensor,
            'segmentation': seg_tensor,
            'classification': torch.tensor(sample['label'], dtype=torch.long),
            'patient_id': sample['patient_id'],
            'tumor_size': sample.get('tumor_size_mm', -1)
        }

        if duct_mask is not None:
            result['duct_mask'] = torch.from_numpy(duct_mask).unsqueeze(0).float()
        if vessel_mask is not None:
            result['vessel_mask'] = torch.from_numpy(vessel_mask).unsqueeze(0).float()

        return result


# ============================================================================
# MODEL ARCHITECTURE (SAME AS BEFORE BUT WITH IMPROVEMENTS)
# ============================================================================

class ResidualBlock3D(nn.Module):
    """3D Residual block with batch normalization and dropout"""

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out


class AnatomyAwareCNNEncoder(nn.Module):
    """3D CNN Encoder with anatomy-aware features"""

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()

        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.encoder1 = self._make_layer(base_channels, base_channels, 2)
        self.encoder2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.encoder3 = self._make_layer(base_channels * 2, base_channels * 4, 3, stride=2)
        self.encoder4 = self._make_layer(base_channels * 4, base_channels * 8, 3, stride=2)
        self.encoder5 = self._make_layer(base_channels * 8, base_channels * 16, 2, stride=2)

        self.aux_duct_seg = nn.Conv3d(base_channels * 4, 1, kernel_size=1)
        self.aux_vessel_seg = nn.Conv3d(base_channels * 8, 1, kernel_size=1)

    def _make_layer(self, in_channels: int, out_channels: int,
                    num_blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [ResidualBlock3D(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.init_conv(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)

        aux_duct = self.aux_duct_seg(x3)
        aux_vessel = self.aux_vessel_seg(x4)

        return {
            'features': [x1, x2, x3, x4, x5],
            'aux_duct': aux_duct,
            'aux_vessel': aux_vessel
        }


class MultiHeadSelfAttention3D(nn.Module):
    """Multi-head self-attention for 3D volumes"""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0.1, proj_drop: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_weights = attn.clone()
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_weights


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and MLP"""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention3D(dim, num_heads, attn_drop=dropout,
                                             proj_drop=dropout)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.attn_weights = None

    def forward(self, x):
        attn_out, attn_weights = self.attn(self.norm1(x))
        self.attn_weights = attn_weights
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MultiScaleTransformerModule(nn.Module):
    """Multi-scale Transformer for global context"""

    def __init__(self, in_channels: int, num_heads: int = 8, depth: int = 4):
        super().__init__()
        self.in_channels = in_channels

        self.patch_embed = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, 2000, in_channels))

        self.blocks = nn.ModuleList([
            TransformerBlock(in_channels, num_heads) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, C, D, H, W = x.shape

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        N = x.shape[1]
        x = x + self.pos_embed[:, :N, :]

        attn_maps = []
        for block in self.blocks:
            x = block(x)
            if block.attn_weights is not None:
                attn_maps.append(block.attn_weights)

        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, D, H, W)

        return x, attn_maps


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connections"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels,
                                           kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ResidualBlock3D(in_channels + skip_channels, out_channels),
            ResidualBlock3D(out_channels, out_channels)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class VolumetricDecoder(nn.Module):
    """Volumetric Decoder with skip connections"""

    def __init__(self, base_channels: int = 32):
        super().__init__()

        self.decoder4 = DecoderBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.decoder3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.decoder2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.decoder1 = DecoderBlock(base_channels * 2, base_channels, base_channels)

        self.deep_sup4 = nn.Conv3d(base_channels * 8, 2, kernel_size=1)
        self.deep_sup3 = nn.Conv3d(base_channels * 4, 2, kernel_size=1)
        self.deep_sup2 = nn.Conv3d(base_channels * 2, 2, kernel_size=1)

        self.final_conv = nn.Conv3d(base_channels, 2, kernel_size=1)

    def forward(self, transformer_features, skip_connections):
        x1, x2, x3, x4 = skip_connections
        x5 = transformer_features

        d4 = self.decoder4(x5, x4)
        ds4 = self.deep_sup4(d4)

        d3 = self.decoder3(d4, x3)
        ds3 = self.deep_sup3(d3)

        d2 = self.decoder2(d3, x2)
        ds2 = self.deep_sup2(d2)

        d1 = self.decoder1(d2, x1)
        final_out = self.final_conv(d1)

        return {
            'final': final_out,
            'deep_sup': [ds4, ds3, ds2]
        }


class ExplainablePancreaticCancerDetector(nn.Module):
    """
    Complete 3D XDL Framework for Pancreatic Cancer Detection
    Production-ready with integrated XAI
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 base_channels: int = 32, transformer_depth: int = 4):
        super().__init__()

        self.encoder = AnatomyAwareCNNEncoder(in_channels, base_channels)
        self.transformer = MultiScaleTransformerModule(
            base_channels * 16, num_heads=8, depth=transformer_depth
        )
        self.decoder = VolumetricDecoder(base_channels)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        self.attention_maps = None

    def forward(self, x):
        encoder_out = self.encoder(x)
        features = encoder_out['features']

        transformer_out, attn_maps = self.transformer(features[-1])
        self.attention_maps = attn_maps

        decoder_out = self.decoder(transformer_out, features[:4])

        pooled = self.global_pool(transformer_out).flatten(1)
        classification = self.classifier(pooled)

        return {
            'segmentation': decoder_out['final'],
            'classification': classification,
            'deep_supervision': decoder_out['deep_sup'],
            'aux_anatomy': {
                'duct': encoder_out['aux_duct'],
                'vessel': encoder_out['aux_vessel']
            }
        }

    def get_xai_visualization(self, x):
        """Generate 3D attention-based XAI visualization"""
        with torch.no_grad():
            _ = self.forward(x)

            if self.attention_maps is None:
                return None

            B = x.shape[0]
            attn_agg = torch.stack(self.attention_maps).mean(dim=0)
            attn_agg = attn_agg.mean(dim=1)

            D, H, W = x.shape[2:]
            target_size = (D // 16, H // 16, W // 16)

            attn_map = attn_agg.mean(dim=1)
            attn_map = attn_map.reshape(B, *target_size)

            attn_map = F.interpolate(
                attn_map.unsqueeze(1),
                size=(D, H, W),
                mode='trilinear',
                align_corners=False
            )

            return attn_map


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        pred = pred[:, 1, ...]  # Foreground class
        target = (target == 1).float()

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined loss with deep supervision and auxiliary losses"""

    def __init__(self, aux_weight: float = 0.3, deep_sup_weight: float = 0.4,
                 use_dice: bool = True):
        super().__init__()
        self.aux_weight = aux_weight
        self.deep_sup_weight = deep_sup_weight
        self.use_dice = use_dice

        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss() if use_dice else None
        self.cls_loss = nn.CrossEntropyLoss()
        self.aux_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        # Main segmentation loss
        seg_loss = self.ce_loss(outputs['segmentation'], targets['segmentation'])
        if self.use_dice:
            seg_loss = seg_loss + self.dice_loss(outputs['segmentation'],
                                                 targets['segmentation'])

        # Classification loss
        cls_loss = self.cls_loss(outputs['classification'], targets['classification'])

        # Deep supervision
        deep_sup_loss = 0
        for ds_out in outputs['deep_supervision']:
            target_down = F.interpolate(
                targets['segmentation'].float().unsqueeze(1),
                size=ds_out.shape[2:],
                mode='nearest'
            ).long().squeeze(1)
            deep_sup_loss += self.ce_loss(ds_out, target_down)
        deep_sup_loss /= len(outputs['deep_supervision'])

        # Auxiliary anatomy losses
        aux_loss = 0
        if 'duct_mask' in targets:
            aux_loss += self.aux_loss(
                outputs['aux_anatomy']['duct'],
                targets['duct_mask']
            )
        if 'vessel_mask' in targets:
            aux_loss += self.aux_loss(
                outputs['aux_anatomy']['vessel'],
                targets['vessel_mask']
            )

        total_loss = (seg_loss + cls_loss +
                      self.deep_sup_weight * deep_sup_loss +
                      self.aux_weight * aux_loss)

        return {
            'total': total_loss,
            'segmentation': seg_loss,
            'classification': cls_loss,
            'deep_supervision': deep_sup_loss,
            'auxiliary': aux_loss
        }


# ============================================================================
# XAI EVALUATION METRICS
# ============================================================================

class XAIMetrics:
    """Quantitative XAI validation metrics"""

    @staticmethod
    def compute_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor,
                    threshold: float = 0.5) -> float:
        """
        Compute Intersection over Union (IoU) between predicted
        attention map and ground truth segmentation
        """
        pred_binary = (pred_mask > threshold).float()
        gt_binary = (gt_mask > 0).float()

        intersection = (pred_binary * gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum() - intersection

        if union == 0:
            return 0.0

        iou = (intersection / union).item()
        return iou

    @staticmethod
    def compute_pointing_game_accuracy(attn_map: torch.Tensor,
                                       gt_mask: torch.Tensor) -> float:
        """
        Compute Pointing Game Accuracy: whether the voxel of maximum
        attribution falls within the tumor boundary
        """
        # Find voxel with maximum attention
        max_idx = attn_map.flatten().argmax()

        # Convert to 3D coordinates
        D, H, W = attn_map.shape[-3:]
        z = max_idx // (H * W)
        y = (max_idx % (H * W)) // W
        x = max_idx % W

        # Check if max attention point is inside tumor
        is_inside = gt_mask[0, 0, z, y, x] > 0

        return float(is_inside.item())

    @staticmethod
    def batch_compute_metrics(attn_maps: torch.Tensor,
                              gt_masks: torch.Tensor) -> Dict[str, float]:
        """Compute metrics for a batch"""
        batch_size = attn_maps.shape[0]

        ious = []
        pointing_accs = []

        for i in range(batch_size):
            iou = XAIMetrics.compute_iou(attn_maps[i], gt_masks[i])
            pointing_acc = XAIMetrics.compute_pointing_game_accuracy(
                attn_maps[i], gt_masks[i]
            )

            ious.append(iou)
            pointing_accs.append(pointing_acc)

        return {
            'mean_iou': np.mean(ious),
            'std_iou': np.std(ious),
            'mean_pointing_accuracy': np.mean(pointing_accs),
            'std_pointing_accuracy': np.std(pointing_accs)
        }


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class PerformanceMetrics:
    """Comprehensive performance metrics for model evaluation"""

    @staticmethod
    def compute_confusion_matrix(preds: torch.Tensor,
                                 targets: torch.Tensor) -> Dict[str, int]:
        """Compute TP, TN, FP, FN"""
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()

        tp = np.sum((preds == 1) & (targets == 1))
        tn = np.sum((preds == 0) & (targets == 0))
        fp = np.sum((preds == 1) & (targets == 0))
        fn = np.sum((preds == 0) & (targets == 1))

        return {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}

    @staticmethod
    def compute_metrics(confusion: Dict[str, int]) -> Dict[str, float]:
        """Compute sensitivity, specificity, accuracy, F1"""
        tp, tn, fp, fn = confusion['tp'], confusion['tn'], confusion['fp'], confusion['fn']

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'precision': precision,
            'f1_score': f1
        }

    @staticmethod
    def stratify_by_tumor_size(preds: torch.Tensor,
                               targets: torch.Tensor,
                               tumor_sizes: List[float],
                               size_threshold: float = 20.0) -> Dict[str, Dict]:
        """Stratify performance by tumor size (<2cm vs >=2cm)"""
        small_preds, small_targets = [], []
        large_preds, large_targets = [], []

        for pred, target, size in zip(preds, targets, tumor_sizes):
            if size < size_threshold:
                small_preds.append(pred)
                small_targets.append(target)
            else:
                large_preds.append(pred)
                large_targets.append(target)

        results = {}

        if small_preds:
            small_preds = torch.tensor(small_preds)
            small_targets = torch.tensor(small_targets)
            confusion = PerformanceMetrics.compute_confusion_matrix(small_preds, small_targets)
            results['small_lesions'] = PerformanceMetrics.compute_metrics(confusion)

        if large_preds:
            large_preds = torch.tensor(large_preds)
            large_targets = torch.tensor(large_targets)
            confusion = PerformanceMetrics.compute_confusion_matrix(large_preds, large_targets)
            results['large_lesions'] = PerformanceMetrics.compute_metrics(confusion)

        return results


# ============================================================================
# TRAINING ENGINE
# ============================================================================

class Trainer:
    """Production-ready training engine with logging and checkpointing"""

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 device: torch.device,
                 output_dir: Path,
                 logger: logging.Logger,
                 num_epochs: int = 100,
                 early_stopping_patience: int = 15):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0

        self.train_history = []
        self.val_history = []

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        loss_components = {'seg': 0.0, 'cls': 0.0, 'deep_sup': 0.0, 'aux': 0.0}

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs} [Train]')

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            targets = {
                'segmentation': batch['segmentation'].to(self.device),
                'classification': batch['classification'].to(self.device)
            }

            if 'duct_mask' in batch:
                targets['duct_mask'] = batch['duct_mask'].to(self.device)
            if 'vessel_mask' in batch:
                targets['vessel_mask'] = batch['vessel_mask'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Compute loss
            losses = self.criterion(outputs, targets)
            loss = losses['total']

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            loss_components['seg'] += losses['segmentation'].item()
            loss_components['cls'] += losses['classification'].item()
            loss_components['deep_sup'] += losses['deep_supervision'].item()
            loss_components['aux'] += losses['auxiliary'].item()

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })

        # Average losses
        n_batches = len(self.train_loader)
        avg_loss = total_loss / n_batches
        for key in loss_components:
            loss_components[key] /= n_batches

        return {'total_loss': avg_loss, **loss_components}

    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_tumor_sizes = []

        xai_ious = []
        xai_pointing_accs = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs} [Val]')

            for batch in pbar:
                images = batch['image'].to(self.device)
                targets = {
                    'segmentation': batch['segmentation'].to(self.device),
                    'classification': batch['classification'].to(self.device)
                }

                if 'duct_mask' in batch:
                    targets['duct_mask'] = batch['duct_mask'].to(self.device)
                if 'vessel_mask' in batch:
                    targets['vessel_mask'] = batch['vessel_mask'].to(self.device)

                # Forward pass
                outputs = self.model(images)
                losses = self.criterion(outputs, targets)

                total_loss += losses['total'].item()

                # Get predictions
                preds = outputs['classification'].argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(batch['classification'].tolist())
                all_tumor_sizes.extend(batch['tumor_size'])

                # Compute XAI metrics
                attn_maps = self.model.get_xai_visualization(images)
                if attn_maps is not None:
                    gt_masks = batch['segmentation'].unsqueeze(1).to(self.device)
                    xai_metrics = XAIMetrics.batch_compute_metrics(attn_maps, gt_masks)
                    xai_ious.append(xai_metrics['mean_iou'])
                    xai_pointing_accs.append(xai_metrics['mean_pointing_accuracy'])

        # Compute performance metrics
        avg_loss = total_loss / len(self.val_loader)

        all_preds = torch.tensor(all_preds)
        all_targets = torch.tensor(all_targets)

        confusion = PerformanceMetrics.compute_confusion_matrix(all_preds, all_targets)
        metrics = PerformanceMetrics.compute_metrics(confusion)

        # Stratify by tumor size
        size_stratified = PerformanceMetrics.stratify_by_tumor_size(
            all_preds, all_targets, all_tumor_sizes, size_threshold=20.0
        )

        # XAI metrics
        xai_results = {
            'mean_iou': np.mean(xai_ious) if xai_ious else 0.0,
            'mean_pointing_acc': np.mean(xai_pointing_accs) if xai_pointing_accs else 0.0
        }

        return {
            'loss': avg_loss,
            'overall': metrics,
            'size_stratified': size_stratified,
            'xai_metrics': xai_results
        }

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'train_history': self.train_history,
            'val_history': self.val_history
        }

        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f'Saved best checkpoint at epoch {epoch + 1}')

    def train(self):
        """Full training loop"""
        self.logger.info('Starting training...')
        self.logger.info(f'Total epochs: {self.num_epochs}')
        self.logger.info(f'Training samples: {len(self.train_loader.dataset)}')
        self.logger.info(f'Validation samples: {len(self.val_loader.dataset)}')

        for epoch in range(self.num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_history.append(train_metrics)

            # Validate
            val_metrics = self.validate(epoch)
            self.val_history.append(val_metrics)

            # Log metrics
            self.logger.info(f'\nEpoch {epoch + 1}/{self.num_epochs}')
            self.logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}")
            self.logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            self.logger.info(f"Val Sensitivity: {val_metrics['overall']['sensitivity']:.4f}")
            self.logger.info(f"Val Specificity: {val_metrics['overall']['specificity']:.4f}")
            self.logger.info(f"Val F1: {val_metrics['overall']['f1_score']:.4f}")

            # Log size-stratified metrics
            if 'small_lesions' in val_metrics['size_stratified']:
                small_metrics = val_metrics['size_stratified']['small_lesions']
                self.logger.info(f"Small Lesions (<2cm) Sensitivity: {small_metrics['sensitivity']:.4f}")

            # Log XAI metrics
            self.logger.info(f"XAI IoU: {val_metrics['xai_metrics']['mean_iou']:.4f}")
            self.logger.info(f"XAI Pointing Accuracy: {val_metrics['xai_metrics']['mean_pointing_acc']:.4f}")

            # Learning rate scheduling
            self.scheduler.step()

            # Check for improvement
            current_f1 = val_metrics['overall']['f1_score']
            is_best = current_f1 > self.best_val_f1

            if is_best:
                self.best_val_f1 = current_f1
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                self.logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                break

        self.logger.info('Training completed!')
        self.logger.info(f'Best validation F1: {self.best_val_f1:.4f}')

        # Save training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump({
                'train': self.train_history,
                'val': self.val_history
            }, f, indent=2)


# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class InferenceEngine:
    """Production inference engine with XAI visualization"""

    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 preprocessor: CTPreprocessor,
                 logger: logging.Logger):

        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.preprocessor = preprocessor
        self.logger = logger

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f'Loaded checkpoint from {checkpoint_path}')

        if 'metrics' in checkpoint:
            self.logger.info(f"Checkpoint metrics: {checkpoint['metrics']}")

    def preprocess_ct_scan(self, ct_path: Path) -> Tuple[torch.Tensor, Dict]:
        """Load and preprocess CT scan"""
        nii = nib.load(ct_path)
        volume = nii.get_fdata()
        spacing = nii.header.get_zooms()

        original_shape = volume.shape
        processed_volume = self.preprocessor.preprocess(volume, spacing)

        tensor = torch.from_numpy(processed_volume).unsqueeze(0).unsqueeze(0).float()

        metadata = {
            'original_shape': original_shape,
            'spacing': spacing,
            'preprocessed_shape': processed_volume.shape
        }

        return tensor, metadata

    @torch.no_grad()
    def predict(self, ct_path: Path,
                generate_xai: bool = True) -> Dict:
        """
        Run inference on a CT scan

        Returns:
            Dictionary with predictions, segmentation, and XAI visualization
        """
        self.logger.info(f'Running inference on {ct_path}')

        # Preprocess
        ct_tensor, metadata = self.preprocess_ct_scan(ct_path)
        ct_tensor = ct_tensor.to(self.device)

        # Forward pass
        outputs = self.model(ct_tensor)

        # Get predictions
        class_probs = F.softmax(outputs['classification'], dim=1)
        predicted_class = class_probs.argmax(dim=1).item()
        confidence = class_probs[0, predicted_class].item()

        # Get segmentation
        seg_probs = F.softmax(outputs['segmentation'], dim=1)
        seg_mask = seg_probs.argmax(dim=1).cpu().numpy()[0]

        # Generate XAI visualization
        xai_map = None
        if generate_xai:
            xai_map = self.model.get_xai_visualization(ct_tensor)
            if xai_map is not None:
                xai_map = xai_map.cpu().numpy()[0, 0]

        results = {
            'prediction': {
                'class': predicted_class,
                'class_name': 'PDAC' if predicted_class == 1 else 'Normal',
                'confidence': confidence,
                'probabilities': {
                    'normal': class_probs[0, 0].item(),
                    'pdac': class_probs[0, 1].item()
                }
            },
            'segmentation': seg_mask,
            'xai_visualization': xai_map,
            'metadata': metadata
        }

        self.logger.info(f"Prediction: {results['prediction']['class_name']} "
                         f"(confidence: {confidence:.4f})")

        return results

    def visualize_results(self, results: Dict,
                          output_path: Optional[Path] = None,
                          slice_idx: Optional[int] = None):
        """Visualize inference results"""
        seg_mask = results['segmentation']
        xai_map = results['xai_visualization']

        if slice_idx is None:
            slice_idx = seg_mask.shape[0] // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Segmentation
        axes[0].imshow(seg_mask[slice_idx], cmap='gray')
        axes[0].set_title('Segmentation Prediction')
        axes[0].axis('off')

        # XAI heatmap
        if xai_map is not None:
            axes[1].imshow(xai_map[slice_idx], cmap='jet', alpha=0.7)
            axes[1].set_title('XAI Attention Map')
            axes[1].axis('off')

            # Overlay
            axes[2].imshow(seg_mask[slice_idx], cmap='gray')
            axes[2].imshow(xai_map[slice_idx], cmap='jet', alpha=0.5)
            axes[2].set_title('Overlay')
            axes[2].axis('off')

        plt.suptitle(f"Prediction: {results['prediction']['class_name']} "
                     f"(Confidence: {results['prediction']['confidence']:.2f})")

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            self.logger.info(f'Saved visualization to {output_path}')

        plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main_training():
    """Main training pipeline"""
    # Setup
    output_dir = Path('./output')
    logger = setup_logging(output_dir / 'logs')

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Data preprocessing
    preprocessor = CTPreprocessor(
        hu_min=-87.0,
        hu_max=199.0,
        target_spacing=(1.0, 1.0, 1.0),
        target_size=(64, 128, 128)
    )

    augmentation = DataAugmentation3D(
        rotation_range=15.0,
        zoom_range=(0.9, 1.1),
        noise_std=0.05,
        apply_prob=0.5
    )

    # Datasets
    data_dir = Path('./data')
    train_dataset = PancreaticCancerDataset(
        data_dir=data_dir,
        split='train',
        preprocessor=preprocessor,
        augmentation=augmentation
    )

    val_dataset = PancreaticCancerDataset(
        data_dir=data_dir,
        split='val',
        preprocessor=preprocessor,
        augmentation=None
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model
    model = ExplainablePancreaticCancerDetector(
        in_channels=1,
        num_classes=2,
        base_channels=32,
        transformer_depth=4
    )

    logger.info(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Loss and optimizer
    criterion = CombinedLoss(
        aux_weight=0.3,
        deep_sup_weight=0.4,
        use_dice=True
    )

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir / 'checkpoints',
        logger=logger,
        num_epochs=100,
        early_stopping_patience=15
    )

    # Train
    trainer.train()


def main_inference():
    """Main inference pipeline"""
    # Setup
    output_dir = Path('./output')
    logger = setup_logging(output_dir / 'logs')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Preprocessor
    preprocessor = CTPreprocessor()

    # Model
    model = ExplainablePancreaticCancerDetector(
        in_channels=1,
        num_classes=2,
        base_channels=32,
        transformer_depth=4
    )

    # Inference engine
    engine = InferenceEngine(
        model=model,
        device=device,
        preprocessor=preprocessor,
        logger=logger
    )

    # Load checkpoint
    checkpoint_path = output_dir / 'checkpoints' / 'checkpoint_best.pth'
    engine.load_checkpoint(checkpoint_path)

    # Run inference on a sample
    ct_scan_path = Path('./data/test/sample_001.nii.gz')
    results = engine.predict(ct_scan_path, generate_xai=True)

    # Visualize
    vis_output = output_dir / 'visualizations' / 'sample_001_results.png'
    vis_output.parent.mkdir(parents=True, exist_ok=True)
    engine.visualize_results(results, output_path=vis_output)

    logger.info('Inference completed successfully')


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'inference':
        main_inference()
    else:
        main_training()