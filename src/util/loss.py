# --------------------------------------------------------
# What Matters When Repurposing Diffusion Models for General Dense Perception Tasks? (https://arxiv.org/abs/2403.06090)
# Github source: https://github.com/aim-uofa/GenPercept
# Copyright (c) 2024, Advanced Intelligent Machines (AIM)
# Licensed under The BSD 2-Clause License [see LICENSE for details]
# Author: Guangkai Xu (https://github.com/guangkaixu/)
# --------------------------------------------------------------------------
# This code is based on Marigold and diffusers codebases
# https://github.com/prs-eth/marigold
# https://github.com/huggingface/diffusers
# --------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/aim-uofa/GenPercept#%EF%B8%8F-citation
# More information about the method can be found at https://github.com/aim-uofa/GenPercept
# --------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#We use only CombinedSegLoss
def get_loss(loss_name, **kwargs):
    return CombinedSegLoss(**kwargs)

#Other loss functions is used only as example
class L1LossWithMask:
    def __init__(self, batch_reduction=False):
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        diff = depth_pred - depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        loss = torch.sum(torch.abs(diff)) / n
        if self.batch_reduction:
            loss = loss.mean()
        return loss


class MeanAbsRelLoss:
    def __init__(self) -> None:
        # super().__init__()
        pass

    def __call__(self, pred, gt):
        diff = pred - gt
        rel_abs = torch.abs(diff / gt)
        loss = torch.mean(rel_abs, dim=0)
        return loss


class SILogMSELoss:
    def __init__(self, lamb, log_pred=True, batch_reduction=True):
        """Scale Invariant Log MSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogMSELoss, self).__init__()
        self.lamb = lamb
        self.pred_in_log = log_pred
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        log_depth_pred = (
            depth_pred if self.pred_in_log else torch.log(torch.clip(depth_pred, 1e-8))
        )
        log_depth_gt = torch.log(depth_gt)

        diff = log_depth_pred - log_depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)

        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
        loss = first_term - second_term
        if self.batch_reduction:
            loss = loss.mean()
        return loss


class SILogRMSELoss:
    def __init__(self, lamb, alpha, log_pred=True):
        """Scale Invariant Log RMSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            alpha:
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogRMSELoss, self).__init__()
        self.lamb = lamb
        self.alpha = alpha
        self.pred_in_log = log_pred

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        log_depth_pred = depth_pred if self.pred_in_log else torch.log(depth_pred)
        log_depth_gt = torch.log(depth_gt)
        # borrowed from https://github.com/aliyun/NeWCRFs
        # diff = log_depth_pred[valid_mask] - log_depth_gt[valid_mask]
        # return torch.sqrt((diff ** 2).mean() - self.lamb * (diff.mean() ** 2)) * self.alpha

        diff = log_depth_pred - log_depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)
        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
        loss = torch.sqrt(first_term - second_term).mean() * self.alpha
        return loss


# ========= Segmentation Loss Functions =========

class DiceLoss:
    def __init__(self, smooth=1.0, weight=None, batch_reduction=True):
        """Dice loss for segmentation tasks
        
        Args:
            smooth (float, optional): Smoothing term to avoid division by zero. Defaults to 1.0.
            weight (torch.Tensor, optional): Class weights. Defaults to None.
            batch_reduction (bool, optional): Whether to average over the batch. Defaults to True.
        """
        self.smooth = smooth
        self.weight = weight
        self.batch_reduction = batch_reduction
        
    def __call__(self, prediction, target, valid_mask=None):
        """
        Args:
            prediction: Model prediction logits of shape [B, C, H, W]
            target: Ground truth indices of shape [B, H, W]
            valid_mask: Optional mask to apply to the loss, shape [B, 1, H, W]
        """
        B, C, H, W = prediction.size()
        
        # Convert logits to probabilities
        probs = F.softmax(prediction, dim=1)
        
        # One-hot encode the target
        target_one_hot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()
        
        # Apply mask if provided
        if valid_mask is not None:
            probs = probs * valid_mask
            target_one_hot = target_one_hot * valid_mask
        
        # Flatten for dice calculation
        probs_flat = probs.view(B, C, -1)
        targets_flat = target_one_hot.view(B, C, -1)
        
        # Calculate Dice score for each class
        intersection = torch.sum(probs_flat * targets_flat, dim=2)
        union = torch.sum(probs_flat, dim=2) + torch.sum(targets_flat, dim=2)
        
        # Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Apply class weights if provided
        if self.weight is not None:
            weight = self.weight.to(dice.device)
            dice = dice * weight
            
        # Calculate Dice loss (1 - Dice coefficient)
        dice_loss = 1.0 - dice.mean(dim=1)  # Mean across classes
        
        # Apply batch reduction if requested
        if self.batch_reduction:
            dice_loss = dice_loss.mean()  # Mean across batch
            
        return dice_loss


class FocalLoss:
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean', batch_reduction=True):
        """Focal loss for segmentation tasks
        
        Args:
            alpha (float, optional): Weighting factor. Defaults to 0.25.
            gamma (float, optional): Focusing parameter. Defaults to 2.0.
            weight (torch.Tensor, optional): Class weights. Defaults to None.
            reduction (str, optional): Reduction method ('mean', 'sum', 'none'). Defaults to 'mean'.
            batch_reduction (bool, optional): Whether to average over the batch. Defaults to True.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.batch_reduction = batch_reduction
        
    def __call__(self, prediction, target, valid_mask=None):
        """
        Args:
            prediction: Model prediction logits of shape [B, C, H, W]
            target: Ground truth indices of shape [B, H, W]
            valid_mask: Optional mask to apply to the loss, shape [B, 1, H, W]
        """
        B, C, H, W = prediction.size()
        
        # Reshape for cross entropy
        logits = prediction.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*H*W, C]
        targets = target.view(-1)  # [B*H*W]
        
        # Apply mask if provided
        if valid_mask is not None:
            mask_flat = valid_mask.view(-1).bool()
            logits = logits[mask_flat]
            targets = targets[mask_flat]
            
        # Move weight to device if provided
        weight = self.weight.to(logits.device) if self.weight is not None else None
        
        # Calculate CE loss
        ce_loss = F.cross_entropy(logits, targets, weight=weight, reduction='none')
        
        # Calculate probabilities and focal weighting
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha balancing
        if self.alpha is not None:
            alpha_weight = torch.ones_like(targets).float() * self.alpha
            alpha_weight[targets == 0] = 1 - self.alpha  # Different weight for background
            focal_weight = alpha_weight * focal_weight
            
        # Apply focal weighting to CE loss
        focal_loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
            
        # Apply batch reduction if needed
        if not self.batch_reduction and self.reduction != 'none':
            focal_loss = focal_loss.view(B, -1).mean(dim=1)
            
        return focal_loss


class CombinedSegLoss:
    def __init__(self, num_classes, ce_weight=1.0, dice_weight=1.0, focal_weight=0.5, 
                 class_weights=None, batch_reduction=True, return_dict=False):
        """Combined segmentation loss (CE + Dice + Focal)
        
        Args:
            num_classes (int): Number of segmentation classes
            ce_weight (float, optional): Weight for CE loss. Defaults to 1.0.
            dice_weight (float, optional): Weight for Dice loss. Defaults to 1.0.
            focal_weight (float, optional): Weight for Focal loss. Defaults to 0.5.
            class_weights (torch.Tensor, optional): Class weights. Defaults to None.
            batch_reduction (bool, optional): Whether to average over the batch. Defaults to True.
            return_dict (bool, optional): Whether to return loss components dict. Defaults to False.
        """
        self.num_classes = num_classes
        
        # Weights for each loss component
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # Class weights for handling class imbalance
        self.class_weights = class_weights
        
        # Batch reduction flag
        self.batch_reduction = batch_reduction
        
        # Return components flag
        self.return_dict = return_dict
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        self.dice_loss = DiceLoss(weight=class_weights, batch_reduction=False)
        self.focal_loss = FocalLoss(weight=class_weights, reduction='none', batch_reduction=False)
        
    def __call__(self, prediction, target, valid_mask=None):
        """
        Compute combined segmentation loss (CE + Dice + Focal).

        Args:
            prediction: Model prediction logits of shape [B, C, H, W] or [B, H, W]
            target: Ground truth indices of shape [B, H, W]
            valid_mask: Optional mask of shape [B, 1, H, W]
            
        Returns:
            torch.Tensor or (torch.Tensor, dict): Combined loss and optionally a dictionary of loss components
        """
        # If prediction is 3D, add a channel dimension
        if prediction.dim() == 3:
            # [B, H, W] -> [B, 1, H, W]
            prediction = prediction.unsqueeze(1)

        B, C, H, W = prediction.size()
        # If head returned a single-channel map, replicate across classes
        if C == 1 and self.num_classes > 1:
            prediction = prediction.repeat(1, self.num_classes, 1, 1)
            C = self.num_classes

        assert C == self.num_classes, f"Expected {self.num_classes} classes, got {C}"

        # Move class_weights to the same device as prediction
        if self.class_weights is not None:
            cw = self.class_weights.to(prediction.device)
            self.ce_loss.weight = cw
            self.dice_loss.weight = cw
            self.focal_loss.weight = cw

        # Align spatial dims to target
        tH, tW = target.shape[1], target.shape[2]
        if (H, W) != (tH, tW):
            prediction = F.interpolate(prediction, size=(tH, tW), mode='bilinear', align_corners=False)
            B, C, H, W = prediction.size()

        # Cross-entropy loss
        ce_flat = self.ce_loss(prediction, target)
        if valid_mask is not None:
            ce_flat = ce_flat * valid_mask.view(-1)
            n_valid = valid_mask.sum()
            ce = ce_flat.sum() / (n_valid + 1e-8) if n_valid > 0 else ce_flat.sum() * 0.0
        else:
            ce = ce_flat.mean()

        # Reshape CE loss to match batch dimension for non-batch reduction
        if not self.batch_reduction:
            ce = ce_flat.view(B, -1).mean(dim=1)

        # Dice loss
        dice = self.dice_loss(prediction, target, valid_mask)

        # Focal loss
        focal = self.focal_loss(prediction, target, valid_mask)

        # Combine weighted losses
        combined_loss = (
            self.ce_weight * ce +
            self.dice_weight * dice +
            self.focal_weight * focal
        )
        
        # Apply batch reduction if needed
        if self.batch_reduction:
            combined_loss = combined_loss.mean()

        # Return loss components if requested
        if self.return_dict:
            loss_dict = {
                'ce_loss': ce.detach().mean().item() if not self.batch_reduction else ce.item(),
                'dice_loss': dice.detach().mean().item() if not self.batch_reduction else dice.item(),
                'focal_loss': focal.detach().mean().item() if not self.batch_reduction else focal.item(),
                'combined_loss': combined_loss.item() if self.batch_reduction else combined_loss.detach().mean().item()
            }
            return combined_loss, loss_dict
        
        return combined_loss