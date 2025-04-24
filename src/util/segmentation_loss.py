import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLoss(nn.Module):
    """
    Combined weighted CE + Dice loss without ignore_index.
    Поддерживает target в виде:
      - [B, H, W] int-карта
      - [B, 1, H, W] один канал
      - [B, 3, H, W] цветная RGB-маска → автоматически маппится в индексы
    Число каналов логитов приводится к num_classes из JSON.
    """
    def __init__(
        self,
        class_info_path: str,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1e-5,
    ):
        super().__init__()
        info = json.load(open(class_info_path, 'r'))
        self.num_classes = int(info['num_classes'])
        # веса для CE
        self.register_buffer(
            'class_weights',
            torch.tensor(info['class_weights'], dtype=torch.float)
        )
        # цвета для маппинга RGB→idx, shape=[num_classes,3]
        self.register_buffer(
            'class_colors',
            torch.tensor(info['class_colors'], dtype=torch.long)
        )
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def _prepare_target(self, target: torch.Tensor) -> torch.LongTensor:
        """
        Приводит любой target к карте индексов [B,H,W], dtype=int64.
        """
        if target.dim() == 3:
            return target.long()
        elif target.dim() == 4 and target.shape[1] == 1:
            return target.squeeze(1).long()
        elif target.dim() == 4 and target.shape[1] == 3:
            B, _, H, W = target.shape
            # из [0..1]→[0..255]
            rgb = (target * 255.0).round().long()
            flat = rgb.permute(0,2,3,1).reshape(-1,3)  # [B*H*W,3]
            eq = (flat.unsqueeze(1) == self.class_colors.unsqueeze(0)).all(dim=-1)
            idx = eq.float().argmax(dim=1)            # [B*H*W]
            return idx.reshape(B, H, W)
        else:
            raise ValueError(f'Unsupported target shape {tuple(target.shape)}')

    def forward(
        self,
        prediction: torch.Tensor,       # [B, C_model, H, W]
        target: torch.Tensor,       # как в описании
        mask: torch.Tensor = None,  # [B, H, W] или [B,1,H,W] или [B,3,H,W]
    ) -> torch.Tensor:
        B, C_model, H, W = prediction.shape
        device = prediction.device

        # 1) приводим логиты к нужному числу каналов
        if C_model < self.num_classes:
            raise ValueError(
                f'Logits have {C_model} channels < num_classes={self.num_classes}'
            )
        elif C_model > self.num_classes:
            # просто обрезаем «лишние» каналы справа
            prediction = prediction[:, :self.num_classes, :, :]

        # 2) готовим таргет [B,H,W]
        tgt_idx = self._prepare_target(target).to(device)

        # 3) weighted CE вручную
        log_probs = F.log_softmax(prediction, dim=1)        # [B,NC,H,W]
        wpx       = self.class_weights[tgt_idx]        # [B,H,W]
        logp      = log_probs.gather(1, tgt_idx.unsqueeze(1)).squeeze(1)  # [B,H,W]
        ce_map    = - wpx * logp                       # [B,H,W]

        # 4) маскирование CE
        if mask is not None:
            # приводим mask→[B,H,W]
            if mask.dim() == 4:
                if mask.shape[1] == 1:
                    m2d = mask.squeeze(1)
                else:
                    m2d = mask.any(dim=1)
            else:
                m2d = mask
            m2d = m2d.float().to(device)
            ce_map = ce_map * m2d
            denom  = m2d.sum()
        else:
            denom = B * H * W

        ce_loss = ce_map.sum() / (denom + 1e-6)

        # 5) Dice loss
        probs = torch.exp(log_probs)                   # [B,NC,H,W]
        tgt_oh = F.one_hot(                             # [B,H,W,NC]
            tgt_idx, num_classes=self.num_classes
        ).permute(0,3,1,2).float()                      # [B,NC,H,W]

        if mask is not None:
            m4d   = m2d.unsqueeze(1)                    # [B,1,H,W]
            probs = probs * m4d
            tgt_oh = tgt_oh * m4d

        dims       = (0,2,3)
        inter      = torch.sum(probs * tgt_oh, dims)
        card       = torch.sum(probs + tgt_oh, dims)
        dice_score = (2 * inter + self.smooth) / (card + self.smooth)
        dice_loss  = torch.mean(1. - dice_score)

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss