import os
import logging
import numpy as np
import torch
from PIL import Image
from .base_dataset import BaseDataset, PerceptionFileNameMode


class FloorplanSegmentationDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(
            min_depth=0.0,
            max_depth=1e5,
            has_filled_depth=False,
            name_mode=PerceptionFileNameMode.rgb_i_d,
            **kwargs,
        )

    def _get_data_path(self, index):
        filename_line = self.filenames[index]

        # Get data path
        rgb_rel_path = filename_line[0]

        depth_rel_path, filled_rel_path, normal_rel_path, matting_rel_path, dis_rel_path, seg_rel_path = None, None, None, None, None, None
        seg_rel_path = filename_line[1]

        return rgb_rel_path, depth_rel_path, filled_rel_path, normal_rel_path, matting_rel_path, dis_rel_path, seg_rel_path
