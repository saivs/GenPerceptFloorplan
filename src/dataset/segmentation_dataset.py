import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode

from .base_dataset import BaseDataset, PerceptionFileNameMode, DatasetMode

class SegmentationDataset(BaseDataset):
    def __init__(
        self,
        mode: DatasetMode,
        filename_ls_path: str,
        dataset_dir: str,
        disp_name: str,
        name_mode: PerceptionFileNameMode = PerceptionFileNameMode.id,
        augmentation_args: dict = None,
        resize_to_hw=None,
        rgb_transform=lambda x: x / 255.0 * 2 - 1,  # [0, 255] -> [-1, 1]
        class_colors=None,  # Предварительно загруженные цвета классов
        class_indices=None,  # Предварительно загруженные индексы классов
        num_classes=None,   # Предварительно загруженное количество классов
        **kwargs,
    ) -> None:
        super().__init__(
            mode=mode,
            filename_ls_path=filename_ls_path,
            dataset_dir=dataset_dir,
            disp_name=disp_name,
            name_mode=name_mode,
            min_depth=0,
            max_depth=1e8,
            has_filled_depth=False,
            depth_transform=None,
            augmentation_args=augmentation_args,
            resize_to_hw=resize_to_hw,
            rgb_transform=rgb_transform,
            **kwargs,
        )
        
        # Если предоставлены предварительно загруженные данные о классах, используем их
        if class_colors is not None and class_indices is not None and num_classes is not None:
            # Преобразование строковых ключей в кортежи для class_indices
            if isinstance(class_indices, dict):
                self.class_indices = {}
                for key, value in class_indices.items():
                    # Преобразование ключа '0_0_255' в кортеж (0, 0, 255)
                    if '_' in key:
                        tuple_key = tuple(map(int, key.split('_')))
                        self.class_indices[tuple_key] = value
                    else:
                        self.class_indices[key] = value
            else:
                self.class_indices = class_indices
            
            self.class_colors = [tuple(color) for color in class_colors]
            self.num_classes = num_classes
            print(f"Using pre-loaded class information: {self.num_classes} classes")
        else:
            # Инициализируем из первой маски сегментации, как в оригинальном классе
            self.class_colors = None
            self.class_indices = None
            self.num_classes = None
            
            # Инициализация отображения классов из первой маски сегментации
            if len(self.filenames) > 0 and len(self.filenames[0]) >= 2:
                self._initialize_class_mapping(self.filenames[0][1])

    def _initialize_class_mapping(self, first_seg_path):
        seg_img = self._read_image(first_seg_path, convert_rgb=True)
        
        # Reshape to get unique colors
        unique_colors = np.unique(seg_img.reshape(-1, seg_img.shape[-1]), axis=0)
        
        # Create color to index mapping
        self.class_colors = [tuple(color) for color in unique_colors]
        self.class_indices = {tuple(color): idx for idx, color in enumerate(unique_colors)}
        self.num_classes = len(self.class_colors)
        
        print(f"Found {self.num_classes} unique classes in segmentation data")
        print(f"Class colors: {self.class_colors}")

    def _color_to_class_indices(self, seg_img):
        H, W, C = seg_img.shape
        seg_indices = np.zeros((H, W), dtype=np.int64)
        
        # Reshape to process all pixels at once
        pixels = seg_img.reshape(-1, C)
        
        # For each color class, find matching pixels and assign class index
        for color, class_idx in self.class_indices.items():
            # Match pixels with this color
            color_array = np.array(color)
            matches = np.all(pixels == color_array, axis=1)
            
            # Set class index for matching pixels
            seg_indices.flat[matches] = class_idx
            
        return seg_indices

    def _class_indices_to_color(self, seg_indices):
        H, W = seg_indices.shape
        seg_img = np.zeros((H, W, 3), dtype=np.uint8)
        
        for class_idx, color in enumerate(self.class_colors):
            mask = (seg_indices == class_idx)
            seg_img[mask] = color
            
        return seg_img

    def _get_data_path(self, index):
        filename_line = self.filenames[index]
        
        # Get data path
        rgb_rel_path = filename_line[0]
        
        seg_rel_path = None
        if DatasetMode.RGB_ONLY != self.mode and len(filename_line) >= 2:
            seg_rel_path = filename_line[1]
        
        # Fill other paths with None to match BaseDataset's expected return values
        depth_rel_path = None
        filled_rel_path = None
        normal_rel_path = None
        matting_rel_path = None
        dis_rel_path = None
            
        return rgb_rel_path, depth_rel_path, filled_rel_path, normal_rel_path, matting_rel_path, dis_rel_path, seg_rel_path

    def _get_data_item(self, index):
        rgb_rel_path, _, _, _, _, _, seg_rel_path = self._get_data_path(index=index)

        rasters = {}

        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        if DatasetMode.RGB_ONLY != self.mode:
            # Segmentation data
            seg_data = self._load_seg_data(
                seg_rel_path=seg_rel_path, 
                shape=(rasters["rgb_norm"].shape[1], rasters["rgb_norm"].shape[2]),
            )
            rasters.update(seg_data)
            
            # Create valid mask (all pixels are valid for segmentation)
            H, W = rasters["rgb_norm"].shape[1:3]
            valid_mask = torch.ones((1, H, W), dtype=torch.bool)
            rasters["valid_mask_raw_seg"] = valid_mask

        other = {"index": index, "rgb_relative_path": rgb_rel_path}

        return rasters, other

    def _load_seg_data(self, seg_rel_path, shape):
        outputs = {}
        
        try:
            # Read segmentation image
            seg_img = self._read_image(seg_rel_path, convert_rgb=True)
            
            # Convert segmentation image to class indices (one-hot encoding)
            seg_indices = self._color_to_class_indices(seg_img)
            
            # Convert to PyTorch tensor
            # We'll store both the raw RGB segmentation and the class indices
            seg_raw = np.transpose(seg_img, (2, 0, 1))  # [rgb, H, W]
            seg_raw_linear = torch.from_numpy(seg_raw).float()  # [3, H, W]
            outputs["seg_raw_linear"] = seg_raw_linear.clone()
            
            # Store class indices tensor
            seg_indices_tensor = torch.from_numpy(seg_indices).long()  # [H, W]
            outputs["seg_class_indices"] = seg_indices_tensor
            
            # Store number of classes for reference
            outputs["num_classes"] = self.num_classes
            
        except Exception as e:
            print(f"Error loading segmentation data: {e}")
            # Create empty segmentation data with correct shape
            seg_raw = np.zeros((3, shape[0], shape[1]), dtype=np.float32)
            seg_raw_linear = torch.from_numpy(seg_raw).float()  # [3, H, W]
            outputs["seg_raw_linear"] = seg_raw_linear.clone()
            
            # Create empty class indices
            seg_indices = np.zeros((shape[0], shape[1]), dtype=np.int64)
            outputs["seg_class_indices"] = torch.from_numpy(seg_indices).long()
            outputs["num_classes"] = self.num_classes

        return outputs

    def _get_valid_mask_seg(self, seg: torch.Tensor):
        valid_mask_seg = (seg != -1).any(dim=(0))[None]
        return valid_mask_seg

    def _training_preprocess(self, rasters):
        # Call the parent class's preprocessing
        rasters = super()._training_preprocess(rasters)
        
        # Handle segmentation-specific preprocessing if needed
        if "seg_raw_linear" in rasters and "seg_raw_norm" not in rasters:
            # Normalize segmentation to [-1, 1] range for model input
            rasters["seg_raw_norm"] = (rasters["seg_raw_linear"].float() / 255.0) * 2.0 - 1.0
        
        return rasters