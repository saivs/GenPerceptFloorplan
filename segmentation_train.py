import json
import torch
import os
from omegaconf import OmegaConf
from src.util.loss import CombinedSegLoss
from src.dataset.segmentation_dataset import SegmentationDataset

def load_class_info(json_path):
    """
    Загружает информацию о классах из JSON файла
    
    Args:
        json_path (str): Путь к файлу class_info.json
        
    Returns:
        dict: Словарь с информацией о классах
    """
    with open(json_path, 'r') as f:
        class_info = json.load(f)
    return class_info

def create_loss_with_class_weights(class_info_path, ce_weight=1.0, dice_weight=1.0, focal_weight=0.5):
    """
    Создает функцию потерь с весами классов из файла class_info.json
    
    Args:
        class_info_path (str): Путь к файлу class_info.json
        ce_weight (float): Вес для Cross Entropy loss
        dice_weight (float): Вес для Dice loss
        focal_weight (float): Вес для Focal loss
        
    Returns:
        CombinedSegLoss: Функция потерь с заданными весами классов
    """
    # Загрузка информации о классах
    class_info = load_class_info(class_info_path)
    
    # Получение весов классов
    class_weights = torch.tensor(class_info['class_weights'], dtype=torch.float32)
    
    # Создание функции потерь
    loss_fn = CombinedSegLoss(
        num_classes=class_info['num_classes'],
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        focal_weight=focal_weight,
        class_weights=class_weights,
        batch_reduction=True,
        return_dict=True
    )
    
    return loss_fn

def create_dataset_with_class_info(config_path, class_info_path):
    """
    Создает датасет с информацией о классах из файла class_info.json
    
    Args:
        config_path (str): Путь к файлу конфигурации
        class_info_path (str): Путь к файлу class_info.json
        
    Returns:
        SegmentationDataset: Датасет с предварительно загруженной информацией о классах
    """
    # Загрузка конфигурации
    cfg = OmegaConf.load(config_path)
    
    # Загрузка информации о классах
    class_info = load_class_info(class_info_path)
    
    # Создание датасета
    dataset = SegmentationDataset(
        mode=cfg.mode,
        filename_ls_path=cfg.filenames,
        dataset_dir=os.path.join(cfg.base_data_dir, cfg.dataset_dir),
        disp_name=cfg.disp_name,
        name_mode=cfg.name_mode,
        augmentation_args=cfg.augmentation_args if hasattr(cfg, 'augmentation_args') else None,
        resize_to_hw=cfg.resize_to_hw if hasattr(cfg, 'resize_to_hw') else None,
        class_colors=class_info['class_colors'],  # Предварительно загруженные цвета классов
        class_indices=class_info['class_indices'],  # Предварительно загруженные индексы классов
        num_classes=class_info['num_classes']  # Предварительно загруженное количество классов
    )
    
    return dataset

# Пример использования
if __name__ == "__main__":
    # Пути к файлам
    class_info_path = "config/class_info.json"
    config_path = "config/dataset/dataset_train.yaml"
    
    # Создание функции потерь с весами классов
    loss_fn = create_loss_with_class_weights(class_info_path)
    print(f"Created loss function with class weights from {class_info_path}")
    
    # Создание датасета с информацией о классах
    dataset = create_dataset_with_class_info(config_path, class_info_path)
    print(f"Created dataset with class information from {class_info_path}")