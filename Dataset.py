import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import monai.transforms as transforms
from monai.data import ThreadDataLoader
import os
from torch.nn.utils.rnn import pad_sequence

def get_transforms(mode, image_size):
    image_key = ["image"]
    load_and_ensure_channel = [
        transforms.LoadImaged(keys=image_key, image_only=True, ensure_channel_first=True),
    ]
    if mode == 'train':
        data_augmentation = [
            transforms.RandFlipd(keys=image_key, prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=image_key, prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=image_key, prob=0.5, spatial_axis=2),
        ]
    else: 
        data_augmentation = []
        
    post_processing = [
        transforms.Resized(keys=image_key, spatial_size=image_size, mode="trilinear"),
        transforms.ScaleIntensityRanged(keys=image_key, a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.ToTensord(keys=image_key) 
    ]
    return transforms.Compose(load_and_ensure_channel + data_augmentation + post_processing)


class SurvivalDataset(Dataset):

    def __init__(self, csv_path, mode, image_size, time_points: np.ndarray):
        if mode not in ['train', 'val', 'test']: 
            raise ValueError("mode must be 'train', 'val' 或 'test'")
        self.mode, self.image_size = mode, image_size
        
        self.time_points = torch.tensor(time_points, dtype=torch.float32)
        self.num_multiclass = len(time_points)
        
        try:
            full_df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            full_df = pd.read_csv(csv_path, encoding='gbk')
        
        if 'split' not in full_df.columns:
            raise ValueError(f"Error: 'split' not in CSV '{csv_path}'")
        
        self.df = full_df[full_df['split_std'] == self.mode].reset_index(drop=True)
            
        print(f"--- {self.mode.capitalize()} Dataset Initialized ---")
        print(f"Mode: {self.mode}, Target Time Points: {time_points.tolist()}, Samples: {len(self.df)}")
        self.transform = get_transforms(mode=self.mode, image_size=self.image_size)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        patient_row = self.df.loc[idx]
        base_filepath = patient_row['output_filepath']
        patient_dir = os.path.dirname(base_filepath)
        
        image_paths = []
        if os.path.exists(patient_dir):
            for filename in sorted(os.listdir(patient_dir)):
                if '.nii' in filename or '.gz' in filename:
                    image_paths.append(os.path.join(patient_dir, filename))
        
        image_tensors_list = [self.transform({'image': path})['image'] for path in image_paths]

        if image_tensors_list:
            images_tensor = torch.stack(image_tensors_list, dim=0)
            if images_tensor.shape[1] == 1:
                images_tensor = images_tensor.repeat(1, 3, 1, 1, 1)
        else:
            images_tensor = torch.empty((0, 3, *self.image_size), dtype=torch.float32)

        time, event = patient_row['PFS'], patient_row['is_progress']
        
        multiclass_labels = torch.zeros(self.num_multiclass, dtype=torch.float32)
        multiclass_masks = torch.zeros(self.num_multiclass, dtype=torch.float32)
        
        for i, t_point in enumerate(self.time_points):
            if time >= t_point:
                multiclass_labels[i] = 1.0
                multiclass_masks[i] = 1.0
            else:
                if event == 1:
                    multiclass_labels[i] = 0.0
                    multiclass_masks[i] = 1.0
                else:
                    multiclass_labels[i] = 0.0
                    multiclass_masks[i] = 0.0

        return {
            "images": images_tensor,
            "time": torch.tensor(time, dtype=torch.float32),
            "event": torch.tensor(event, dtype=torch.float32),
            "multiclass_labels": multiclass_labels,
            "multiclass_masks": multiclass_masks
        }

def custom_collate_fn(batch):
    image_sequences = [item['images'] for item in batch]
    lengths = torch.tensor([seq.shape[0] for seq in image_sequences], dtype=torch.int64)
    
    padded_images = pad_sequence(image_sequences, batch_first=True, padding_value=0)
    
    times = torch.stack([item['time'] for item in batch])
    events = torch.stack([item['event'] for item in batch])
    multiclass_labels = torch.stack([item['multiclass_labels'] for item in batch])
    multiclass_masks = torch.stack([item['multiclass_masks'] for item in batch])

    return {
        'images': padded_images,
        'lengths': lengths,
        'time': times,
        'event': events,
        'multiclass_labels': multiclass_labels,
        'multiclass_masks': multiclass_masks
    }

def get_dataloaders(csv_path, image_size, time_points, batch_size=4, num_workers=4):

    train_dataset = SurvivalDataset(
        csv_path=csv_path, mode='train', image_size=image_size, time_points=time_points
    )
    val_dataset = SurvivalDataset(
        csv_path=csv_path, mode='val', image_size=image_size, time_points=time_points
    )
    test_dataset = SurvivalDataset(
        csv_path=csv_path, mode='test', image_size=image_size, time_points=time_points
    )
    
    print("\n--- Creating Train/Val/Test DataLoaders ---")
    
    train_loader = ThreadDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, collate_fn=custom_collate_fn, pin_memory=True
    )
    val_loader = ThreadDataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=custom_collate_fn, pin_memory=True
    )
    test_loader = ThreadDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=custom_collate_fn, pin_memory=True
    )
    
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}
