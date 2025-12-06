import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import monai.transforms as transforms
from monai.data import ThreadDataLoader
import os
from torch.nn.utils.rnn import pad_sequence

# get_transforms 函数保持不变
def get_transforms(mode, image_size):
    image_key = ["image"]
    load_and_ensure_channel = [
        transforms.LoadImaged(keys=image_key, image_only=True, ensure_channel_first=True),
    ]
    # <<< 改动 1: 验证集和测试集都不应使用数据增强 >>>
    # 原逻辑只有 train 和 else，现在明确 val 和 test 都不增强
    if mode == 'train':
        data_augmentation = [
            transforms.RandFlipd(keys=image_key, prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=image_key, prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=image_key, prob=0.5, spatial_axis=2),
        ]
    else: # mode is 'val' or 'test'
        data_augmentation = []
        
    post_processing = [
        transforms.Resized(keys=image_key, spatial_size=image_size, mode="trilinear"),
        transforms.ScaleIntensityRanged(keys=image_key, a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.ToTensord(keys=image_key) # 确保输出是Tensor
    ]
    return transforms.Compose(load_and_ensure_channel + data_augmentation + post_processing)

# <<< 改动 2: SurvivalDataset 类修改 >>>
class SurvivalDataset(Dataset):
    # fold_idx 参数不再需要，已移除
    def __init__(self, csv_path, mode, image_size, time_points: np.ndarray):
        # 允许 mode 为 'train', 'val', 'test'
        if mode not in ['train', 'val', 'test']: 
            raise ValueError("mode 必须是 'train', 'val' 或 'test'")
        self.mode, self.image_size = mode, image_size
        
        self.time_points = torch.tensor(time_points, dtype=torch.float32)
        self.num_multiclass = len(time_points)
        
        # 尝试用更健壮的方式读取CSV
        try:
            full_df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            full_df = pd.read_csv(csv_path, encoding='gbk')
        
        # 核心改动：根据 'split' 列筛选数据，而不是 'k' 列
        if 'split' not in full_df.columns:
            raise ValueError(f"错误: CSV文件 '{csv_path}' 中未找到 'split' 列。请先运行数据划分脚本。")
        
        self.df = full_df[full_df['split_std'] == self.mode].reset_index(drop=True)
            
        print(f"--- {self.mode.capitalize()} Dataset Initialized ---")
        # 移除了打印 Fold 的信息
        print(f"Mode: {self.mode}, Target Time Points: {time_points.tolist()}, Samples: {len(self.df)}")
        self.transform = get_transforms(mode=self.mode, image_size=self.image_size)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        # __getitem__ 内部逻辑完全不变，因为它只关心 self.df 的内容
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

# custom_collate_fn 函数保持不变
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

# <<< 改动 3: get_dataloaders 函数大改 >>>
def get_dataloaders(csv_path, image_size, time_points, batch_size=4, num_workers=4):
    """
    根据 'split' 列创建训练、验证和测试集的 DataLoader。
    """
    # 实例化三个数据集
    train_dataset = SurvivalDataset(
        csv_path=csv_path, mode='train', image_size=image_size, time_points=time_points
    )
    val_dataset = SurvivalDataset(
        csv_path=csv_path, mode='test', image_size=image_size, time_points=time_points
    )
    test_dataset = SurvivalDataset(
        csv_path=csv_path, mode='val', image_size=image_size, time_points=time_points
    )
    
    print("\n--- Creating Train/Val/Test DataLoaders ---")
    
    # 创建三个 DataLoader
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
    
    # 返回一个包含三个 loader 的字典
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}

# <<< 改动 4: 全新、功能更全面的测试脚本 >>>
if __name__ == '__main__':
    # 假设你已经运行了 prepare_dataset_split.py 并生成了这个文件
    # !! 请确保这个路径是正确的 !!
    CSV_FILE_PATH = r"/data/yuanjiahong/yhh/code/utils_1024/id_adress_pfs_kfolder_v4_testeasy811.csv"

    MODEL_IMAGE_SIZE = (48, 256, 256) 
    BATCH_SIZE = 4
    FIVE_TIME_POINTS = np.array([6.0, 8.0, 10.0, 12.0, 16.0])
    NUM_TIME_POINTS = len(FIVE_TIME_POINTS)

    # --- 模拟数据生成 ---
    if not os.path.exists(CSV_FILE_PATH):
        print(f"\n[警告] 未找到CSV文件 '{CSV_FILE_PATH}'。将创建一个模拟CSV文件用于调试。")
        dummy_data_dir = "./dummy_patient_data"
        os.makedirs(dummy_data_dir, exist_ok=True)
        dummy_nii_path = os.path.join(dummy_data_dir, "scan1.nii.gz")
        if not os.path.exists(dummy_nii_path):
            import SimpleITK as sitk
            dummy_image = sitk.GetImageFromArray(np.zeros(MODEL_IMAGE_SIZE, dtype=np.int16))
            sitk.WriteImage(dummy_image, dummy_nii_path)

        dummy_data = {
            'output_filepath': [dummy_nii_path] * 10,
            'PFS': [5.0, 7.0, 9.5, 11.0, 15.0, 18.0, 13.0, 20.0, 4.0, 22.0],
            'is_progress': [1, 1, 1, 0, 1, 1, 0, 0, 1, 0],
            'k': [0]*10,
            'split': ['train']*7 + ['val']*1 + ['test']*2
        }
        pd.DataFrame(dummy_data).to_csv(CSV_FILE_PATH, index=False)
        print(f"模拟CSV文件已创建在 '{CSV_FILE_PATH}'\n")

    print("="*70)
    print(" " * 15 + "DETAILED TEST: Train/Val/Test Data Pipeline")
    print("="*70)
    
    try:
        # --- 1. 创建 DataLoaders ---
        dataloaders = get_dataloaders(
            csv_path=CSV_FILE_PATH,
            image_size=MODEL_IMAGE_SIZE,
            time_points=FIVE_TIME_POINTS,
            batch_size=BATCH_SIZE,
            num_workers=0 
        )
        
        # --- 2. 验证 DataLoaders 结构 ---
        print("\n--- [检查 1] DataLoader 结构验证 ---")
        expected_keys = ['train', 'val', 'test']
        print(f" > 返回的字典包含的键: {list(dataloaders.keys())}")
        assert all(k in dataloaders for k in expected_keys), "错误: 字典中缺少必要的键！"
        print(" > 结构验证通过！\n")

        # --- 3. 逐一检查每个 DataLoader ---
        full_df = pd.read_csv(CSV_FILE_PATH)
        for split_name, loader in dataloaders.items():
            print("-" * 70)
            #print(f"--- [检查] 详细分析 <{split_name.toUpperCase()}> DataLoader ---")
            
            expected_samples = len(full_df[full_df['split_std'] == split_name])
            actual_samples = len(loader.dataset)
            print(f" > 样本数: 预期(CSV中)={expected_samples}, 实际(Dataset中)={actual_samples}")
            assert expected_samples == actual_samples, "错误: 样本数不匹配！"

            if not loader:
                print(" > Loader 为空，跳过后续检查。")
                continue

            batch_data = next(iter(loader))
            print(" > 成功获取一个数据批次！")

            print("\n   --- Batch 内容详细信息 ---")
            current_batch_size = batch_data['images'].shape[0]
            for key, tensor in batch_data.items():
                print(f"   - {key:<18}: shape={list(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}")
            
            # <<< 核心修正: 从 [1:] 改为 [2:]，以适应变长的序列 >>>
            # 这个断言检查的是(通道数, 深度, 高度, 宽度)
            assert batch_data['images'].shape[2:] == (3, *MODEL_IMAGE_SIZE), "影像张量形状错误"
            
            assert len(batch_data['lengths']) == current_batch_size, "lengths 长度错误"
            assert batch_data['multiclass_labels'].shape == (current_batch_size, NUM_TIME_POINTS), "标签形状错误"
            assert batch_data['multiclass_masks'].shape == (current_batch_size, NUM_TIME_POINTS), "掩码形状错误"
            print("   > 所有张量形状符合预期！")

            print("\n   --- 第一个样本内容抽样检查 ---")
            idx = 0
            time_sample = batch_data['time'][idx].item()
            event_sample = batch_data['event'][idx].item()
            labels_sample = batch_data['multiclass_labels'][idx].tolist()
            masks_sample = batch_data['multiclass_masks'][idx].tolist()

            print(f"   > 原始数据: PFS={time_sample:.2f} 月, Event={'发生' if event_sample == 1 else '删失'}")
            print(f"   > 目标时间点: {FIVE_TIME_POINTS.tolist()}")
            print(f"   > 生成的标签: {labels_sample}")
            print(f"   > 生成的掩码: {masks_sample}")
            
            expected_labels, expected_masks = [], []
            for t_point in FIVE_TIME_POINTS:
                if time_sample >= t_point:
                    expected_labels.append(1.0); expected_masks.append(1.0)
                else:
                    if event_sample == 1:
                        expected_labels.append(0.0); expected_masks.append(1.0)
                    else:
                        expected_labels.append(0.0); expected_masks.append(0.0)
            assert labels_sample == expected_labels, "错误: 标签生成逻辑不匹配！"
            assert masks_sample == expected_masks, "错误: 掩码生成逻辑不匹配！"
            print("   > 抽样检查通过，标签和掩码生成逻辑正确！")
        
        print("\n" + "="*70)
        print(" " * 10 + "🎉 Train/Val/Test 数据管道升级成功，所有详细检查通过！ 🎉")
        print("="*70)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[错误] 测试过程中发生异常: {e}")