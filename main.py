# main_split.py
import os
import torch
import torch.nn as nn
import numpy as np
import json
import pandas as pd
from collections import OrderedDict

from SurvivalModel import SurvivalModel 
from Trainer import Trainer        
from Dataset import get_dataloaders 

try:
    from encoder.configuration import LamedConfig
except ImportError as e:
    print(f"警告: 无法导入 LamedConfig: {e}。将使用占位符。")
    class LamedConfig: image_size, hidden_size = (48, 256, 256), 768


def main():
    class Config:
        CSV_PATH = 'path/to/your/csv'
        MODEL_CONFIG_PATH = 'path/to/your/config/'
        VISION_WEIGHTS_PATH = 'path/to/your/weoght' 
        OUTPUT_DIR_BASE = 'path/to/your/result'

        RESUME_FROM_PATH = None 


        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        NUM_EPOCHS = 200
        BATCH_SIZE = 8
        WEIGHT_DECAY = 1e-5
        EARLY_STOPPING_PATIENCE = 50 
        FOCAL_LOSS_GAMMA = 2.0 

        ATTENTION_MODE = 'dissimilar'
        
        TIME_POINTS = np.array([3.0, 6.0, 9.0, 12.0, 15.0])
        STAGE1_EPOCHS = 1
        NUM_EARLY_BINS = 1
        STAGE2_LR_RATIO = 0.5
        CONSISTENCY_WEIGHT = 0.2
        
        UNFREEZE_EPOCH = 2
        FINETUNE_UNFREEZE_LAYERS = 2
        
        INITIAL_LEARNING_RATE = 1e-4
        FINETUNE_LEARNING_RATE = 1e-5
        
        TRANSFORMER_NHEAD = 8
        TRANSFORMER_DIM_FEEDFORWARD_RATIO = 2
        DROPOUT_RATE = 0.3
        TASK_SPECIFIC_HIDDEN_DIM = 256

    cfg = Config()
    
    run_name = (f'mode_{cfg.ATTENTION_MODE}_consist{cfg.CONSISTENCY_WEIGHT}_'
                f'gamma{cfg.FOCAL_LOSS_GAMMA}_lr{cfg.INITIAL_LEARNING_RATE}_test')
    output_dir = os.path.join(cfg.OUTPUT_DIR_BASE, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"所有输出将保存在: {output_dir}")


    print("\n--- 1. Prepare Data Loader ---")
    try:
        with open(cfg.MODEL_CONFIG_PATH, 'r') as f: model_config_dict = json.load(f)
        model_config = LamedConfig(**model_config_dict)
    except Exception as e:
        print(f"Fail to load config: {e}"); return

    dataloaders = get_dataloaders(
        csv_path=cfg.CSV_PATH, 
        image_size=model_config.image_size,
        time_points=cfg.TIME_POINTS, 
        batch_size=cfg.BATCH_SIZE, 
        num_workers=4
    )
    
    full_df = pd.read_csv(cfg.CSV_PATH)
    train_df = full_df[full_df['split'] == 'train']
    pfs_median = train_df['PFS'].median()
    time_points_array = np.array(cfg.TIME_POINTS)
    best_teacher_index = (np.abs(time_points_array - pfs_median)).argmin()
    
    print(f"\n--- Dynamic teacher time point select... ---")
    print(f"Teacher time point idx: {best_teacher_index} ({time_points_array[best_teacher_index]} months)")

    print(f"\n--- 2. Prepare Survival Model (Mode: {cfg.ATTENTION_MODE}) ---")
    model = SurvivalModel(
        vision_tower_config=model_config, 
        weights_path=cfg.VISION_WEIGHTS_PATH,
        num_time_bins=1, 
        num_multiclass=len(cfg.TIME_POINTS),
        transformer_nhead=cfg.TRANSFORMER_NHEAD,
        transformer_dim_feedforward=model_config.hidden_size * cfg.TRANSFORMER_DIM_FEEDFORWARD_RATIO,
        dropout_rate=cfg.DROPOUT_RATE, 
        task_specific_hidden_dim=cfg.TASK_SPECIFIC_HIDDEN_DIM,
        attention_mode=cfg.ATTENTION_MODE
    )
    
    if cfg.RESUME_FROM_PATH and os.path.exists(cfg.RESUME_FROM_PATH):
        state_dict = torch.load(cfg.RESUME_FROM_PATH, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    
    if torch.cuda.device_count() > 1:
        print(f"\nDetected {torch.cuda.device_count()} GPUs，use DataParallel")
        model = nn.DataParallel(model)
    model.to(cfg.DEVICE)

    print("\n--- 3. Prepare Trainer ---")
    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        num_epochs=cfg.NUM_EPOCHS,
        time_points=cfg.TIME_POINTS,
        learning_rate=cfg.INITIAL_LEARNING_RATE,
        focal_loss_gamma=cfg.FOCAL_LOSS_GAMMA,
        consistency_weight=cfg.CONSISTENCY_WEIGHT,
        weight_decay=cfg.WEIGHT_DECAY,
        teacher_index=best_teacher_index,
        num_early_bins=cfg.NUM_EARLY_BINS,
        device=cfg.DEVICE
    )
    
    trained_model, history_df = trainer.fit(
        stage1_epochs=cfg.STAGE1_EPOCHS,
        stage2_lr_ratio=cfg.STAGE2_LR_RATIO,
        unfreeze_epoch=cfg.UNFREEZE_EPOCH,
        finetune_lr=cfg.FINETUNE_LEARNING_RATE,
        num_layers_to_finetune=cfg.FINETUNE_UNFREEZE_LAYERS,
        early_stopping_patience=cfg.EARLY_STOPPING_PATIENCE,
        output_dir=output_dir
    )

    print("\n" + "="*60)
    print("Mention Accomplished! ")
    print("="*60)

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        

    main()

