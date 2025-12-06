# main_split.py
import os
import torch
import torch.nn as nn
import numpy as np
import json
import pandas as pd
from collections import OrderedDict

# <<< 改动 1: 从新文件中导入模块 >>>
# 建议将修改后的文件重命名，以区分旧版本
from SurvivalModel_reverse_v3 import SurvivalModel 
from Trainer_test import Trainer         # 导入新的 Trainer
from Dataset_test import get_dataloaders # 导入新的 get_dataloaders

# --- LamedConfig 导入 (保持不变) ---
try:
    from m3d_lamed_model_0810.configuration_d3d_lamed import LamedConfig
except ImportError as e:
    print(f"警告: 无法导入 LamedConfig: {e}。将使用占位符。")
    class LamedConfig: image_size, hidden_size = (48, 256, 256), 768

# <<< 改动 2: evaluate_model 函数已不再需要，可以安全删除 >>>
# Trainer 内部已经包含了更完善的评估逻辑。

def main():
    class Config:
        # --- 路径与基础设置 ---
        # <<< 改动 3: CSV_PATH 现在指向带有 'split' 列的文件 >>>
        CSV_PATH = r"/data/yuanjiahong/yhh/code/utils_1024/id_adress_pfs_kfolder_v4_test.csv"
        MODEL_CONFIG_PATH = r"/data/yuanjiahong/yhh/llm_open_dig_4-15/config.json"
        VISION_WEIGHTS_PATH = r"/data/yuanjiahong/yhh/code/utils/vision_tower_only_weights_0810.pt" 
        OUTPUT_DIR_BASE = '/data/yuanjiahong/yhh/code/saved_models/ablation' # 建议为新实验建一个新目录

        RESUME_FROM_PATH = None 

        # --- 核心训练设置 ---
        # <<< 改动 4: FOLD_TO_TRAIN 已被移除，不再需要 >>>
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        NUM_EPOCHS = 200
        BATCH_SIZE = 8
        WEIGHT_DECAY = 1e-5
        EARLY_STOPPING_PATIENCE = 50 # 建议设一个合理的值，200太大了
        FOCAL_LOSS_GAMMA = 2.0 

        # --- 模型与实验模式 ---
        ATTENTION_MODE = 'dissimilar'
        
        # --- 训练策略参数 ---
        TIME_POINTS = np.array([3.0, 6.0, 9.0, 12.0, 15.0])
        STAGE1_EPOCHS = 1
        NUM_EARLY_BINS = 1
        STAGE2_LR_RATIO = 0.5
        CONSISTENCY_WEIGHT = 0.2
        
        UNFREEZE_EPOCH = 2
        FINETUNE_UNFREEZE_LAYERS = 2
        
        INITIAL_LEARNING_RATE = 1e-4
        FINETUNE_LEARNING_RATE = 1e-5
        
        # --- 模型超参数 ---
        TRANSFORMER_NHEAD = 8
        TRANSFORMER_DIM_FEEDFORWARD_RATIO = 2
        DROPOUT_RATE = 0.3
        TASK_SPECIFIC_HIDDEN_DIM = 256

    cfg = Config()
    
    # <<< 改动 5: 实验命名不再需要 fold 信息 >>>
    run_name = (f'mode_{cfg.ATTENTION_MODE}_consist{cfg.CONSISTENCY_WEIGHT}_'
                f'gamma{cfg.FOCAL_LOSS_GAMMA}_lr{cfg.INITIAL_LEARNING_RATE}_test')
    output_dir = os.path.join(cfg.OUTPUT_DIR_BASE, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"所有输出将保存在: {output_dir}")

    # ==================== 1. 准备数据 ====================
    print("\n--- 1. 准备数据加载器 ---")
    try:
        with open(cfg.MODEL_CONFIG_PATH, 'r') as f: model_config_dict = json.load(f)
        model_config = LamedConfig(**model_config_dict)
    except Exception as e:
        print(f"加载模型配置文件时出错: {e}"); return

    # <<< 改动 6: 调用新的 get_dataloaders，不再需要 fold_idx >>>
    dataloaders = get_dataloaders(
        csv_path=cfg.CSV_PATH, 
        image_size=model_config.image_size,
        time_points=cfg.TIME_POINTS, 
        batch_size=cfg.BATCH_SIZE, 
        num_workers=4
    )
    
    # <<< 改动 7: "动态教师选择"逻辑更新，只使用训练集 >>>
    # 确保只在训练集上计算中位数，避免信息泄露
    full_df = pd.read_csv(cfg.CSV_PATH)
    train_df = full_df[full_df['split'] == 'train']
    pfs_median = train_df['PFS'].median()
    time_points_array = np.array(cfg.TIME_POINTS)
    best_teacher_index = (np.abs(time_points_array - pfs_median)).argmin()
    
    print(f"\n--- 动态教师选择 (基于训练集) ---")
    print(f"训练集PFS中位数: {pfs_median:.2f} 月")
    print(f"选择的最佳教师索引: {best_teacher_index} (对应时间点: {time_points_array[best_teacher_index]} 月)")

    # ==================== 2. 准备模型 (几乎不变) ====================
    print(f"\n--- 2. 准备模型 (模式: {cfg.ATTENTION_MODE}) ---")
    model = SurvivalModel(
        vision_tower_config=model_config, 
        weights_path=cfg.VISION_WEIGHTS_PATH,
        num_time_bins=1, # 用于 Cox head，保持为1
        num_multiclass=len(cfg.TIME_POINTS),
        transformer_nhead=cfg.TRANSFORMER_NHEAD,
        transformer_dim_feedforward=model_config.hidden_size * cfg.TRANSFORMER_DIM_FEEDFORWARD_RATIO,
        dropout_rate=cfg.DROPOUT_RATE, 
        task_specific_hidden_dim=cfg.TASK_SPECIFIC_HIDDEN_DIM,
        attention_mode=cfg.ATTENTION_MODE
    )
    
    if cfg.RESUME_FROM_PATH and os.path.exists(cfg.RESUME_FROM_PATH):
        print(f"\n--- 正在从检查点恢复训练: {cfg.RESUME_FROM_PATH} ---")
        state_dict = torch.load(cfg.RESUME_FROM_PATH, map_location='cpu')
        model.load_state_dict(state_dict, strict=True) # 简化了加载逻辑
    
    if torch.cuda.device_count() > 1:
        print(f"\n检测到 {torch.cuda.device_count()} 个 GPU，启用 DataParallel。")
        model = nn.DataParallel(model)
    model.to(cfg.DEVICE)

    # ==================== 3. 准备训练器 (几乎不变) ====================
    print("\n--- 3. 准备训练器 ---")
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
    
    # ==================== 4. 开始训练 ====================
    print("\n" + "="*60 + "\n开始混合策略训练...\n" + "="*60)
    
    # trainer.fit 会处理所有事情：训练、验证、早停、保存模型、以及最终评估
    trained_model, history_df = trainer.fit(
        stage1_epochs=cfg.STAGE1_EPOCHS,
        stage2_lr_ratio=cfg.STAGE2_LR_RATIO,
        unfreeze_epoch=cfg.UNFREEZE_EPOCH,
        finetune_lr=cfg.FINETUNE_LEARNING_RATE,
        num_layers_to_finetune=cfg.FINETUNE_UNFREEZE_LAYERS,
        early_stopping_patience=cfg.EARLY_STOPPING_PATIENCE,
        output_dir=output_dir
    )

    # <<< 改动 8: 移除独立的最终评估部分 >>>
    # Trainer 内部的 fit 函数末尾已经自动完成了最终评估，无需在此重复。
    print("\n" + "="*60)
    print("训练和最终评估流程已全部完成。")
    print(f"所有结果均已保存在目录: {output_dir}")
    print("="*60)

if __name__ == '__main__':
    # 设置随机种子以保证实验可复现
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    main()