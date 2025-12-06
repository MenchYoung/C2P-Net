#reversev3：全局池化--注意力
# '===== LOADING SurvivalModel (Version: Switchable Iterative Cross-Attention) ====='
print("===== LOADING SurvivalModel (Version: Switchable Iterative Cross-Attention) =====")

import torch
import torch.nn as nn
import json
import torch.nn.functional as F

# --- 导入和模拟类保持不变 ---
try:
    from m3d_lamed_model_0810.configuration_d3d_lamed import LamedConfig
    from m3d_lamed_model_0810.vison_tower_component0810 import Swin3DTower
except ImportError:
    class LamedConfig: hidden_size = 1024
    class Swin3DTower(nn.Module):
        def __init__(self, config): super().__init__(); self.hidden_size = config.hidden_size
        def forward(self, x): return torch.randn(x.shape[0], 2, self.hidden_size)
# --- 放在 SurvivalModel 类的定义之前 ---

class AttentionPooler(nn.Module):
    """
    一个简单的注意力池化模块。
    输入: 一个 token 序列 (B, N, D)
    输出: 一个全局特征向量 (B, D)
    """
    def __init__(self, d_model: int):
        super().__init__()
        # 这个小型网络会为每个 token 计算一个“重要性分数”
        self.attention_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, N, D)
        
        # 1. 计算每个 token 的原始分数
        # a_scores shape: (B, N, 1)
        a_scores = self.attention_net(x)
        
        # 2. 使用 softmax 将分数转换为权重 (在 N 维度上)
        # a_weights shape: (B, N, 1)
        a_weights = torch.softmax(a_scores, dim=1)
        
        # 3. 加权求和
        # (B, N, 1) * (B, N, D) -> 广播乘法
        # torch.sum 在 dim=1 上求和 -> (B, D)
        weighted_sum = torch.sum(a_weights * x, dim=1)
        
        return weighted_sum
    

class SurvivalModel(nn.Module):
    def __init__(self, vision_tower_config, weights_path,
                 num_time_bins: int, num_multiclass: int = 5,
                 transformer_nhead=8,
                 transformer_dim_feedforward=2048, dropout_rate=0.3, 
                 task_specific_hidden_dim=256,
                 # <<< 核心修改 1: 新增 attention_mode 参数 >>>
                 # 'similar' -> 标准交叉注意力 (查相同)
                 # 'dissimilar' -> 负点积交叉注意力 (查不同)
                 attention_mode: str = 'similar'):
        super().__init__()
        
        # --- 视觉编码器部分不变 ---
        self.vision_encoder = Swin3DTower(vision_tower_config)
        
        state_dict = torch.load(weights_path, map_location='cpu')
        self.vision_encoder.load_state_dict(state_dict)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        self.d_model = self.vision_encoder.hidden_size
        self.num_multiclass = num_multiclass
        
        # <<< 核心修改 2: 验证并保存 attention_mode >>>
        if attention_mode not in ['similar', 'dissimilar']:
            raise ValueError("attention_mode 必须是 'similar' 或 'dissimilar'")
        self.attention_mode = attention_mode
        print(f"  - [注意力模式] 当前模式设置为: '{self.attention_mode}'")

        self.feature_pooler = AttentionPooler(d_model=self.d_model)
        print(f"  - [新组件] Attention Pooling 模块已启用，用于聚合视觉特征。")

        # <<< 核心修正 3: 创建两个独立的交叉注意力层实例 >>>
        # 使用 nn.ModuleList 来规范地管理这两个层
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.d_model, nhead=transformer_nhead,
                dim_feedforward=transformer_dim_feedforward, dropout=dropout_rate,
                batch_first=True
            ) for _ in range(2) # 创建两个独立的层
        ])
        print(f"  - [新引擎] 迭代式交叉注意力模块构建完成 (d_model={self.d_model})。")

        # --- 任务头和自动加权参数部分不变 ---
        self.cox_task_head = nn.Sequential(
            nn.Linear(self.d_model, task_specific_hidden_dim),
            nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(task_specific_hidden_dim, num_time_bins)
        )
        print(f"  - [任务头1] Cox Head 构建完成。")

        self.multiclass_task_head = nn.Sequential(
            nn.Linear(self.d_model, task_specific_hidden_dim),
            nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(task_specific_hidden_dim, self.num_multiclass) # 使用 self.num_multiclass
        )
        print(f"  - [任务头2] Multiclass Head 构建完成。")

        # --- 自动加权参数 ---
        self.log_var_cox = nn.Parameter(torch.zeros(1))
        self.log_var_bce = nn.Parameter(torch.zeros(1))
        
        print("\n--- [可切换交叉注意力版] 多任务模型构建完成！---\n")

    def forward(self, images_padded, lengths):
        # --- 1. 特征提取 (不变) ---
        batch_size, padded_seq_len, C, D, H, W = images_padded.shape
        images_flattened = images_padded.view(batch_size * padded_seq_len, C, D, H, W)
        
        # 为了处理可变长度，我们需要知道哪些是真实的影像，哪些是填充的
        # 创建一个掩码，标记出真实影像的位置
        mask = torch.arange(padded_seq_len, device=lengths.device)[None, :] < lengths[:, None]
        valid_indices = mask.view(-1) # 将掩码展开成一维
        
        # 只将真实的影像送入编码器，避免浪费计算
        valid_images = images_flattened[valid_indices]
        if valid_images.shape[0] == 0: # 处理一个批次全是空序列的极端情况
            # 返回一个符合格式的零张量
            dummy_output = torch.zeros(batch_size, self.d_model, device=images_padded.device)
            return {
            "survival_logits": survival_logits,
            "multiclass_logits": multiclass_logits,
            "log_var_cox": self.log_var_cox,
            "log_var_bce": self.log_var_bce
        }

        features_output = self.vision_encoder(valid_images)
        global_features_flat = self.feature_pooler(features_output)
        
        # 创建一个正确形状的“画布”，然后把特征填回去
        features_sequential = torch.zeros(batch_size * padded_seq_len, self.d_model, device=images_padded.device)
        features_sequential[valid_indices] = global_features_flat
        features_sequential = features_sequential.view(batch_size, padded_seq_len, self.d_model)

        # --- 2. 动态的、逐样本的交叉注意力 ---
        final_features_list = []
        # 我们需要遍历批次中的每一个样本，因为它们的长度不同
        for i in range(batch_size):
            # 获取当前样本的真实序列长度
            seq_len = lengths[i].item()
            # 获取当前样本的真实特征序列 (去除填充部分)
            patient_features = features_sequential[i, :seq_len, :]

            # --- 根据序列长度，执行不同的策略 ---
            if seq_len == 0:
                # 没有任何影像，输出一个零向量
                final_patient_feature = torch.zeros(1, self.d_model, device=patient_features.device)

            elif seq_len == 1:
                # 只有一个时间点 (基线)，直接使用该特征
                final_patient_feature = patient_features[0:1, :] # 保持序列维度
            
            elif seq_len == 2:
                # 有两个时间点，执行一次交叉注意力
                F_baseline = patient_features[0:1, :].unsqueeze(0) # 形状变为 [1, 1, d_model]
                F2 = patient_features[1:2, :].unsqueeze(0)
                
                memory2 = -F2 if self.attention_mode == 'dissimilar' else F2
                
                F_updated = self.cross_attn_layers[0](tgt=F_baseline, memory=memory2)
                final_patient_feature = F_updated.squeeze(0) # 移除 batch 维度
                
            else: # seq_len >= 3
                # 有三个或更多时间点，执行完整的两次迭代
                F_baseline = patient_features[0:1, :].unsqueeze(0)
                F2 = patient_features[1:2, :].unsqueeze(0)
                F3 = patient_features[2:3, :].unsqueeze(0) # 我们只用前三个

                memory2 = -F2 if self.attention_mode == 'dissimilar' else F2
                memory3 = -F3 if self.attention_mode == 'dissimilar' else F3
                
                F_updated = self.cross_attn_layers[0](tgt=F_baseline, memory=memory2)
                F_final = self.cross_attn_layers[1](tgt=F_updated, memory=memory3)
                final_patient_feature = F_final.squeeze(0)

            final_features_list.append(final_patient_feature)

        # --- 3. 收集结果并预测 ---
        # 将处理完的每个样本的最终特征拼接回一个批次
        final_batch_feature = torch.cat(final_features_list, dim=0).squeeze(1)
        
        survival_logits = self.cox_task_head(final_batch_feature)
        multiclass_logits = self.multiclass_task_head(final_batch_feature)
        
        return {
            "survival_logits": survival_logits,
            "multiclass_logits": multiclass_logits,
            "log_var_cox": self.log_var_cox,
            "log_var_bce": self.log_var_bce
        }
    
    def unfreeze_encoder_layers(self, num_layers_to_unfreeze: int, is_init: bool = False):
        # ... 代码完全不变 ...
        prefix = "初始化解冻" if is_init else "动态解冻"
        print("\n" + "="*20)
        try:
            encoder_stages = self.vision_encoder.vision_tower.model.features
        except AttributeError:
            print("警告: 无法找到 vision_encoder.vision_tower.model.features 结构，跳过解冻。")
            print("="*20 + "\n"); return
            
        num_total_stages = len(encoder_stages)
        if num_layers_to_unfreeze == -1:
            layers_to_unfreeze = num_total_stages
            print(f"{prefix}：解冻所有 {layers_to_unfreeze} 个 Vision Encoder 的 Transformer stages...")
        elif num_layers_to_unfreeze > 0:
            layers_to_unfreeze = min(num_layers_to_unfreeze, num_total_stages)
            print(f"{prefix}：解冻最后 {layers_to_unfreeze} 个 Vision Encoder 的 Transformer stages...")
        else:
            print(f"{prefix}：指令为解冻0层，不执行任何操作。")
            print("="*20 + "\n"); return
        for stage in encoder_stages[-layers_to_unfreeze:]:
            for param in stage.parameters():
                param.requires_grad = True
        print(f"{prefix}操作完成！"); print("="*20 + "\n")

# =========================================================================
# ======================== 验证脚本 (演示如何使用新参数) ========================
# =========================================================================
if __name__ == '__main__':


    config_json_path = r"/data/yuanjiahong/yhh/llm_open_dig_4-15/config.json"
    weights_file_path = r"/data/yuanjiahong/yhh/code/utils/vision_tower_only_weights_0810.pt"
    with open(config_json_path, 'r') as f: config_dict = json.load(f)
    config = LamedConfig(**config_dict)
    def print_trainable_params(model):
        num_total = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"检查模型参数: 可训练/总参数 = {num_trainable / 1e6:.2f} M / {num_total / 1e6:.2f} M")
        print("-" * 50)

    COX_OUTPUT_DIM = 1
    NUM_MULTICLASS = 5
    print("\n\n=== 测试场景 1: 标准交叉注意力 ('similar' mode) ===")
    model_similar = SurvivalModel(
        vision_tower_config=config,
        weights_path=weights_file_path,
        num_time_bins=1,
        num_multiclass=5,
        attention_mode='similar' # <<< 指定模式
    )
    
    batch_size = 4
    max_seq_len = 3
    dummy_images = torch.randn(batch_size, max_seq_len, 3, 48, 256, 256)
    dummy_lengths = torch.tensor([3] * batch_size, dtype=torch.int64) 
    
    outputs_similar = model_similar(dummy_images, dummy_lengths)
    print("前向传播成功！")
    print(f"输出 survival_logits 形状: {outputs_similar['survival_logits'].shape}")

    # --- 场景2: 测试“查差异”模式 ---
    print("\n\n=== 测试场景 2: 差异性交叉注意力 ('dissimilar' mode) ===")
    model_dissimilar = SurvivalModel(
        vision_tower_config=config,
        weights_path=weights_file_path,
        num_time_bins=1,
        num_multiclass=5,
        attention_mode='dissimilar' # <<< 指定模式
    )
    
    outputs_dissimilar = model_dissimilar(dummy_images, dummy_lengths)
    print("前向传播成功！")
    print(f"输出 survival_logits 形状: {outputs_dissimilar['survival_logits'].shape}")
    print("-" * 50)
    print("模型结构升级成功！现在可以通过 attention_mode 参数灵活切换策略。")