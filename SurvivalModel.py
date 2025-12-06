print("===== LOADING SurvivalModel (Version: Switchable Iterative Cross-Attention) =====")

import torch
import torch.nn as nn
import json
import torch.nn.functional as F


try:
    from encoder.configuration import LamedConfig
    from encoder.vison_tower_component import Swin3DTower
except ImportError:
    class LamedConfig: hidden_size = 1024
    class Swin3DTower(nn.Module):
        def __init__(self, config): super().__init__(); self.hidden_size = config.hidden_size
        def forward(self, x): return torch.randn(x.shape[0], 2, self.hidden_size)


class AttentionPooler(nn.Module):
    """
    Input:  (B, N, D)
    Output:  (B, D)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a_scores = self.attention_net(x)
        a_weights = torch.softmax(a_scores, dim=1)
        weighted_sum = torch.sum(a_weights * x, dim=1)
        
        return weighted_sum
    

class SurvivalModel(nn.Module):
    def __init__(self, vision_tower_config, weights_path,
                 num_time_bins: int, num_multiclass: int = 5,
                 transformer_nhead=8,
                 transformer_dim_feedforward=2048, dropout_rate=0.3, 
                 task_specific_hidden_dim=256,
                 attention_mode: str = 'similar'):
        super().__init__()
        

        self.vision_encoder = Swin3DTower(vision_tower_config)
        
        state_dict = torch.load(weights_path, map_location='cpu')
        self.vision_encoder.load_state_dict(state_dict)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        self.d_model = self.vision_encoder.hidden_size
        self.num_multiclass = num_multiclass
        
        if attention_mode not in ['similar', 'dissimilar']:
            raise ValueError("attention_mode must be 'similar' 或 'dissimilar'")
        self.attention_mode = attention_mode
        print(f"  - [Attention Mode] : '{self.attention_mode}'")

        self.feature_pooler = AttentionPooler(d_model=self.d_model)
                     
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.d_model, nhead=transformer_nhead,
                dim_feedforward=transformer_dim_feedforward, dropout=dropout_rate,
                batch_first=True
            ) for _ in range(2) 
        ])

        self.cox_task_head = nn.Sequential(
            nn.Linear(self.d_model, task_specific_hidden_dim),
            nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(task_specific_hidden_dim, num_time_bins)
        )

        self.multiclass_task_head = nn.Sequential(
            nn.Linear(self.d_model, task_specific_hidden_dim),
            nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(task_specific_hidden_dim, self.num_multiclass) 
        )

        self.log_var_cox = nn.Parameter(torch.zeros(1))
        self.log_var_bce = nn.Parameter(torch.zeros(1))
        

    def forward(self, images_padded, lengths):

        batch_size, padded_seq_len, C, D, H, W = images_padded.shape
        images_flattened = images_padded.view(batch_size * padded_seq_len, C, D, H, W)

        mask = torch.arange(padded_seq_len, device=lengths.device)[None, :] < lengths[:, None]
        valid_indices = mask.view(-1) 
        

        valid_images = images_flattened[valid_indices]
        if valid_images.shape[0] == 0: 
            dummy_output = torch.zeros(batch_size, self.d_model, device=images_padded.device)
            return {
            "survival_logits": survival_logits,
            "multiclass_logits": multiclass_logits,
            "log_var_cox": self.log_var_cox,
            "log_var_bce": self.log_var_bce
        }

        features_output = self.vision_encoder(valid_images)
        global_features_flat = self.feature_pooler(features_output)
        
        features_sequential = torch.zeros(batch_size * padded_seq_len, self.d_model, device=images_padded.device)
        features_sequential[valid_indices] = global_features_flat
        features_sequential = features_sequential.view(batch_size, padded_seq_len, self.d_model)

        final_features_list = []
        for i in range(batch_size):
            seq_len = lengths[i].item()
            patient_features = features_sequential[i, :seq_len, :]


            if seq_len == 0:
                final_patient_feature = torch.zeros(1, self.d_model, device=patient_features.device)

            elif seq_len == 1:
                final_patient_feature = patient_features[0:1, :] 
                
            elif seq_len == 2:
                F_baseline = patient_features[0:1, :].unsqueeze(0) 
                F2 = patient_features[1:2, :].unsqueeze(0)
                memory2 = -F2 if self.attention_mode == 'dissimilar' else F2
                F_updated = self.cross_attn_layers[0](tgt=F_baseline, memory=memory2)
                final_patient_feature = F_updated.squeeze(0) 
                
            else: 
                F_baseline = patient_features[0:1, :].unsqueeze(0)
                F2 = patient_features[1:2, :].unsqueeze(0)
                F3 = patient_features[2:3, :].unsqueeze(0) 

                memory2 = -F2 if self.attention_mode == 'dissimilar' else F2
                memory3 = -F3 if self.attention_mode == 'dissimilar' else F3
                
                F_updated = self.cross_attn_layers[0](tgt=F_baseline, memory=memory2)
                F_final = self.cross_attn_layers[1](tgt=F_updated, memory=memory3)
                final_patient_feature = F_final.squeeze(0)

            final_features_list.append(final_patient_feature)


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
        
        prefix = "Initial unfreeze" if is_init else "Dynamical unfreeze"
        print("\n" + "="*20)
        try:
            encoder_stages = self.vision_encoder.vision_tower.model.features
        except AttributeError:
            print("Warrning: Can`t find vision_encoder.vision_tower.model.features，will skip ")
            print("="*20 + "\n"); return
            
        num_total_stages = len(encoder_stages)
        if num_layers_to_unfreeze == -1:
            layers_to_unfreeze = num_total_stages
            print(f"{prefix}：unfreeze {layers_to_unfreeze} Transformer stages of the vision encoder...")
        elif num_layers_to_unfreeze > 0:
            layers_to_unfreeze = min(num_layers_to_unfreeze, num_total_stages)
            print(f"{prefix}：unfreeze last {layers_to_unfreeze} ransformer stages of the vision encoder...")
        else:
            print("="*20 + "\n"); return
        for stage in encoder_stages[-layers_to_unfreeze:]:
            for param in stage.parameters():
                param.requires_grad = True

