import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

try:
    from SurvivalAnalysis_multi import cox_loss, FocalLoss, c_index
except ImportError:
    print("Error: Can`t import functions from 'SurvivalAnalysis_multi.py'")
    def cox_loss(*args): raise NotImplementedError("cox_loss not implemented")
    class FocalLoss(nn.Module):
        def forward(self, *args): raise NotImplementedError("FocalLoss not implemented")
    def c_index(*args): raise NotImplementedError("c_index not implemented")

def pearson_correlation_loss(risk_scores, neg_probabilities):
    x, y = risk_scores.view(-1), neg_probabilities.view(-1)
    if len(x) < 2:
        return torch.tensor(0.0, device=x.device, requires_grad=True)
    vx, vy = x - torch.mean(x), y - torch.mean(y)
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return -cost

class Trainer:
    def __init__(self, model, dataloaders,
                 num_epochs: int,
                 time_points: np.ndarray,
                 focal_loss_gamma: float = 2.0,
                 learning_rate: float = 1e-4,
                 consistency_weight: float = 0.0,
                 weight_decay: float = 1e-5,
                 teacher_index: int = -1,
                 num_early_bins: int = 0,
                 device: str ='cuda'):
        
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.device = device
        self.num_epochs = num_epochs
        self.initial_lr = learning_rate
        self.weight_decay = weight_decay
        self.consistency_weight = consistency_weight
        self.teacher_index = teacher_index
        self.num_early_bins = num_early_bins
        self.time_points = time_points
        
        self.cox_criterion = cox_loss
        self.focal_criterion = FocalLoss(alpha=0.25, gamma=focal_loss_gamma).to(device)
        if self.consistency_weight > 0:
            self.consistency_criterion = pearson_correlation_loss
        
        self.optimizer = None
        self.scheduler = None

        if self.consistency_weight > 0:
            print(f"  - L_Pearson 损失权重 {self.consistency_weight} (effect on Stage 2 and Stage 3)。")
        else:
            print("  - L_Pearson: Not Work")

    def _create_optimizer_and_scheduler(self, lr, num_epochs_for_scheduler):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            print("警告: 模型中没有可训练的参数。"); self.optimizer, self.scheduler = None, None; return
        self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, num_epochs_for_scheduler))
        print(f"Optimizer AdamW created, learning rate: {lr}")

    def _train_epoch(self, current_stage: int):
        self.model.train()
        loss_tracker = {
            'total': 0.0, 'cox_raw': 0.0, 'focal_raw': 0.0, 
            'consistency': 0.0, 'auto_weighted': 0.0,
            'weight_cox': 0.0, 'weight_focal': 0.0 
        }
        all_risk_scores, all_times, all_events = [], [], []
        
        progress_bar = tqdm(self.dataloaders['train'], desc=f"Training (Stage {current_stage})", leave=False, dynamic_ncols=True)
        
        for batch in progress_bar:
            images, lengths = batch['images'].to(self.device), batch['lengths']
            time, event = batch['time'].to(self.device), batch['event'].to(self.device)
            labels, masks = batch['multiclass_labels'].to(self.device), batch['multiclass_masks'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images, lengths)
            survival_logits, multi_logits = outputs['survival_logits'], outputs['multiclass_logits']
            log_var_cox, log_var_bce = outputs['log_var_cox'], outputs['log_var_bce']
            loss_cox_raw = self.cox_criterion(survival_logits, time, event)
            
            if current_stage == 1 and self.num_early_bins > 0:
                loss_focal_raw = self.focal_criterion(multi_logits[:, self.num_early_bins:], labels[:, self.num_early_bins:], masks[:, self.num_early_bins:])
            else:
                loss_focal_raw = self.focal_criterion(multi_logits, labels, masks)

            precision_cox = torch.exp(-log_var_cox); precision_bce = torch.exp(-log_var_bce)
            loss_auto_weighted = (precision_cox * loss_cox_raw + precision_bce * loss_focal_raw + 0.5 * log_var_cox + 0.5 * log_var_bce).mean()
            total_loss = loss_auto_weighted
            loss_consistency = torch.tensor(0.0)
            
            if self.consistency_weight > 0 and current_stage in [2, 3]:
                teacher_slice = slice(self.teacher_index, self.teacher_index + 1)
                prob_teacher = torch.sigmoid(multi_logits[:, teacher_slice])
                loss_consistency = self.consistency_criterion(survival_logits, -prob_teacher)
                total_loss += self.consistency_weight * loss_consistency

            total_loss.backward(); self.optimizer.step()
            
            loss_tracker['total'] += total_loss.item(); loss_tracker['auto_weighted'] += loss_auto_weighted.item()
            loss_tracker['cox_raw'] += loss_cox_raw.mean().item(); loss_tracker['focal_raw'] += loss_focal_raw.mean().item()
            loss_tracker['consistency'] += loss_consistency.item(); loss_tracker['weight_cox'] += precision_cox.mean().item()
            loss_tracker['weight_focal'] += precision_bce.mean().item()

            all_risk_scores.append(survival_logits.detach().cpu()); all_times.append(time.cpu()); all_events.append(event.cpu())
            progress_bar.set_postfix(loss=f'{total_loss.item():.3f}', w_cox=f"{precision_cox.mean().item():.2f}", w_focal=f"{precision_bce.mean().item():.2f}")

        num_batches = len(self.dataloaders['train'])
        avg_losses = {k: v / num_batches for k, v in loss_tracker.items()}
        avg_losses['train_c_index'] = c_index(torch.cat(all_risk_scores), torch.cat(all_times), torch.cat(all_events))
        return avg_losses

    def _evaluate_epoch(self, mode: str):
        
        self.model.eval()
        all_risks, all_times, all_events = [], [], []
        all_multi_logits, all_multi_labels, all_multi_masks = [], [], []

        dataloader = self.dataloaders[mode]
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating ({mode.upper()})", leave=False, dynamic_ncols=True):
                images, lengths = batch['images'].to(self.device), batch['lengths']
                time, event = batch['time'], batch['event']
                labels, masks = batch['multiclass_labels'], batch['multiclass_masks']
                
                outputs = self.model(images, lengths)
                
                all_risks.append(outputs['survival_logits'].cpu())
                all_times.append(time.cpu())
                all_events.append(event.cpu())
                all_multi_logits.append(outputs['multiclass_logits'].cpu())
                all_multi_labels.append(labels.cpu())
                all_multi_masks.append(masks.cpu())

        risks, times, events = torch.cat(all_risks), torch.cat(all_times), torch.cat(all_events)
        logits, labels, masks = torch.cat(all_multi_logits), torch.cat(all_multi_labels), torch.cat(all_multi_masks)

        c_idx = c_index(risks, times, events)

        probs = torch.sigmoid(logits)
        auc_scores = []
        for i in range(self.time_points.shape[0]):
            class_mask = masks[:, i].bool()
            true_labels, pred_probs = labels[:, i][class_mask], probs[:, i][class_mask]
            
            if len(torch.unique(true_labels)) > 1:
                try:
                    auc = roc_auc_score(true_labels.numpy(), pred_probs.numpy())
                    auc_scores.append(auc)
                except ValueError: pass
        
        avg_auc = np.mean(auc_scores) if auc_scores else 0.0

        predictions_df = None
        try:
            df_data = {'PFS': times.numpy(), 'event': events.numpy(), 'risk_score': risks.numpy().flatten()}
            for i, t in enumerate(self.time_points):
                df_data[f'prob_survival_{t}m'] = probs[:, i].numpy()
                df_data[f'label_survival_{t}m'] = labels[:, i].numpy()
                df_data[f'mask_{t}m'] = masks[:, i].numpy()
            predictions_df = pd.DataFrame(df_data)
        except Exception as e:
            print(f"Warning: Create DataFrame failed: {e}")
        
        return c_idx, avg_auc, predictions_df
        
    def fit(self, 
            stage1_epochs: int,
            stage2_lr_ratio: float,
            unfreeze_epoch: int,
            finetune_lr: float, 
            num_layers_to_finetune: int,
            early_stopping_patience: int = 15, 
            output_dir: str = './output'):
        
        os.makedirs(output_dir, exist_ok=True)
        paths = {
            'best': os.path.join(output_dir, 'best_model.pt'),
            'last': os.path.join(output_dir, 'last_epoch_model.pt'),
            'log': os.path.join(output_dir, 'training_log.csv'),
            'val_preds': os.path.join(output_dir, 'best_model_val_predictions.csv'),
            'test_preds': os.path.join(output_dir, 'best_model_test_predictions.csv') # 新增
        }
        
        history, best_val_avg_auc, epochs_no_improve = [], -1.0, 0
        
        unfreeze_epoch = max(unfreeze_epoch, stage1_epochs + 1)
        
        stage_definitions = [
            {'name': 'Stage 1: Mid-Term Warm-up', 'epochs': stage1_epochs, 'stage_code': 1, 'lr': self.initial_lr, 'unfreeze': False},
            {'name': 'Stage 2: Joint Training & Alignment', 'epochs': unfreeze_epoch - stage1_epochs, 'stage_code': 2, 'lr': self.initial_lr * stage2_lr_ratio, 'unfreeze': False},
            {'name': 'Stage 3: Constrained Fine-Tuning', 'epochs': self.num_epochs - unfreeze_epoch, 'stage_code': 3, 'lr': finetune_lr, 'unfreeze': True}
        ]
        
        current_epoch_counter = 0; stop_outer_loop = False
        
        for stage in stage_definitions:
            if stage['epochs'] <= 0: continue
            print(f"\n{'='*20} {stage['name']}: {stage['epochs']} epochs {'='*20}")
            if stage['unfreeze']:
                model_to_unfreeze = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                model_to_unfreeze.unfreeze_encoder_layers(num_layers_to_finetune)
            self._create_optimizer_and_scheduler(stage['lr'], stage['epochs'])

            for i in range(stage['epochs']):
                current_epoch_counter += 1
                if self.num_epochs > 0:
                    print(f"\n--- Epoch {current_epoch_counter}/{self.num_epochs} ---")

                train_metrics = self._train_epoch(current_stage=stage['stage_code'])
                
                # 使用新的评估函数
                val_c_index, val_avg_auc, val_preds_df = self._evaluate_epoch(mode='val')
                
                test_c_index, test_avg_auc, _ = self._evaluate_epoch(mode='test')
                
                if self.scheduler: self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0
                w_cox, w_focal = train_metrics.get('weight_cox', -1), train_metrics.get('weight_focal', -1)
    
                print(f"Epoch {current_epoch_counter} Summary: Train Loss: {train_metrics['total']:.4f}, Train C-Index: {train_metrics['train_c_index']:.4f}")
                print(f"  - Validation: C-Index: {val_c_index:.4f}, Avg AUC: {val_avg_auc:.4f}")
                print(f"  - Test Set:   C-Index: {test_c_index:.4f}, Avg AUC: {test_avg_auc:.4f} (Monitoring only)")
                print(f"  - LR: {current_lr:.8f}, Auto-Weights -> Cox: {w_cox:.4f}, Focal: {w_focal:.4f}")
                
                log_entry = train_metrics.copy()
                log_entry.update({
                    'epoch': current_epoch_counter, 'learning_rate': current_lr, 'stage': stage['stage_code'],
                    'val_c_index': val_c_index, 'val_avg_auc': val_avg_auc,
                    'test_c_index': test_c_index, 'test_avg_auc': test_avg_auc
                })
                history.append(log_entry)
                pd.DataFrame(history).to_csv(paths['log'], index=False)
                
                if val_avg_auc > best_val_avg_auc:
                    print(f"  -- Val AUC Improved ({best_val_avg_auc:.4f} --> {val_avg_auc:.4f}). Saving best model and val predictions... --")
                    best_val_avg_auc = val_avg_auc
                    model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
                    torch.save(model_state, paths['best'])
                    if val_preds_df is not None: val_preds_df.to_csv(paths['val_preds'], index=False)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f"  -- Val AUC did not improve. Patience: {epochs_no_improve}/{early_stopping_patience} --")

                if epochs_no_improve >= early_stopping_patience:
                    print(f"\nEarly stopping triggered at Epoch {current_epoch_counter}.")
                    stop_outer_loop = True; break
            
            if stop_outer_loop: break

        print(f"\n--- Training Finished ---")
        print(f"Best validation AUC achieved: {best_val_avg_auc:.4f}")
        last_model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        torch.save(last_model_state, paths['last'])
        
        print("\n--- Performing Final Evaluation on Test Set with Best Model ---")
        if os.path.exists(paths['best']):
            best_state_dict = torch.load(paths['best'], map_location=self.device)
            self.model.load_state_dict(best_state_dict)
            final_test_c, final_test_auc, final_test_preds_df = self._evaluate_epoch(mode='test')
            
            print("\n" + "="*50)
            print(" " * 12 + "Final Reported Performance")
            print("="*50)
            print(f"  - Final Test C-Index: {final_test_c:.4f}")
            print(f"  - Final Test Avg AUC: {final_test_auc:.4f}")
            print("="*50)

            if final_test_preds_df is not None:
                final_test_preds_df.to_csv(paths['test_preds'], index=False)
                print(f"Final test predictions saved to {paths['test_preds']}")
        else:
            print("Warning: Best model file not found. Skipping final evaluation.")


        return self.model, pd.DataFrame(history)
