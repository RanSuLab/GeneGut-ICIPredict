# -*- coding: utf-8 -*-
import torch
import os
import argparse
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from Models import VAE_prior_knowledge
from Tools import (VAE_prior_loss_function, RPKM_DatasetBuilder, load_protein_prior_knowledge,
                   plot_training_metrics, save_results_to_csv, FilteredDataset, sample_distribution)
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
import pandas as pd

def set_seed(seed=42):
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def train_model(model, dataloader, optimizer, device, reference_features):
    model.train()
    optimizer.zero_grad()
    train_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    for _, data, labels in dataloader:
        data, labels = data.float().to(device), labels.float().to(device)
        recon_x, mu, cls_output = model(data, reference_features)
        pre_labels = (cls_output > 0.5).float()

        loss = VAE_prior_loss_function(recon_x, data, mu, cls_output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
  
        all_labels.extend(labels.cpu().numpy())    
        all_preds.extend(pre_labels.detach().cpu().numpy())  
        all_probs.extend(cls_output.detach().cpu().numpy()) 

    final_loss = train_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds) 
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float('nan')  # AUC
    aupr = average_precision_score(all_labels, all_probs) 
    return final_loss, acc, auc, aupr

def validate_model(epoch,model, dataloader, device, reference_features):
    model.eval()
    val_loss = 0.0
    sample_name_list = []
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for sample_name, data, labels in dataloader:
            sample_name_list.extend(sample_name)
            data, labels = data.float().to(device), labels.float().to(device)
            recon_x, mu, cls_output = model(data, reference_features)
            pre_labels = (cls_output > 0.5).float()
           
            loss = VAE_prior_loss_function(recon_x, data, mu, cls_output, labels)
            val_loss += loss.item()
        
            all_labels.extend(labels.cpu().numpy())  
            all_preds.extend(pre_labels.cpu().numpy())  
            all_probs.extend(cls_output.cpu().numpy())  
    if (epoch + 1) % 10 == 0:  
        print(f'epoch-{epoch + 1},true label: {all_labels}')
        print(f'epoch-{epoch + 1},predict label: {all_preds}')

    final_loss = val_loss / len(dataloader)

    return final_loss, sample_name_list, all_labels,all_preds,all_probs

def get_metrics(all_labels,all_preds,all_probs):
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float('nan')
    aupr = average_precision_score(all_labels, all_probs)

    return acc, auc, aupr, precision, recall, f1
def main(args):
    os.makedirs(args.model_save_path, exist_ok=True)
    result_folder = str(args.model_save_path) + '/in_cohort/' + 'ROC_5fold_file'  # PD-1orCTLA4  CIBI
    os.makedirs(result_folder, exist_ok=True)
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    dataset = RPKM_DatasetBuilder(args.label_file, args.RPKM_features, args.task)

    save_path = os.path.dirname(args.RPKM_features)
    if args.task == 'BOR':
        print("Current task: predicting response to immunotherapy.")
        mi_result_path = os.path.join(save_path, "mutual_info_results.npz")
    if args.task == 'not_progression_free':
        print("Current task: predicting non-progression-free status.")
        mi_result_path = os.path.join(save_path, "mutual_info_results_NPF.npz")
    if os.path.exists(mi_result_path) and args.is_use_mutual_info == '1':
        print(f"Using features selected by the mutual information method; the top {args.mutual_info_K} features were selected.")
        mi_data = np.load(mi_result_path)
        mi_scores = mi_data['mi_scores']
        filtered_dataset = FilteredDataset(dataset, mi_scores, K=args.mutual_info_K)
        labels = [label for _, _, label in filtered_dataset.samples]
        sample_data = filtered_dataset.samples[0][1]
        input_dim = sample_data.shape[0]
    if not os.path.exists(mi_result_path) and args.is_use_mutual_info == '1':
        print("Mutual information has not been computed. Please run mutual_info.py before executing the training code.")
        return

    if args.is_use_mutual_info == '0':
        filtered_dataset = dataset
        labels = [label for _, _, label in filtered_dataset.samples]
        sample_data = filtered_dataset.samples[0][1]
        input_dim = sample_data.shape[0]
        print(f"Mutual information-based feature selection is not used; the current feature dimension is {input_dim}.")

 
    reference_features = load_protein_prior_knowledge(args.prior_file_path).to(device)
    reference_feature_dim = reference_features.shape[1]

    model = VAE_prior_knowledge(input_dim, 512, 256, reference_feature_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    stratified_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(stratified_kf.split(filtered_dataset.samples, labels), 1):
        print('-' * 25, f" Fold {fold} ", '-' * 25)
        train_samples = [filtered_dataset.samples[i] for i in train_idx]
        val_samples = [filtered_dataset.samples[i] for i in val_idx]

        train_dataloader = DataLoader(train_samples, batch_size=8, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(val_samples, batch_size=8, shuffle=True)

        train_accuracy_list = []
        train_loss_list = []

        for epoch in range(args.epochs):
            train_loss, train_acc, train_auc, train_aupr = train_model(model,
                                                                       train_dataloader,
                                                                       optimizer,
                                                                       device,
                                                                       reference_features)
            if (epoch + 1) % 10 == 0: 
                print(f"Epoch [{epoch + 1}/{args.epochs}], "
                      f"Accuracy: {train_acc:.4f}, "
                      f"Loss: {train_loss:.4f}, "
                      f"AUC: {train_auc:.4f}, "
                      f"AUPR: {train_aupr:.4f}")
            train_accuracy_list.append(train_acc)
            train_loss_list.append(train_loss)

        val_loss, sample_name_list, all_labels, all_preds, all_probs = validate_model(epoch,model,val_dataloader,device,reference_features)
        val_acc, val_auc, val_aupr, val_precision, val_recall, val_f1 = get_metrics(all_labels,all_preds,all_probs)

        plot_training_metrics(train_accuracy_list, train_loss_list, fold, args.model_save_path)

        save_csv_path = os.path.join(result_folder, f'fold{fold}_results.csv')
        df = pd.DataFrame({
            'sample': sample_name_list,
            'label': all_labels,
            'pred': all_preds,
            'prob': all_probs
        })
        df.to_csv(save_csv_path, index=False)
 
        print("=" * 50)
        print(f"{'Metric (Validation)':<20} | {'Value'}")
        print("-" * 50)
        print(f"{'Loss':<20} | {val_loss:.4f}")
        print(f"{'Accuracy':<20} | {val_acc:.4f}")
        print(f"{'AUC':<20} | {val_auc:.4f}")
        print(f"{'AUPR':<20} | {val_aupr:.4f}")
        print(f"{'Precision':<20} | {val_precision:.4f}")
        print(f"{'Recall':<20} | {val_recall:.4f}")
        print(f"{'F1-score':<20} | {val_f1:.4f}")

        file_save = os.path.join(args.model_save_path, 'result.csv')
        save_results_to_csv(file_save, fold, val_acc, val_precision, val_recall, val_f1, val_auc, val_aupr)

        model_filename = os.path.join(args.model_save_path, f'model_fold{fold}.pth')
        torch.save(model.state_dict(), model_filename)

        print("=" * 50)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and evaluate VAE with prior knowledge and RPKM")
    parser.add_argument('--RPKM_features', type=str,required=True, help="Dataset of RPKM features")
    parser.add_argument('--label_file', type=str, required=True,help="CSV file with labels")
    parser.add_argument('--task',type=str,default='BOR', help="BOR;not_progression_free")
    parser.add_argument('--prior_file_path',type=str,default="../dataset/CA209-538_embedding.pth",help="Prior knowledge file path")
    parser.add_argument('--model_save_path',type=str,required=True,help="Directory to save trainned models")
    parser.add_argument('--cuda_device', type=int, default=3, help='CUDA device index (default: 3)')
    parser.add_argument('--epochs', type=int, default=500, help='epochs (default: 500)')
    parser.add_argument('--is_use_mutual_info', type=str, required=True, help='Whether to use mutual information to select features')
    parser.add_argument('--mutual_info_K', type=int, default=80000, help='mutual information K (default: 80000)')


    args = parser.parse_args()
    main(args)

