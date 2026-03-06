# -*- coding: UTF-8 -*-

import torch
import os
import argparse
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from Models import VAE_prior_knowledge
from Tools import RPKM_DatasetBuilder, load_protein_prior_knowledge, FilteredDataset
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import os

def save_to_csv(save_file_path, sample_name_list, all_labels, all_preds, all_probs):

    assert len(sample_name_list) == len(all_labels) == len(all_preds) == len(all_probs), "All lists must be of equal length!"
    df = pd.DataFrame({
        "Sample Name": sample_name_list,
        "True Label": all_labels,
        "Predicted Label": all_preds,
        "Predicted Probability": all_probs
    })

    save_file = save_file_path+'.csv'
    df.to_csv(save_file, index=False, encoding='utf-8')


def plot_roc_curve(ROC_result, save_file_path):
    plt.figure(figsize=(6, 6)) 
    colors = ['#EC8305', '#024CAA', '#091057','#8B5DFF','#B03052']

    for idx, (labels, pred_probs) in enumerate(ROC_result):
   
        fpr, tpr, _ = roc_curve(labels, pred_probs)
        roc_auc = auc(fpr, tpr)  


        plt.plot(fpr, tpr, lw=1, label=f'Fold_{idx + 1} (AUC = {roc_auc:.2f})', color=colors[idx])

    plt.plot([0, 1], [0, 1], lw=1,color='grey', linestyle='--') 

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 − Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')

    save_path = save_file_path + '.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 
    plt.close()  

    print(f"ROC 曲线已保存至: {save_path}")

def test_model(model_path, dataset, reference_features, save_file_path,device):
    test_data = DataLoader(dataset, batch_size=8, shuffle=True)

    sample_data = dataset.samples[0][1]
    input_dim = sample_data.shape[0]

    model = VAE_prior_knowledge(input_dim, 512, 256, 320)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    sample_name_list = []
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for sample_name, data, labels in test_data:
            data, labels = data.float().to(device), labels.float().to(device)
            _, _, cls_output = model(data, reference_features)
            pre_labels = (cls_output > 0.5).float()
            sample_name_list.extend(sample_name)
            all_labels.extend(labels.cpu().numpy())  
            all_preds.extend(pre_labels.cpu().numpy()) 
            all_probs.extend(cls_output.cpu().numpy())  

    save_to_csv(save_file_path, sample_name_list, all_labels, all_preds, all_probs)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float('nan')  # AUC
    aupr = average_precision_score(all_labels, all_probs) 

    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:    {f1:.4f}")
    print(f"AUC:        {auc:.4f}")
    print(f"AUPR:       {aupr:.4f}")

    return all_labels, all_probs, acc, precision, recall, f1, auc, aupr

def save_metrics_to_csv(metric_list, save_path):
    df = pd.DataFrame(metric_list, columns=["Fold", "Accuracy", "Precision", "Recall", "F1 score", "AUC", "AUPR"])

    avg_values = ["Average"] + df.iloc[:, 1:].mean().tolist()
    df.loc[len(df)] = avg_values

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved metrics to {save_path}")


def main(args):
    print('-' * 50)
    dataset = RPKM_DatasetBuilder(args.label_file, args.RPKM_features, args.task)
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    if args.task == 'BOR':
        mi_result_path = os.path.join(args.mutual_info_dir, "mutual_info_results.npz")
    if args.task == 'not_progression_free':
        mi_result_path = os.path.join(args.mutual_info_dir, "mutual_info_results_NPF.npz")


    if args.is_use_mutual_info == '1':
        print(f'Use mutual information-based feature selection method: Yes')
        mi_data = np.load(mi_result_path)
        mi_scores = mi_data['mi_scores']
        filtered_dataset = FilteredDataset(dataset, mi_scores, K=80000)

    if args.is_use_mutual_info == '0':
        filtered_dataset = dataset
        print(f'Use mutual information-based feature selection method: No')

    # Load prior knowledge (this feature is not actually used during the testing phase, but is required for initialization)
    reference_features = load_protein_prior_knowledge(args.prior_file_path).to(device)

    test_name = os.path.basename(os.path.dirname(args.label_file))
    print('test data:{}'.format(test_name))
    result_foder =  str(args.mode_dirc) + '/test_results/'+ str(test_name)  #  PD-1 or CTLA4  CIBI
    os.makedirs(result_foder, exist_ok=True)
    print('-' * 50)

    ROC_result = [] 
    all_metrics = []

    for fold_index in range(1,6):
        print("=" * 20,'fold_',fold_index,"=" * 19)
        model_path = str(args.mode_dirc) + '/model_fold'+str(fold_index) +'.pth'
        save_file_path = result_foder + '/model_'+ str(fold_index) + '_test_'+ str(test_name)
        all_labels, all_probs, acc, precision, recall, f1, auc, aupr = test_model(model_path, filtered_dataset, reference_features, save_file_path,device)
        ROC_result.append((all_labels,all_probs))
        all_metrics.append([fold_index, acc, precision, recall, f1, auc, aupr])
        print("=" * 40)

    fig_save_path = result_foder + '/AUC'
    plot_roc_curve(ROC_result, fig_save_path)

    csv_save_path = os.path.join(result_foder, f"all_metrics.csv")
    save_metrics_to_csv(all_metrics, csv_save_path)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate VAE with prior knowledge and RPKM")
    parser.add_argument('--RPKM_features', type=str, required=True, help="Dataset of RPKM features")
    parser.add_argument('--label_file', type=str, required=True, help="CSV file with labels")
    parser.add_argument('--prior_file_path', type=str, required=True, help="Prior knowledge file path")
    parser.add_argument('--task', type=str, default='BOR', help="BOR;not_progression_free")
    parser.add_argument('--mode_dirc',type=str)
    parser.add_argument('--cuda_device', type=int, default=3, help='CUDA device index (default: 3)')
    parser.add_argument('--mutual_info_dir', type=str, required=True)
    parser.add_argument('--is_use_mutual_info', type=str, required=True, help='Whether to use mutual information to select features')

    args = parser.parse_args()
    main(args)