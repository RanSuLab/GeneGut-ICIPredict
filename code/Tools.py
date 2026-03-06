import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch import nn
import random
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch.nn.functional as F


def set_random_seed(seed):
    torch.manual_seed(seed)   
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    random.seed(seed) 
    np.random.seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RPKM_DatasetBuilder(Dataset):
    def __init__(self, label_file, feature_file,task = 'BOR'):
        self.label_df = pd.read_csv(label_file)  
        self.feature_df = pd.read_csv(feature_file)
        if task == 'not_progression_free':
            label_column = 2
            print('Task: not_progression_free')
        if task == 'BOR':
            label_column = 3
            print('Task: BOR')

        features = self.feature_df.iloc[:, 1:].values
        scaler = StandardScaler()  
        scaled_features = scaler.fit_transform(features.T).T
        scaled_features = scaled_features.astype(np.float32)
        self.feature_df.iloc[:, 1:] = scaled_features

        self.samples = []
        for col in self.feature_df.columns[1:]: 
            sample_name = col  
            sample_features = self.feature_df[col].values 

            label_row = self.label_df[self.label_df.iloc[:, 0] == sample_name]
            if not label_row.empty:
                label_value = label_row.iloc[0, label_column]
                label = bool(label_value) 
                self.samples.append((sample_name, sample_features, label))
            else:
                print(f"Warning: Label not found for sample '{sample_name}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name, features, label = self.samples[idx]
        return sample_name, features, label

def merge_RPKM_Clinical_datasets2(rpkm_dataset, clinical_dataset):
    merged_samples = []

    clinical_dict = dict(clinical_dataset.get_samples())

    for sample_name, rpkm_features, label in rpkm_dataset.samples:
        if sample_name not in clinical_dict:
            raise ValueError(
                f"Sample {sample_name} missing in clinical dataset"
            )

        clinical_features = clinical_dict[sample_name]

        merged_features = np.concatenate(
            [rpkm_features, clinical_features], axis=0
        )

        merged_samples.append(
            (sample_name, merged_features, label)
        )

    rpkm_dataset.samples = merged_samples
    return rpkm_dataset


def sample_distribution(dataset):
    positive_samples = 0
    negative_samples = 0

    for sample_folder, data, label in dataset:
        if label:
            positive_samples += 1
        else:
            negative_samples += 1

    return f"Positive samples: {positive_samples},Negative samples: {negative_samples}"


def save_results_to_csv(file_name, fold, accuracy, precision, recall, f1, auc, AUP):
    file_exists = False
    try:
        with open(file_name, 'r') as f:
            file_exists = True
    except FileNotFoundError:
        pass

    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(['Fold', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'AUPR'])

        writer.writerow([fold, float(accuracy), float(precision), float(recall), float(f1), float(auc), float(AUP)])


class FilteredDataset(Dataset):
    def __init__(self, original_dataset, mi_scores, K):
        self.samples = [] 
        top_k_indices = np.argsort(mi_scores)[-K:][::-1]

        for sample_name, features, label in original_dataset.samples:
            reduced_features = features[top_k_indices]  
            self.samples.append((sample_name, reduced_features, label))

        print(f"Filtered dataset created with top {K} features.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def plot_training_metrics(accuracy_list, loss_list, fold, save_path):
    epochs = range(1, len(accuracy_list) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy_list, label='Accuracy', color='#1230AE', marker='o')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss_list, label='Loss', color='#EB5B00', marker='o')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    save_folder = save_path + '/training_figure'
    os.makedirs(save_folder, exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_folder +'/fold_' + str(fold) + '.png')  
    plt.close()  


def VAE_prior_loss_function(recon_x, x, mu, cls_output, target):
    MSE_loss = F.mse_loss(recon_x, x, reduction='sum')
    lambda_MSE = 0.1
    MSE_labmbda_loss = lambda_MSE * MSE_loss

    KDL_loss = 0.5 * (mu.pow(2)).sum(dim=1).mean() 
    lambda_KDL = 0.2
    KDL_lambda_loss = KDL_loss * lambda_KDL

    classification_loss = F.binary_cross_entropy(cls_output.squeeze(), target.float().squeeze(), reduction='sum')
    return MSE_labmbda_loss + KDL_lambda_loss + classification_loss


def load_protein_prior_knowledge(prior_file_path):
    prior_features_list = torch.load(prior_file_path, weights_only=True)
    prior_features_list = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in prior_features_list]
    prior_features = torch.stack(prior_features_list)
    return prior_features


