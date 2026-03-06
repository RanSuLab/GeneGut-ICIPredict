# -*- coding: UTF-8 -*-


import sys
import os
import pandas as pd
from Tools import RPKM_DatasetBuilder
from sklearn.feature_selection import mutual_info_classif, SelectKBest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def mutual_info_plots(task, cumulative_mi, mi_scores, save_dir, feature_names=None):

    if task == 'BOR':
        fig_dir = os.path.join(save_dir, "BOR-mutual_info_fig_contig200")
    if task == 'not_progression_free':
        fig_dir = os.path.join(save_dir, "NPF-mutual_info_fig_contig200")
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(12, 7))
    plt.plot(range(1, len(cumulative_mi) + 1), cumulative_mi, marker='.', linestyle='-', label="Cumulative Contribution")
    plt.axhline(y=0.95, color='r', linestyle='--', label="95% Contribution")
    plt.xlabel("Number of Selected Features (K)")
    plt.ylabel("Cumulative Mutual Information Contribution")
    plt.title("Cumulative Mutual Information Contribution Curve")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(fig_dir, "cumulative_mi_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

    top_k = 20
    top_features = np.argsort(mi_scores)[-top_k:][::-1]
    top_mi_scores = mi_scores[top_features]

    if feature_names is not None:
        top_feature_names = [feature_names[i] for i in top_features]
    else:
        top_feature_names = [f"Feature {i}" for i in top_features]

    plt.figure(figsize=(14, 10), dpi=300)
    plt.barh(range(top_k), top_mi_scores, color='b', align='center')
    plt.yticks(range(top_k), top_feature_names, fontsize=8)
    plt.xlabel("Mutual Information Score")
    plt.ylabel("Feature")
    plt.title("Top 20 Important Features")
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(fig_dir, "top_features_bar.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 7), dpi=300)
    plt.scatter(range(len(mi_scores)), mi_scores, alpha=0.5)
    plt.xlabel("Feature Index")
    plt.ylabel("Mutual Information Score")
    plt.title("Mutual Information Distribution Across Features")
    plt.savefig(os.path.join(fig_dir, "mi_scatter.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 7), dpi=300)
    plt.hist(mi_scores, bins=50, color='g', alpha=0.7)
    plt.xlabel("Mutual Information Score")
    plt.ylabel("Number of Features")
    plt.title("Distribution of Mutual Information Scores")
    plt.savefig(os.path.join(fig_dir, "mi_histogram.png"), dpi=300, bbox_inches='tight')
    plt.close()


def save_mi_ranking_to_csv(task, mi_scores, save_path, feature_names):

    assert len(feature_names) == len(mi_scores), "特征名数量与得分数量不一致"

    if task == 'BOR':
        fig_dir = os.path.join(save_path, "BOR-mutual_info_fig_contig200")
    if task == 'not_progression_free':
        fig_dir = os.path.join(save_path, "NPF-mutual_info_fig_contig200")
    os.makedirs(fig_dir, exist_ok=True)

    sorted_indices = np.argsort(mi_scores)[::-1]
    sorted_scores = mi_scores[sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]

    df = pd.DataFrame({
        "Feature": sorted_feature_names,
        "MI_Score": sorted_scores
    })

    csv_path = os.path.join(fig_dir, f"all_features_sorted.csv")

    df.to_csv(csv_path, index=False)


def check_mutual_info(label_file, RPKM_features, task):
    dataset = RPKM_DatasetBuilder(label_file, RPKM_features, task)
    print(dataset.samples[0])
    X = np.array([features for _, features, _ in dataset.samples]) 
    print(f'X.shape = {X.shape}')
    y = np.array([label for _, _, label in dataset.samples]) 

    feature_names = list(dataset.feature_df.iloc[:, 0])

    save_path = os.path.dirname(RPKM_features)
    if task == 'BOR':
        mi_result_path = os.path.join(save_path, "mutual_info_results.npz")
    elif task == 'not_progression_free':
        mi_result_path = os.path.join(save_path, "mutual_info_results_NPF.npz")
    else:
        raise ValueError(f"Unknown Task {task}")

    if os.path.exists(mi_result_path):
        data = np.load(mi_result_path)
        mi_scores = data['mi_scores']
        cumulative_mi = data['cumulative_mi']
    else:
        mi_scores = mutual_info_classif(X, y, discrete_features=False)
        sorted_mi_scores = np.sort(mi_scores)[::-1]
        cumulative_mi = np.cumsum(sorted_mi_scores) / np.sum(sorted_mi_scores)
        np.savez(mi_result_path, mi_scores=mi_scores, cumulative_mi=cumulative_mi)


    mutual_info_plots(task, cumulative_mi, mi_scores, save_path, feature_names)

    save_mi_ranking_to_csv(task, mi_scores, save_path, feature_names)


if __name__ == "__main__":
    task = 'BOR' # BOR or not_progression_free
    # CA209 - 538
    label_file1 = "../dataset/PRJEB43119/PRJEB43119_label.csv"
    RPKM_features1 = "../dataset/PRJEB43119/CA209-538-200contig_ReferenceGenome_RPKM_Features_PRJEB43119.csv"
    check_mutual_info(label_file1, RPKM_features1, task)