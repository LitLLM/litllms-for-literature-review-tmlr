"""
Visualization script for literature recommendation metrics.

This script calculates and visualizes precision, recall, and normalized recall metrics
for literature recommendation systems, comparing original citations with GPT-4 and
SPECTER reranking methods.

Usage:
    python visualize.py --dataset [latest|rw2308] --output_file metrics_plot.pdf
"""
import os
os.environ['KMP_AFFINITY'] = 'disabled'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import argparse

sns.set_style("whitegrid")
colors = sns.color_palette("colorblind")

# Configuration dictionary
CONFIG = {
    "latest": {
        "ground_truth": "/mnt/home/agentic_attribution/dataset/ground_truth/ground_truth_latest_500_final_sqmq.csv",
        "cite": "/mnt/home/agentic_attribution/dataset/previous_results/latest/cited_papers_aggregated_450_0.2_gpt-4-turbo-preview.csv",
        "gpt_rerank": "/mnt/home/agentic_attribution/dataset/previous_results/latest/reranked_papers_aggregated_450_0.2_gpt-4-turbo-preview.csv",
        "specter_rerank": "/mnt/home/agentic_attribution/dataset/previous_results/latest/reranked_papers_aggregated_450_0.2_emb.csv",
    },
    "rw2308": {
        "ground_truth": "/mnt/home/agentic_attribution/dataset/ground_truth/ground_truth_rw2308_500.csv",
        "cite": "/mnt/home/agentic_attribution/dataset/previous_results/aug_23/cited_papers_aggregated_100_0.2.csv",
        "gpt_rerank": "/mnt/home/agentic_attribution/dataset/previous_results/aug_23/reranked_papers_aggregated_100_0.2.csv",
        "specter_rerank": "/mnt/home/agentic_attribution/dataset/previous_results/aug_23/reranked_papers_aggregated_100_0.2_emb.csv",
    },
}


def calculate_metrics_at_k(k, df_ground_truth, df_cite, df_rerank):
    """
    Calculate precision, recall, and normalized recall metrics for a given k value.

    Args:
        k (int): The number of top recommendations to consider.
        df_ground_truth (pd.DataFrame): DataFrame containing ground truth data.
        df_cite (pd.DataFrame): DataFrame containing citation data.
        df_rerank (pd.DataFrame): DataFrame containing reranked data.

    Returns:
        dict: A dictionary containing calculated metrics.
    """
    precisions_original = []
    recalls_original = []
    normalized_recalls_original = []
    precisions_reranked = []
    recalls_reranked = []
    normalized_recalls_reranked = []

    for i in range(len(df_ground_truth)):
        if not df_ground_truth.iloc[i]['ground_references_ids']:
            continue
        if pd.isna(df_ground_truth.iloc[i]['abstract']):
            continue

        ground_truth_papers = set(id.strip() for id in df_ground_truth.iloc[i]['ground_references_ids'])
        recommended_papers = [id.strip() for id in df_cite.iloc[i]['cited_references_id']]
        reranked_papers = [id.strip() for id in df_rerank.iloc[i]['reranked_papers_id']]

        intersected_list_original = set(recommended_papers).intersection(ground_truth_papers)
        top_k_recommended = set(recommended_papers[:k])
        top_k_reranked = set(reranked_papers[:k])

        relevant_in_top_k_original = top_k_recommended.intersection(intersected_list_original)
        relevant_in_top_k_reranked = top_k_reranked.intersection(intersected_list_original)

        recall_original = len(relevant_in_top_k_original) / len(ground_truth_papers) if ground_truth_papers else 0
        precision_original = len(relevant_in_top_k_original) / k if k > 0 else 0

        recall_reranked = len(relevant_in_top_k_reranked) / len(ground_truth_papers) if ground_truth_papers else 0
        precision_reranked = len(relevant_in_top_k_reranked) / k if k > 0 else 0

        normalized_recall_original = len(relevant_in_top_k_original) / len(intersected_list_original) if intersected_list_original else 0
        normalized_recall_reranked = len(relevant_in_top_k_reranked) / len(intersected_list_original) if intersected_list_original else 0

        precisions_original.append(precision_original)
        recalls_original.append(recall_original)
        normalized_recalls_original.append(normalized_recall_original)
        precisions_reranked.append(precision_reranked)
        recalls_reranked.append(recall_reranked)
        normalized_recalls_reranked.append(normalized_recall_reranked)

    return {
        'precisions_original': precisions_original,
        'recalls_original': recalls_original,
        'precisions_reranked': precisions_reranked,
        'recalls_reranked': recalls_reranked,
        'mean_normalized_recall_original': np.mean(normalized_recalls_original),
        'mean_normalized_recall_reranked': np.mean(normalized_recalls_reranked)
    }


def extract_ids(papers_string):
    """
    Extract paper IDs from a string.

    Args:
        papers_string (str): String containing paper IDs.

    Returns:
        list: List of extracted paper IDs.
    """
    if pd.isna(papers_string):
        return []
    pattern = r'([a-f0-9]{40})'
    matches = re.findall(pattern, papers_string)
    return matches

def parse_papers_reranking(combined_papers):
    """
    Parse reranked papers string into a list of paper IDs and titles.

    Args:
        combined_papers (str): String containing reranked papers information.

    Returns:
        list: List of parsed paper IDs and titles.
    """
    if pd.isna(combined_papers):
        return np.nan
    pattern = r"ID: (.*?) - Title: (.*?)"
    matches = re.findall(pattern, combined_papers)
    return [f"{match[0].strip()} {match[1]}" for match in matches]


def to_percentage(x, pos):
    """Convert number to percentage string."""
    return "%1.0f%%" % (x * 100)


def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")


def plot_metrics(df_ground_truth, df_cite, df_gpt_rerank, df_specter_rerank, output_file):
    df_ground_truth['ground_references_ids'] = df_ground_truth['cited_references'].apply(extract_ids)
    df_cite['cited_references_id'] = df_cite['combined_cited_papers'].apply(extract_ids)
    df_gpt_rerank['combined_reranked_papers'] = df_gpt_rerank['combined_reranked_papers'].fillna('')
    df_gpt_rerank['reranked_papers_id'] = df_gpt_rerank['combined_reranked_papers'].apply(parse_papers_reranking)
    df_specter_rerank['combined_reranked_papers'] = df_specter_rerank['combined_reranked_papers'].fillna('')
    df_specter_rerank['reranked_papers_id'] = df_specter_rerank['combined_reranked_papers'].apply(parse_papers_reranking)

    k_values = [1, 3, 5, 7, 10, 15, 25, 50, 100]
    metrics = {
        'precision_original': [],
        'recall_original': [],
        'normalized_recall_original': [],
        'precision_gpt_reranked': [],
        'recall_gpt_reranked': [],
        'normalized_recall_gpt_reranked': [],
        'precision_specter_reranked': [],
        'recall_specter_reranked': [],
        'normalized_recall_specter_reranked': []
    }

    for k in k_values:
        result_gpt = calculate_metrics_at_k(k, df_ground_truth, df_cite, df_gpt_rerank)
        result_specter = calculate_metrics_at_k(k, df_ground_truth, df_cite, df_specter_rerank)
        
        metrics['precision_original'].append(np.mean(result_gpt['precisions_original']))
        metrics['recall_original'].append(np.mean(result_gpt['recalls_original']))
        metrics['normalized_recall_original'].append(result_gpt['mean_normalized_recall_original'])
        
        metrics['precision_gpt_reranked'].append(np.mean(result_gpt['precisions_reranked']))
        metrics['recall_gpt_reranked'].append(np.mean(result_gpt['recalls_reranked']))
        metrics['normalized_recall_gpt_reranked'].append(result_gpt['mean_normalized_recall_reranked'])
        
        metrics['precision_specter_reranked'].append(np.mean(result_specter['precisions_reranked']))
        metrics['recall_specter_reranked'].append(np.mean(result_specter['recalls_reranked']))
        metrics['normalized_recall_specter_reranked'].append(result_specter['mean_normalized_recall_reranked'])

    # Plot and save to PDF
    with PdfPages(output_file) as pdf:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel('k (Top-k Recommendations)', fontsize=14)
        ax1.set_ylabel('Precision', color=colors[0], fontsize=14)
        ax1.yaxis.set_major_formatter(FuncFormatter(to_percentage))

        # Plotting Precision values
        ax1.plot(k_values, metrics['precision_original'], color=colors[0], label='Precision (Original)', marker='o', linewidth=2)
        ax1.plot(k_values, metrics['precision_gpt_reranked'], color=colors[1], linestyle='dashed', label='Precision (GPT-4 Rerank)', marker='x', linewidth=2)
        ax1.plot(k_values, metrics['precision_specter_reranked'], color=colors[2], linestyle='dotted', label='Precision (SPECTER Rerank)', marker='s', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=colors[0])

        # Secondary y-axis for Normalized Recall
        ax2 = ax1.twinx()
        ax2.set_ylabel('Normalized Recall', color=colors[3], fontsize=14)
        ax2.yaxis.set_major_formatter(FuncFormatter(to_percentage))
        ax2.plot(k_values, metrics['normalized_recall_original'], color=colors[3], label='Normalized Recall (Original)', marker='o', linewidth=2)
        ax2.plot(k_values, metrics['normalized_recall_gpt_reranked'], color=colors[4], linestyle='dashed', label='Normalized Recall (GPT-4 Rerank)', marker='x', linewidth=2)
        ax2.plot(k_values, metrics['normalized_recall_specter_reranked'], color=colors[5], linestyle='dotted', label='Normalized Recall (SPECTER Rerank)', marker='s', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=colors[3])

        # Final adjustments for title, layout, and legend
        plt.title('RollingEval-Dec Dataset', fontweight='bold', fontsize=14)
        fig.tight_layout()
        plt.subplots_adjust(top=0.92)

        # Legend adjustment
        fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.8), bbox_transform=ax1.transAxes, fontsize=14)
        plt.xticks(k_values)

        # Save to PDF
        pdf.savefig(fig)
        plt.close(fig)
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize metrics for literature recommendations.")
    parser.add_argument('--dataset', type=str, required=True, choices=['latest', 'rw2308'],
                        help='The dataset configuration to use')
    parser.add_argument('--output_file', type=str, required=True,
                        help='The output file for the PDF plot (e.g., metrics_plot.pdf)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dataset_config = CONFIG[args.dataset]
    check_file_exists(dataset_config['ground_truth'])
    check_file_exists(dataset_config['cite'])
    check_file_exists(dataset_config['gpt_rerank'])
    check_file_exists(dataset_config['specter_rerank'])

    # Load the datasets
    df_ground_truth = pd.read_csv(dataset_config['ground_truth'], index_col=False)
    df_cite = pd.read_csv(dataset_config['cite'], index_col=False)
    df_rerank_gpt4 = pd.read_csv(dataset_config['gpt_rerank'], index_col=False)
    df_rerank_specter = pd.read_csv(dataset_config['specter_rerank'], index_col=False)

    # Plot and save to the specified output file
    plot_metrics(df_ground_truth, df_cite, df_rerank_gpt4, df_rerank_specter, args.output_file)
