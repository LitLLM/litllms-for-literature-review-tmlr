import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from .llm_utils import extract_ids, parse_papers_reranking


def calculate_metrics_at_k(k, df_ground_truth, df_cite, df_rerank):
    precisions_original = []
    recalls_original = []
    normalized_recalls_original = []
    precisions_reranked = []
    recalls_reranked = []
    normalized_recalls_reranked = []
    zero_recall_original_info = []  # To track both index and paperId for original
    zero_recall_reranked_info = []  # To track both index and paperId for reranked

    for i in range(len(df_ground_truth)):  # Let's say 85
        if not df_ground_truth.iloc[i][
            "ground_references_ids"
        ]:  # Skip if ground_references_ids is empty
            print(f"Skipping index {i} because ground_references_ids is empty.")
            continue
        paper_id = df_ground_truth.iloc[i]["paperId"]  # Capture paperId for tracking
        ground_truth_papers = set(
            id.strip() for id in df_ground_truth.iloc[i]["ground_references_ids"]
        )  # let's say 20 (list of IDs)
        recommended_papers = [
            id.strip() for id in df_cite.iloc[i]["cited_references_id"]
        ]  # let's say 100 (list of Recommended IDs)
        reranked_papers = [
            id.strip() for id in df_rerank.iloc[i]["reranked_papers_id"]
        ]  # let's say 100 (list of Reranked IDs)

        intersected_list_original_IL = set(recommended_papers).intersection(
            ground_truth_papers
        )  # retrieved let's say 5
        # intersected_list_reranked_IL = set(reranked_papers).intersection(ground_truth_papers)

        top_k_recommended = set(
            recommended_papers[:k]
        )  # set of top k recommended papers , example k = 5 , top_k_recommended = 5
        top_k_reranked = set(
            reranked_papers[:k]
        )  # set of top k reranked papers , example k = 5 , top_k_reranked = 5

        relevant_in_top_k_original = top_k_recommended.intersection(
            intersected_list_original_IL
        )  # let's say 2
        relevant_in_top_k_reranked = top_k_reranked.intersection(
            intersected_list_original_IL
        )

        # recall_original = len(intersected_list_original_IL) / len(ground_truth_papers) if ground_truth_papers else 0
        recall_original = (
            len(relevant_in_top_k_original) / len(ground_truth_papers)
            if ground_truth_papers
            else 0
        )
        precision_original = len(relevant_in_top_k_original) / k if k > 0 else 0
        # precision_original = len(intersected_list_original_IL) / k if k > 0 else 0

        recall_reranked = (
            len(relevant_in_top_k_reranked) / len(ground_truth_papers)
            if ground_truth_papers
            else 0
        )

        precision_reranked = len(relevant_in_top_k_reranked) / k if k > 0 else 0

        if intersected_list_original_IL:
            # normalized_recall_original = len(relevant_in_top_k_original) / len(intersected_list_original_IL) if intersected_list_original_IL else 0
            normalized_recall_original = (
                len(relevant_in_top_k_original) / len(intersected_list_original_IL)
                if intersected_list_original_IL
                else 0
            )
            normalized_recalls_original.append(normalized_recall_original)
            normalized_recall_reranked = (
                len(relevant_in_top_k_reranked) / len(intersected_list_original_IL)
                if intersected_list_original_IL
                else 0
            )
            normalized_recalls_reranked.append(normalized_recall_reranked)
        else:
            zero_recall_original_info.append((i, paper_id))
            zero_recall_reranked_info.append((i, paper_id))

        precisions_original.append(precision_original)
        recalls_original.append(recall_original)
        precisions_reranked.append(precision_reranked)
        recalls_reranked.append(recall_reranked)

    print(normalized_recalls_original, normalized_recalls_reranked)

    # Calculate the mean of normalized recalls outside the loop
    mean_normalized_recall_original = (
        sum(normalized_recalls_original) / len(normalized_recalls_original)
        if normalized_recalls_original
        else 0
    )
    mean_normalized_recall_reranked = (
        sum(normalized_recalls_reranked) / len(normalized_recalls_reranked)
        if normalized_recalls_reranked
        else 0
    )

    # Additional debug or analysis information
    print(f"Mean Normalized Recall Original: {mean_normalized_recall_original}")
    print(f"Mean Normalized Recall Reranked: {mean_normalized_recall_reranked}")
    # print(f"Zero Recall Original Info (Index, Paper ID): {zero_recall_original_info}")
    # print(f"Zero Recall Reranked Info (Index, Paper ID): {zero_recall_reranked_info}")

    return {
        "precisions_original": precisions_original,
        "recalls_original": recalls_original,
        "normalized_recalls_original": normalized_recalls_original,
        "mean_normalized_recall_original": mean_normalized_recall_original,
        "precisions_reranked": precisions_reranked,
        "recalls_reranked": recalls_reranked,
        "normalized_recalls_reranked": normalized_recalls_reranked,
        "mean_normalized_recall_reranked": mean_normalized_recall_reranked,
    }


def plot_metrics(df_ground_truth, df_cite, df_rerank):
    # Assuming extract_ids and parse_papers_reranking are defined elsewhere
    df_ground_truth["ground_references_ids"] = df_ground_truth[
        "cited_references"
    ].apply(extract_ids)
    df_cite["cited_references_id"] = df_cite["combined_cited_papers"].apply(extract_ids)
    df_rerank["reranked_papers_id"] = df_rerank["combined_reranked_papers"].apply(
        parse_papers_reranking
    )

    k_values = [1, 3, 5, 7, 10, 15, 25, 50, 100]
    precision_original = []
    recall_original = []
    normalized_recall_original = []
    precision_reranked = []
    recall_reranked = []
    normalized_recall_reranked = []

    for k in k_values:
        result = calculate_metrics_at_k(k, df_ground_truth, df_cite, df_rerank)
        precision_original.append(np.mean(result["precisions_original"]))
        recall_original.append(np.mean(result["recalls_original"]))
        normalized_recall_original.append(result["mean_normalized_recall_original"])
        precision_reranked.append(np.mean(result["precisions_reranked"]))
        recall_reranked.append(np.mean(result["recalls_reranked"]))
        normalized_recall_reranked.append(result["mean_normalized_recall_reranked"])

    if not os.path.exists("results/figures"):
        os.makedirs("results/figures")
    # Plot for Precision and Recall
    plt.figure(figsize=(12, 6))
    plt.plot(k_values, precision_original, label="Precision (Original)", marker="o")
    plt.plot(k_values, recall_original, label="Recall (Original)", marker="o")
    plt.plot(k_values, precision_reranked, label="Precision (Reranked)", marker="x")
    plt.plot(k_values, recall_reranked, label="Recall (Reranked)", marker="x")
    plt.xlabel("k (Top-k Recommendations)")
    plt.ylabel("Metrics")
    plt.title(
        "Precision and Recall at Different Values of k for Original and Reranked Papers"
    )
    plt.xticks(k_values)
    plt.legend()
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    plt.savefig("results/figures/precision_recall_85.png")
    plt.show()

    # Plot for Normalized Recall
    plt.figure(figsize=(12, 6))
    plt.plot(
        k_values,
        normalized_recall_original,
        label="Normalized Recall (Original)",
        marker="o",
    )
    plt.plot(
        k_values,
        normalized_recall_reranked,
        label="Normalized Recall (Reranked)",
        marker="x",
    )
    plt.xlabel("k (Top-k Recommendations)")
    plt.ylabel("Normalized Recall")
    plt.title(
        "Normalized Recall at Different Values of k for Original and Reranked Papers"
    )
    plt.xticks(k_values)
    plt.legend()
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    plt.savefig("results/figures/normalized_recall_85.png")
    plt.show()


def main():
    # Load the data
    # df_ground_truth = pd.read_csv("./litrank/dataset/ground_truth/ground_truth_cited_ref30.csv", index_col=False)
    # df_cite = pd.read_csv('./litrank/dataset/cited_papers/cited_papers_test_dummy30_nosort_3prompts.csv', index_col=False)
    # df_rerank = pd.read_csv('./litrank/dataset/rerank_papers/reranked_papers_test_dummy30_nosort_3prompts.csv', index_col=False)

    df_ground_truth = pd.read_csv(
        "/mnt/home/lit-rank/litrank/dataset/ground_truth/ground_truth_cited_ref85_2024.csv",
        index_col=False,
    )
    df_cite = pd.read_csv(
        "/mnt/home/lit-rank/litrank/dataset/cited_papers/cited_papers_test_dummy100_nosort_3prompts_85_final.csv",
        index_col=False,
    )
    df_rerank = pd.read_csv(
        "/mnt/home/lit-rank/litrank/dataset/rerank_papers/reranked_papers_test_dummy100_nosort_3prompts_85_final.csv",
        index_col=False,
    )

    # df_cite = pd.read_csv('/mnt/home/lit-rank/litrank/dataset/cited_papers/cited_papers_test_dummy100_nosort_3prompts_100.csv', index_col=False)
    # df_rerank = pd.read_csv('/mnt/home/lit-rank/litrank/dataset/rerank_papers/reranked_papers_test_dummy100_nosort_3prompts_100.csv', index_col=False)

    plot_metrics(df_ground_truth, df_cite, df_rerank)
