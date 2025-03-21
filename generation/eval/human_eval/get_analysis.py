import pandas as pd
import argparse
import pathlib


class PrepareData: 
    def __init__(self, config):
        self.class_name = self.__class__.__name__
        self.config = config

    def process_row(self, row):
        # Modify columns as needed based on your conditions
        if 'plan' not in row['model_a']:
            row["model_a"], row["model_b"] = row["model_b"], row["model_a"]
            row["rank_a"], row["rank_b"] = row["rank_b"], row["rank_a"]
            row["hallucinate_a"], row["hallucinate_b"] = row["hallucinate_b"], row["hallucinate_a"]
        return row

    def main(self):
        jsonl_path = self.config.file_path
        df = pd.read_json(path_or_buf=jsonl_path, lines=True)
        # print(df)
        subset_df = df[["example_id", "model_a", "model_b", "rank_a", "rank_b", "hallucinate_a", "hallucinate_b", "index"]]
        # print(subset_df)
        file_name="human_output.csv"
        # subset_df.to_csv(file_name, sep='\t')

        output_df = subset_df.apply(lambda row: self.process_row(row), axis=1)
        print(output_df)
        # print(output_df.describe())

        # Create a new DataFrame with rows as values of 'model_a' and columns for counts of 'rank_a' values
        rank_a_df = pd.pivot_table(output_df, index='model_a', columns='rank_a', values='example_id', aggfunc='count', fill_value=0)
        print(rank_a_df)
        rank_b_df = pd.pivot_table(output_df, index='model_b', columns='rank_b', values='example_id', aggfunc='count', fill_value=0)
        print(rank_b_df)
        hallucinate_a_df = pd.pivot_table(output_df, index='model_a', columns='hallucinate_a', values='example_id', aggfunc='count', fill_value=0)
        print(hallucinate_a_df)
        hallucinate_b_df = pd.pivot_table(output_df, index='model_b', columns='hallucinate_b', values='example_id', aggfunc='count', fill_value=0)
        print(hallucinate_b_df)


def parse_args() -> argparse.Namespace:
    """
    Argument parser function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file_path",
        default="sample_output.jsonl",
        help="File path",
    )
    #         default="sample_output_both.jsonl",

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    sample = PrepareData(parsed_args)
    sample.main()
    # cur_dir = pathlib.Path(__file__).parent.resolve()
    # savedir = f"{cur_dir}/outputs"
