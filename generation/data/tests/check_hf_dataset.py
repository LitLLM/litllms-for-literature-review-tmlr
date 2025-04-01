from utils import parse_args, get_hf_dataset, replace_cite_text_with_mxs_cite


def get_pandas_percentile(df, col_name_list):
    # https://gist.github.com/shubhamagarwal92/13e2c41d09156c3810740d7697a883d1
    # https://stackoverflow.com/questions/34026089/create-dataframe-from-another-dataframe-describe-pandas
    describe_df = df[col_name_list].describe(percentiles=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    print(describe_df)
    return describe_df



dataset = get_hf_dataset("shubhamagarwal92/rw_2308_filtered", split="test", small_dataset=False)
# dataset = get_hf_dataset("shubhamagarwal92/rw_2308_filtered", split="test", small_dataset=False, redownload=True)

# print(dataset[1])
print(dataset.column_names)
# ['aid', 'mid', 'abstract', 'corpusid', 'text_except_rw', 'title', 'related_work', 
# 'original_related_work', 'ref_abstract', 'ref_abstract_original', 'ref_abstract_full_text', 
# 'ref_abstract_full_text_original', 'total_cites']
index = 0
# print(dataset[index]["ref_abstract"])
# print(dataset[index]["original_related_work"])
# print(dataset[index]["related_work"])
# print(dataset[index]["ref_abstract"]["cite_N"])
# print(dataset[index]["ref_abstract_original"]["cite_N"])
# original_related_work = dataset[index]["original_related_work"]
# related_work = dataset[index]["related_work"]
# dataset[index]["ref_abstract_original"]["cite_N"]
# df_pandas[["title", "text_except_rw", "found", "total_words"]] = df_pandas.apply(append_data_text, df_to_filter=df, axis=1)
# size, average document length, average summary length, average citation number


def get_dataset_summary(dataset):
    df_pandas = dataset.to_pandas()
    df_pandas['words_abstract'] = df_pandas['abstract'].str.split().str.len()
    df_pandas['words_related_work'] = df_pandas['related_work'].str.split().str.len()
    get_pandas_percentile(df_pandas, col_name_list=["words_abstract", "words_related_work"])



get_dataset_summary(dataset)


dataset = get_hf_dataset("multi_x_science_sum", split="test", small_dataset=False)
get_dataset_summary(dataset)
