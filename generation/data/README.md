### Introduction

We use Semantic Scholar API to get the associated data with the references. Follow the `run_data_creation.sh` for more information. Follow the instructions on the official [website](https://www.semanticscholar.org/product/api#api-key) to get the API key. Please make sure you have the `S2_API_KEY` set in your environment.

```
# For linux you can save your key in bashrc or zshrc.
echo "export S2_API_KEY='yourkey'" >> ~/.bashhrc
```


```
1. Download the S2ORC full dataset (python download_s2orc_full_data.py)
2. Filter the data for getting entries related to MultiX-Science as well as 2308 entries.
3. Save as HF dataset.
```

All the arguments are stored in `utils.py`. Please follow the `filter_s2_data.py` to see the individual steps. 
This script can be extended to get the data related to recent entries. 

### Dataset

We also release the `RollingEval-Aug` as HuggingFace [dataset](https://huggingface.co/datasets/shubhamagarwal92/RollingEval-Aug). You can load the dataset as:


```
from datasets import load_dataset

dataset_name = "shubhamagarwal92/RollingEval-Aug"
split = "test"
dataset = load_dataset(dataset_name, split=split)
```