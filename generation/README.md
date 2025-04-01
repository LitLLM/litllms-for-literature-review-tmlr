### Installation

Download miniconda.

Use [setup.sh](./setup.sh) to install the required packages. Alternatively install using

```
conda create -n litllm-generation python=3.11 -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -c conda-forge transformers -y

pip install spacy tiktoken langchain
python -m spacy download en_core_web_sm
pip install flash-attn==2.1.1 --no-build-isolation
```

Please make sure you have the `OPENAI_API_KEY` set in your environment. We also used Anyscale endpoint in our experiments. 

```
# For linux you can save your key in bashrc or zshrc.
echo "export OPENAI_API_KEY='yourkey'" >> ~/.bashhrc
echo "export ANYSCALE_ENDPOINT_API_KEY=''" >> ~/.bashhrc
```

### Experiments

To run the plan based generation, 

```
cd shell_scripts
bash run_gpt_plan.sh
```


For the data creation scripts, please follow the [data](./data/) folder.

### Dataset

We release the `RollingEval-Aug` as HuggingFace [dataset](https://huggingface.co/datasets/shubhamagarwal92/RollingEval-Aug). You can load the dataset as:


```
from datasets import load_dataset

dataset_name = "shubhamagarwal92/RollingEval-Aug"
split = "test"
dataset = load_dataset(dataset_name, split=split)
```

We also release the Multi-XScience test set with full papers at this [link](https://huggingface.co/datasets/shubhamagarwal92/multi_x_science_test_full). 


### Code suggestions

While we used langchain with single calls for GPT based generation, this could be improved by using [Batch API](https://platform.openai.com/docs/guides/batch) calls from GPT or Gemini. Other way could be to use [async](https://community.openai.com/t/how-to-do-asynchronous-calls-with-the-latest-api-version/478439) calls to speed up LLM generation. This would however require some effort to map the responses with correct prompt index. 

We currently don't commit to support this in the future.  