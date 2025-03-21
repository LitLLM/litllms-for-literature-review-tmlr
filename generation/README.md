### Installation

Download miniconda.

Use [setup.sh](./setup.sh) to install the required packages. Alternatively install using

```
conda create -n litllm-generation python=3.11 -y
```

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -c conda-forge transformers -y
conda install jupyter
pip install ipdb

pip install langchain
pip install spacy

python -m spacy download en_core_web_sm
# or pip install transformers

pip install evaluate==0.4.0

pip install sentencepiece
pip install ninja
pip install flash-attn --no-build-isolation
pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary

pip install scipy
pip install optimum>=1.12.0
# pip install auto-gptq
# pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

```


```
# https://huggingface.co/togethercomputer/LLaMA-2-7B-32K/discussions/25
pip install flash-attn==2.1.1 --no-build-isolation
pip install git+https://github.com/HazyResearch/flash-attention.git@v2.1.1#subdirectory=csrc/rotary

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# https://github.com/PanQiWei/AutoGPTQ/issues/160
pip uninstall -y auto-gptq
pip install https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.2.2/auto_gptq-0.2.2+cu118-cp310-cp310-linux_x86_64.whl

```

```
pip install autopep8
pip install black
pip install pylint
```

```
pip install ray
pip install factool
pip install shortuuid
```

```
pip install arxivscraper pdf2image natsort fuzzysearch pdfkit scrapy 
pip install beautifulsoup4 pylatexenc pdftotext unidecode  python-magic lxml pyalex
pip install psycopg2-binary

# pip install chardet
```

### Experiments

Run the scraper script from repo dir as 

```
python -m autoreview.arxiv_scraper
```

To run the pipeline, 

```
python -m autoreview.models.pipeline
```

#### HF login

See: 
```
# https://discuss.huggingface.co/t/how-to-login-to-huggingface-hub-with-access-token/22498
# python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('MY_HUGGINGFACE_TOKEN_HERE')"    
```

### 6. Data creation 

To install gsutil

https://cloud.google.com/storage/docs/gsutil_install#linux

```
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-444.0.0-linux-x86_64.tar.gz
tar -xf google-cloud-cli-444.0.0-linux-x86.tar.gz
./google-cloud-sdk/install.sh

```

Install awscli

https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

Dataset snapshot release v0.2.0 Latest

https://github.com/mattbierbaum/arxiv-public-datasets/releases/tag/v0.2.0

