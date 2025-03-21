# Run as python -m tests.test_hf_pipeline
# PYTHONPATH=. python tests/test_hf_pipeline.py
# https://stackoverflow.com/questions/72478689/converting-arguments-to-custom-objects-using-dataclasses-package

# https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_plm.py#L231
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm_no_trainer.py#L68
# https://github.com/facebookresearch/llama-recipes/blob/main/utils/config_utils.py
# https://github.com/facebookresearch/llama-recipes/blob/main/utils/memory_utils.py

from transformers import HfArgumentParser, TrainingArguments
import argparse
# from autoreview.models.configs_llama import llama_adapter_config, lora_config, prefix_config
from dataclasses import dataclass, fields

@dataclass
class llama_adapter_config:
    adapter_len: int = 10
    adapter_layers: int = 30
    task_type: str = "CAUSAL_LM"


# def namespace_to_dataclass(namespace, dataclass_type):
#     """
#     Converts an argparse.Namespace object to a dataclass object
#     """
#     dataclass_fields = fields(dataclass_type)
#     kwargs = {field.name: getattr(namespace, field.name) for field in dataclass_fields}
#     return dataclass_type(**kwargs)

def convert_namespace_to_dataclass(namespace, dataclass_type):
    """
    Converts an argparse.Namespace object to a dataclass object
    """
    dataclass_fields = dataclass_type.__annotations__
    dataclass_kwargs = {field: getattr(namespace, field) for field in dataclass_fields}
    return dataclass_type(**dataclass_kwargs)


# def namespace_to_dataclass(namespace, dataclass_type):
#     """
#     Converts an argparse.Namespace object to a dataclass object
#     """
#     return dataclass_type(**vars(namespace))


def parse_args() -> argparse.Namespace:
    """
    Argument parser function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        default="xxxxx",
        help="Model name",
    )
    parser.add_argument(
        "-s",
        "--savedir",
        default="results/auto_review/multixscience/",
        help="Path to save dir",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    print("Args passed!")
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # print(parsed_args)
    # parser =  HfArgumentParser((parsed_args, lora_config))
    # print(parser)
    # config = namespace_to_dataclass(parsed_args, llama_adapter_config)


    config = convert_namespace_to_dataclass(parsed_args, llama_adapter_config)


