import pickle as pkl
import os 
from huggingface_hub import login as hf_login
from transformers import Trainer
from typing import List, Dict
import torch
from .ml_utils import get_gpu_name


def set_env_variables(cache_dir: str = "/mnt/home/cached/", setup_cache: bool = False, do_hf_login: bool = True):
    # get_gpu_name()
    if setup_cache:
        os.environ["HF_HOME"] = cache_dir
        os.environ["TORCH_HOME"] = cache_dir
    HF_TOKEN = os.environ['HF_TOKEN']
    if do_hf_login:
        hf_login(token=HF_TOKEN)    


class CombinedArguments:
    """
    Run as
    combined_args = CombinedArguments(my_dataclass, cmd_args)
    """
    def __init__(self, dataclass_instance, argparse_namespace):
        # Convert the dataclass instance to a dictionary
        dataclass_dict = dataclass_instance.__dict__
        # Remove the "__dataclass_fields__" key from the dictionary
        dataclass_dict.pop("__dataclass_fields__", None)
        # Get the arguments from the argparse namespace as a dictionary
        argparse_dict = vars(argparse_namespace)
        # Combine the two dictionaries
        combined_args = {**dataclass_dict, **argparse_dict}
        # Set the combined arguments as attributes of the class
        self.__dict__.update(combined_args)


class SNOWTrainer(Trainer):
    def __init__(self, savedir, *args, **kwargs):
        self.savedir = savedir
        self.score_list = []
        return super().__init__(*args, **kwargs)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
        self.score_list.append(logs)
        with open(f"{self.savedir}/score_list.pkl", 'wb') as file:
            pkl.dump(self.score_list, file)
