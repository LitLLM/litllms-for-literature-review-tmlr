import argparse
from dataclasses import dataclass

# Define a dataclass with some arguments
@dataclass
class MyDataClass:
    arg1: str
    arg2: int

def parse_args():
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--arg3", type=str, default="Hi!", help="Argument 1")
    parser.add_argument("--arg4", type=int, help="Argument 2")
    args = parser.parse_args()
    return args


class CombinedArguments:
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

if __name__ == "__main__":
    # Parse command-line arguments
    cmd_args = parse_args()

    # Create a dataclass instance
    my_dataclass = MyDataClass(arg1="default_value", arg2=42)

    # Create the CombinedArguments class
    combined_args = CombinedArguments(my_dataclass, cmd_args)

    # Now you can access the combined arguments as class attributes
    print("Combined Arguments:")
    print("arg1:", combined_args.arg1)
    print("arg2:", combined_args.arg2)
    print("arg4:", combined_args.arg3)
    print("arg3:", combined_args.arg4)
