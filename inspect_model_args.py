
import torch
import sys
import os
from transformers import TrainingArguments

def inspect_args(path):
    print(f"Inspecting: {path}")
    if not os.path.exists(path):
        print("File not found.")
        return

    try:
        args = torch.load(path, weights_only=False)
        # TrainingArguments might be saved as the object itself or dict
        if isinstance(args, TrainingArguments):
            print("Found TrainingArguments object.")
            # Verify known attributes that point to data
            if hasattr(args, 'dataset_name'):
                print(f"dataset_name: {args.dataset_name}")
            else:
                print("dataset_name attribute not found in TrainingArguments.")
                
            if hasattr(args, 'train_file'):
                print(f"train_file: {args.train_file}")
                
            # Check for any other hints
            print(f"output_dir: {args.output_dir}")
        else:
            print(f"Loaded object type: {type(args)}")
            print(args)
    except Exception as e:
        print(f"Error loading: {e}")

if __name__ == "__main__":
    # Check both possible locations
    inspect_args("models/kure_finetuned/encoder/training_args.bin")
    inspect_args("models/kure_finetuned/training_args.bin")
