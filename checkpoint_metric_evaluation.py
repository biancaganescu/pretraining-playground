"""
Script to extract additional evaluation metrics from the Pythia model over the course of training.
"""

__author__ = 'Richard Diehl Martinez'

from datasets import load_dataset
import torch
import gc
import os
import json
import math
from transformers import GPTNeoXForCausalLM
from tqdm import tqdm
import click
from huggingface_hub import HfApi
import shutil

import multiprocessing

# Initial constants

checkpoint_dataset = load_dataset(
    "rdiehlmartinez/pythia-pile-presampled",
    "checkpoints",
    split='train',
    num_proc=multiprocessing.cpu_count()
)

model_sizes = ["70m", "160m", "410m", "1b", "1.4b", "2.8b"]

# checkpoint step stored by pythia 

# checkpointing steps used in evaluation by pythia 
checkpoint_steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, ]
checkpoint_steps.extend([3000 + (i * 10000) for i in range(0, 15)])

ORIGINAL_BATCH_SIZE = 1024 
REDUCED_BATCH_SIZE = 128 

MAX_STEP = 142_999 # Last step in training (used to index final batc)

# NOTE: setting up the data batch sizes 

ordered_steps = list(set(checkpoint_dataset['step']))
ordered_steps.sort()
step_to_start_index = {step: i*1024 for i, step in enumerate(ordered_steps)}

def get_data_batch(step, include_labels=True):
    """
    Get a data batch for a given step in the training process.
    """

    step = min(step, MAX_STEP)

    assert(step in step_to_start_index), f"Step {step} not valid checkpoint step."
    start_idx = step_to_start_index[step]
    end_idx = start_idx + 1024

    return {
        "input_ids": torch.tensor(checkpoint_dataset[start_idx:end_idx]['ids'], device='cuda'),
        "labels": torch.tensor(checkpoint_dataset[start_idx:end_idx]['ids'], device='cuda') if include_labels else None
    }

def forward_pass(model, batch, debug=False, verbose=False):
    """
    Perform a forward pass of the model on a given batch of data; assumes that the model 
    has hooks setup to save the hidden states at each layer.
    """
    if debug:
        torch.cuda.memory._record_memory_history(max_entries=100000)
        # split up the last batch into smaller batches that can fit on the GPU 
        # automatically find the largest batch size that can fit on the GPU
        # and then use that to split up the last batch

    batch_index = 0

    total_loss = 0.0

    batch_size = 1
    static_batch_size = None # NOTE: static_batch_size is only set when batch size is reduced

    while batch_index < REDUCED_BATCH_SIZE:

        if verbose:
            print("START OF LOOP")
            print("memory: ", torch.cuda.memory_allocated()/1e9, "GB")

        try:
            if static_batch_size is None:
                _batch_size = batch_size
            else: 
                # NOTE: reached when we've run out of memory and have reduced the batch size
                _batch_size = static_batch_size

            batch_end_index = min(batch_index + _batch_size, REDUCED_BATCH_SIZE)

            if verbose:
                print(f"Batch index: {batch_index}, Batch end index: {batch_end_index}")
                print(f"Batch size: {_batch_size}")

            _inputs = batch['input_ids'][batch_index : batch_end_index]

            if verbose:
                print(f"Shape of current sub-batch inputs: {_inputs.shape}")

            _labels = batch['labels'][batch_index : batch_end_index]

            if verbose:
                print("AFTER INPUTS")
                print("memory: ", torch.cuda.memory_allocated()/1e9, "GB")

            _loss = model(_inputs, labels=_labels).loss.item()

            if verbose:
                print("AFTER MODEL")
                print("memory: ", torch.cuda.memory_allocated()/1e9, "GB")

        except RuntimeError:
            # NOTE: Exception is thrown when the batch size is too large for the GPU

            if batch_size == 1:
                raise Exception("Batch size of 1 is too large for the GPU")

            _batch_size //= 2
            static_batch_size = _batch_size
            if verbose:
                print(f"Reducing batch size to: {_batch_size}")

            gc.collect()
            torch.cuda.empty_cache()

            continue

        total_loss += _loss * _batch_size

        batch_index = batch_end_index

        if static_batch_size is None:
            batch_size *= 2

    if debug:
        torch.cuda.memory._dump_snapshot("memory_snapshot_NA.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    return total_loss/REDUCED_BATCH_SIZE


def load_model(model_size, checkpoint_step):
    """
    Load the model at a given checkpoint step.
    """
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{model_size}-deduped",
        revision=f"step{checkpoint_step}",
        cache_dir=f"./pythia-{model_size}-deduped/step{checkpoint_step}",
    ).to('cuda')

    return model


#### --- MAIN SCRIPT --- ####
@click.command()
@click.option("--model_size", help="Model size to extract metrics from", type=str)
@click.option("--delete_after/--no-delete_after", default=True, help="Delete the activations and weights after saving")
def main(model_size, delete_after):    
    """
    Extract the hidden states, weights and gradients of the Pythia model over the course of training.
    """

    assert(model_size in model_sizes), f"Model size {model_size} not valid."

    os.makedirs("training_evals", exist_ok=True)
    model_folder = f"training_evals/{model_size}"

    # create directory under model_metrics for the model size
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    hf_api = HfApi()
    _hf_files = hf_api.list_repo_files("rdiehlmartinez/pythia-training-evals", repo_type="dataset")
    hf_files = ["training_evals/"+file.replace("models/", "") for file in _hf_files]

    for checkpoint_step in tqdm(checkpoint_steps, leave=False):
        checkpoint_folder = f"{model_folder}/checkpoint_{checkpoint_step}"
        # create directory for the given checkpoint 
        if not os.path.exists(checkpoint_folder):
            os.mkdir(checkpoint_folder)

        # --- Extracting Loss ---
        eval_file_path = os.path.join(
            checkpoint_folder, f"evals.json"
        )
       
        if not eval_file_path in hf_files:
            _model_checkpoint = load_model(model_size, checkpoint_step)
            loss_data_batch = get_data_batch(checkpoint_step) # extracting loss from current batch 
            ppl_data_batch = get_data_batch(MAX_STEP)

            evals = {
                "loss": forward_pass(_model_checkpoint, loss_data_batch, debug=False, verbose=False),
                "ppl": math.exp(forward_pass(_model_checkpoint, ppl_data_batch, debug=False, verbose=False)),
            }

            with open(eval_file_path, 'w') as f:
                f.write(json.dumps(evals))

            hf_api.upload_folder(
                folder_path=checkpoint_folder,
                path_in_repo=f"models/{model_size}/checkpoint_{checkpoint_step}",
                repo_id="rdiehlmartinez/pythia-training-evals",
                repo_type="dataset",
            )
    
    if delete_after:
        # delete the model size folder
        shutil.rmtree(model_folder)
        
if __name__ == "__main__":
    main()
