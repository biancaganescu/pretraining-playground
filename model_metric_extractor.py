"""
Script to extract the hidden states, weights and grads of the Pythia model over the 
course of training.
"""

__author__ = 'Richard Diehl Martinez'

from datasets import load_dataset
import torch
import re 
import gc
import os
from transformers import GPTNeoXForCausalLM
from tqdm import tqdm
import pickle
import click
from huggingface_hub import HfApi

import multiprocessing

import torch.nn.functional as F

# Initial constants

checkpoint_dataset = load_dataset(
    "rdiehlmartinez/pythia-pile-presampled",
    "checkpoints",
    split='train',
    num_proc=multiprocessing.cpu_count()
)

model_sizes = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b"]
model_sizes = ["1.4b", "2.8b", "6.9b"]

# checkpoint step stored by pythia 

# checkpointing steps used in evaluation by pythia 
checkpoint_steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, ]
checkpoint_steps.extend([3000 + (i * 10000) for i in range(0, 15)])

# attention layers to analyze
target_layers_suffix = ["attention.query_key_value", "attention.dense", "mlp.dense_4h_to_h"]

ORIGINAL_BATCH_SIZE = 1024 
REDUCED_BATCH_SIZE = 128 

MAX_STEP = 142_999 # Last step in training (used to index final batc)

# NOTE: setting up the data batch sizes 

ordered_steps = list(set(checkpoint_dataset['step']))
ordered_steps.sort()
step_to_start_index = {step: i*1024 for i, step in enumerate(ordered_steps)}

def get_data_batch(step):
    """
    Get a data batch for a given step in the training process.
    """

    assert(step in step_to_start_index), f"Step {step} not valid checkpoint step."
    start_idx = step_to_start_index[step]
    end_idx = start_idx + 1024

    return {
        "input_ids": torch.tensor(checkpoint_dataset[start_idx:end_idx]['ids'], device='cuda'),
    }

def get_gradient_batches(step: int): 
    """
    Return a generator of data batches for the valid gradient steps around the checkpoint step.
    """
    valid_gradient_steps = list(
        range(max(0, step-5), min(step+6, 143_000))
    ) 
    return ((get_data_batch(step), step) for step in valid_gradient_steps)


class CheckpointStateMetrics:
    """
    Class to save the revelant state metrics for a given checkpoint step.
    """
    def __init__(self, checkpoint_step: int, model_size: int):
        self.checkpoint_step = checkpoint_step
        self.model_size = model_size

    def __repr__(self):
        return f"HiddenStateSaver(checkpoint_step={self.checkpoint_step}, model_size={self.model_size})"

    def get_forward_hook(self, module_name,):
        def _forward_hook(module, _, module_out):

            if "attention.query_key_value" in module_name:
                hidden_states_out = module_out[..., 2*module_out.shape[-1]//3:][:, -1, :].detach().cpu()

            elif "attention.dense" in module_name:
                # Get name of the qkv module in the same layer 
                qkv_module_name = module._global_module_name.replace("attention.dense", "attention.query_key_value")
                previous_module_output = self.checkpoint_activations[qkv_module_name]

                curr_batch_size = module_out.shape[0]
                previous_module_output = previous_module_output[-curr_batch_size:].to('cuda')

                # NOTE: need to call directly to not activate module hook 
                hidden_states_out = F.linear(previous_module_output, module.weight, module.bias)
                hidden_states_out = hidden_states_out.detach().cpu()

            elif "mlp.dense_4h_to_h" in module_name:
                hidden_states_out = module_out.detach().cpu()[:, -1, :]

            # check if there is already a key for the module name 
            if module_name not in self.checkpoint_activations:
                # if there is no key, then we create a new key and store the hidden states
                self.checkpoint_activations[module_name] = hidden_states_out

                # extract the weight matrix just once 
                weight_matrix = module.weight.detach().cpu()
                self.checkpoint_weights[module_name] = weight_matrix
            else:
                # if there is already a key, then we concatenate the new hidden states to the existing ones
                self.checkpoint_activations[module_name] = torch.cat(
                    (self.checkpoint_activations[module_name], hidden_states_out)
                )
        
        return _forward_hook

    def extract_grads(self, model):
        """
        Extract gradients from the target tensors of the model -- assumes that the model has 
        accumulated gradients from one or more backward passes. 
        """

        checkpoint_step_grads = {}
       
        for name, param in model.named_parameters():
            # only do this for the weight matrix of the target_layers_suffix
            if any(suff_name in name for suff_name in target_layers_suffix) and "weight" in name:
                assert(param.grad is not None),\
                    "Gradient is None for layer: {name} at step: {step}"
                name = re.sub(r"\.weight", "", name)
                checkpoint_step_grads[name] = param.grad.detach().cpu() 
        
        return checkpoint_step_grads

    def cleanup_hidden_states(self, batch_index):
        """
        Cleans up the hidden states if we run out of memory during the forward pass. We want 
        to ensure that the hidden states are the same size as the batch index. In practice, 
        the activations at a given layer might be more than batch_index because at that layer
        we did not run out of memory (only later). 
        """

        for layer_name, activations in self.checkpoint_activations.items():
            if activations.shape[0] > batch_index:
                self.checkpoint_activations[layer_name] = activations[:batch_index]

    def save(self, file_name, data):
        with open(file_name, "wb") as f:
            pickle.dump(data, f)

def setup_forward_hooks(model, hidden_state_saver, verbose=False):
    """
    Function to setup forward hooks for the model to save activations and weights at each layer.
    """

    hidden_state_saver.checkpoint_activations = {}
    hidden_state_saver.checkpoint_weights = {}

    forward_hooks = []
    for name, module in model.named_modules():
        # NOTE: We are only interested in the dense layers of the attention heads
        if any(layer in name for layer in target_layers_suffix):
            if verbose:
                print("registering hook for: ", name)
            _forward_hook = module.register_forward_hook(hidden_state_saver.get_forward_hook(name,))
            forward_hooks.append(_forward_hook)

            module._global_module_name = name
    
    return forward_hooks 

def forward_pass(model, batch, checkpoint_state_metrics: CheckpointStateMetrics, debug=False, verbose=False):
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

            if 'labels' in batch: 
                # NOTE: If labels are present, then we are iterating over the gradient batches 
                _labels = batch['labels'][batch_index : batch_end_index]
            else:
                _labels = None 

            if verbose:
                print("AFTER INPUTS")
                print("memory: ", torch.cuda.memory_allocated()/1e9, "GB")

            if _labels is None:
                # we can throw away the outputs, we are only interested in the hidden states
                with torch.no_grad():
                    _ = model(_inputs)

            else: 
                # NOTE: we are performing the forward and backward passes to get the gradients 
                _outputs = model(_inputs, labels=_labels)

                try: 
                    # TODO: test whether the graidnet losses are what is expected
                    _outputs['loss'].backward()
                except: 
                    # NOTE - can't figure out how often we'll have an issue in the backward call 
                    # so just exit
                    raise Exception("Error in backward pass")

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

            if _labels is None:
                # NOTE: this is kind of a hack, _labels None means we are only doing a forward pass
                # Only in this case, we need to clean up the hidden states if we hit an OOM issue
                checkpoint_state_metrics.cleanup_hidden_states(batch_index)

            continue

        batch_index = batch_end_index

        if static_batch_size is None:
            batch_size *= 2

    if debug:
        torch.cuda.memory._dump_snapshot("memory_snapshot_NA.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)


#### --- MAIN SCRIPT --- ####
@click.command()
@click.option("--model_size", help="Model size to extract metrics from", type=str)
@click.option("--delete_after/--no-delete_after", default=True, help="Delete the activations and weights after saving")
def main(model_size, delete_after):    
    """
    Extract the hidden states, weights and gradients of the Pythia model over the course of training.
    """

    assert(model_size in model_sizes), f"Model size {model_size} not valid."

    if not os.path.exists("model_metrics"):
        os.mkdir("model_metrics")

    model_folder = f"model_metrics/{model_size}"

    # create directory under model_metrics for the model size
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    hf_api = HfApi()
    _hf_files = hf_api.list_files("rdiehlmartinez/pythia-training-metrics", repo_type="dataset")
    hf_files = ["model_metrics/"+file.replace("models/", "") for file in _hf_files]

    for checkpoint_step in tqdm(checkpoint_steps, leave=False):
        checkpoint_folder = f"model_metrics/{model_size}/checkpoint_{checkpoint_step}"
        # create directory for the given checkpoint 
        if not os.path.exists(checkpoint_folder):
            os.mkdir(checkpoint_folder)

        # saving out the checkpoint weight and activations  
        activations_file_path = os.path.join(
            checkpoint_folder, f"checkpoint_activations.pickle"
        )
        weights_file_path = os.path.join(
            checkpoint_folder, f"weights_activations.pickle"
        )

        _model_checkpoint = GPTNeoXForCausalLM.from_pretrained(
            f"EleutherAI/pythia-{model_size}-deduped",
            revision=f"step{checkpoint_step}",
            cache_dir=f"./pythia-{model_size}-deduped/step{checkpoint_step}",
            ).to('cuda')

        checkpoint_state_metrics = CheckpointStateMetrics(checkpoint_step, model_size)

        if not (activations_file_path in hf_files and weights_file_path in hf_files):
            # NOTE: these get saved out at the same time 
            forward_hooks = setup_forward_hooks(_model_checkpoint, checkpoint_state_metrics)

            data_batch = get_data_batch(MAX_STEP) # extracting activation information from the last batch

            forward_pass(_model_checkpoint, data_batch, checkpoint_state_metrics, debug=False, verbose=False)

            for _hook in forward_hooks:
                # NOTE: removing the forward hook so that they don't fire during the backward 
                # gradient computatoin 
                _hook.remove()

            checkpoint_state_metrics.save(
                activations_file_path, checkpoint_state_metrics.checkpoint_activations
            )
            checkpoint_state_metrics.save(
                weights_file_path, checkpoint_state_metrics.checkpoint_weights
            )

            hf_api.upload_folder(
                folder_path=checkpoint_folder,
                path_in_repo=f"models/{model_size}/checkpoint_{checkpoint_step}",
                repo_id="rdiehlmartinez/pythia-training-metrics",
                repo_type="dataset",
                allow_patterns=["checkpoint_activations.pickle", "weights_activations.pickle"]
            )

            # NOTE: I think this helps reduce RAM usage
            del checkpoint_state_metrics.checkpoint_activations
            del checkpoint_state_metrics.checkpoint_weights

        gradient_batches = get_gradient_batches(checkpoint_step)

        for _grad_batch, step in gradient_batches:
            grad_step_file_path = os.path.join(
                checkpoint_folder, f"checkpoint_gradients_{step}.pickle"
            )

            if grad_step_file_path in hf_files:
                continue

            # Run the backward pass on the model to get the gradients 
            _model_checkpoint.zero_grad()

            grad_batch = { 
                "labels": _grad_batch['input_ids'].clone().detach(),
                **_grad_batch,
            }

            forward_pass(_model_checkpoint, grad_batch, checkpoint_state_metrics, debug=False, verbose=False)

            # extract the tensor grads 
            grads = checkpoint_state_metrics.extract_grads(_model_checkpoint)

            # save the gradient state metrics

            checkpoint_state_metrics.save(
                grad_step_file_path, grads
            )

            hf_api.upload_folder(
                folder_path=checkpoint_folder,
                path_in_repo=f"models/{model_size}/checkpoint_{checkpoint_step}",
                repo_id="rdiehlmartinez/pythia-training-metrics",
                repo_type="dataset",
                allow_patterns=[f"checkpoint_gradients_{step}.pickle"]
            )
        
        if delete_after:
            # delete the checkpoint folder 
            os.rmdir(checkpoint_folder)
        




if __name__ == "__main__":
    main()
