"""
Script to analyze the hidden states of the Pythia model over time using the CKA metric.
"""

__author__ = 'Richard Diehl Martinez'

from datasets import load_dataset
import torch
import re 
import gc
import os
from transformers import GPTNeoXForCausalLM
from lib.cka import gram_linear, cka
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt


# Initial constants

checkpoint_dataset = load_dataset("rdiehlmartinez/pythia-pile-presampled", "checkpoints", split='train', num_proc=8)

ORIGINAL_BATCH_SIZE = 1024 
REDUCED_BATCH_SIZE = 128 

last_batch = checkpoint_dataset[-ORIGINAL_BATCH_SIZE:]

model_sizes = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b"]

# checkpoint step stored by pythia 

# checkpointing steps used in evaluation by pythia 
checkpoint_steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, ]
checkpoint_steps.extend([3000 + (i * 10000) for i in range(0, 15)])


# OLD CHECKPOINT STEPS
# checkpoint_steps = [1, 512,]
# checkpoint_steps.extend([i * 1000 for i in range(1, 144, 10)])
# checkpoint_steps.append(143000)


class HiddenStateSaver:
    """
    Class to save the hidden states of the model at a given checkpoint step. 
    """
    def __init__(self, checkpoint_step: int, model_size: int):
        self.checkpoint_step = checkpoint_step
        self.model_size = model_size

        self.checkpoint_hidden_states = {}

    def __repr__(self):
        return f"HiddenStateSaver(checkpoint_step={self.checkpoint_step}, model_size={self.model_size})"

    def get_hidden_state_hook(self, module_name,):
        def save_hidden_states_hook(module, module_in, module_out):
            # check if there is already a key for the module name 

            # NOTE Dimensions are (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM) -- 
            # we only care about the last hidden state as a representation of the sequence
            module_output = module_out.detach().cpu()[:, -1, :]

            if module_name not in self.checkpoint_hidden_states:
                # if there is no key, then we create a new key and store the hidden states
                self.checkpoint_hidden_states[module_name] = module_output
            else:
                # if there is already a key, then we concatenate the new hidden states to the existing ones
                self.checkpoint_hidden_states[module_name] = torch.cat(
                    (self.checkpoint_hidden_states[module_name], module_output)
                )
        
        return save_hidden_states_hook

    def cleanup_hidden_states(self):
        last_layer_name = list(self.checkpoint_hidden_states.keys())[-1]
        last_layer = self.checkpoint_hidden_states[last_layer_name]
        last_layer_samples = last_layer.shape[0]

        for layer_name, layer in self.checkpoint_hidden_states.items():
            if layer.shape[0] > last_layer_samples:
                self.checkpoint_hidden_states[layer_name] = layer[:last_layer_samples]



def setup_hooks(model, hidden_state_saver, verbose=False):
    """
    Function to setup hooks for the model to save the hidden states at each layer. 
    """
    hooks = []
    for name, module in model.named_modules():
        # NOTE: We are only interested in the dense layers of the attention heads
        if re.search(r"gpt_neox\.layers\.\d+\.attention\.dense", name):
            if verbose:
                print("Registering hook for: ", name)
            _hook = module.register_forward_hook(hidden_state_saver.get_hidden_state_hook(name,))
            hooks.append(_hook)
    return hooks

def forward_pass(model, batch, hidden_state_savers: list[HiddenStateSaver] = [], debug=False, verbose=False):
    """
    Perform a forward pass of the model on a given batch of data; assumes that the model 
    has hooks setup to save the hidden states at each layer.

    """
    if not isinstance(hidden_state_savers, list):
        hidden_state_savers = [hidden_state_savers]

    model.eval()

    if debug:
        torch.cuda.memory._record_memory_history(max_entries=100000)

        # split up the last batch into smaller batches that can fit on the GPU 
        # automatically find the largest batch size that can fit on the GPU
        # and then use that to split up the last batch

    batch_index = 0

    batch_size = 1
    static_batch_size = None

    while batch_index < REDUCED_BATCH_SIZE:

        if verbose:
            print("START OF LOOP")
            print("memory: ", torch.cuda.memory_allocated()/1e9, "GB")

        try:
            if static_batch_size is None:
                _batch_size = batch_size
            else: 
                _batch_size = static_batch_size

            batch_end_index = min(batch_index + _batch_size, REDUCED_BATCH_SIZE)

            if verbose:
                print(f"Batch index: {batch_index}, Batch end index: {batch_end_index}")
                print(f"Batch size: {_batch_size}")

            _inputs = torch.tensor(batch['ids'][batch_index : batch_end_index], device=torch.device('cuda'))


            if verbose:
                print("AFTER INPUTS")
                print("memory: ", torch.cuda.memory_allocated()/1e9, "GB")


            # we can throw away the outputs, we are only interested in the hidden states
            with torch.no_grad():
                _ = model(_inputs)

            if verbose:
                print("AFTER MODEL")
                print("memory: ", torch.cuda.memory_allocated()/1e9, "GB")

            gc.collect()
            torch.cuda.empty_cache()
            
            if verbose:
                print("AFTER CUDA COLLECT ")
                print("memory: ", torch.cuda.memory_allocated()/1e9, "GB")

        except RuntimeError as e:

            print("Error: ", e)

            _batch_size //= 2
            static_batch_size = _batch_size
            if verbose:
                print(f"Reducing batch size to: {_batch_size}")

            gc.collect()
            torch.cuda.empty_cache()

            for hidden_state_saver in hidden_state_savers:
                hidden_state_saver.cleanup_hidden_states()

            continue

        batch_index = batch_end_index

        if static_batch_size is None:
            batch_size *= 2


    if debug:
        torch.cuda.memory._dump_snapshot("memory_snapshot_NA.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)



#### --- MAIN SCRIPT --- ####

for model_size in model_sizes:
    print("Running analysis for model size: ", model_size)

    pickle_checkpoint_file = f"cka_analysis/cka_scores/cka_{model_size}_scores_over_checkpoint.pickle"

    if not os.path.exists(pickle_checkpoint_file):
        final_checkpoint_step = checkpoint_steps[-1]
        final_model_checkpoint = GPTNeoXForCausalLM.from_pretrained(
            f"EleutherAI/pythia-{model_size}-deduped",
            revision=f"step{final_checkpoint_step}",
            cache_dir=f"./pythia-{model_size}-deduped/step{final_checkpoint_step}",
        ).to('cuda').eval()


        last_checkpoint_hidden_states = HiddenStateSaver(final_checkpoint_step, model_size)
        setup_hooks(final_model_checkpoint, last_checkpoint_hidden_states, verbose=True)

        forward_pass(final_model_checkpoint, last_batch, last_checkpoint_hidden_states, debug=False, verbose=True)

        # Extracting layer names to compare with different model checkpoint steps
        layer_names = list(last_checkpoint_hidden_states.checkpoint_hidden_states.keys())

        cka_scores_over_checkpoint = {}

        for checkpoint_step in tqdm(checkpoint_steps):
            _model_checkpoint = GPTNeoXForCausalLM.from_pretrained(
                f"EleutherAI/pythia-{model_size}-deduped",
                revision=f"step{checkpoint_step}",
                cache_dir=f"./pythia-{model_size}-deduped/step{checkpoint_step}",
                ).to('cuda').eval()

            _hidden_state_saver = HiddenStateSaver(checkpoint_step, model_size)
            setup_hooks(_model_checkpoint, _hidden_state_saver)
            forward_pass(_model_checkpoint, last_batch, _hidden_state_saver, debug=False, verbose=False)

            cka_scores = {}

            for layer_name in layer_names:
                rep1 = last_checkpoint_hidden_states.checkpoint_hidden_states[layer_name].numpy()
                rep2 = _hidden_state_saver.checkpoint_hidden_states[layer_name].numpy()

                cka_scores[layer_name] = cka(gram_linear(rep1), gram_linear(rep2))

            cka_scores_over_checkpoint[checkpoint_step] = cka_scores

            del _model_checkpoint

        
        # save data to picke file 
        with open(pickle_checkpoint_file, "wb") as f:
            pickle.dump(cka_scores_over_checkpoint, f)
    else:
        with open(pickle_checkpoint_file, "rb") as f:
            cka_scores_over_checkpoint = pickle.load(f)

    layer_names = list(cka_scores_over_checkpoint[1].keys())
    checkpoint_steps = list(cka_scores_over_checkpoint.keys())

    # flatting cka_scores_over_checkpoint into a dictionary of lists
    cka_scores_over_checkpoint_flat = {layer_name: [] for layer_name in layer_names}
    for checkpoint_step, cka_scores in cka_scores_over_checkpoint.items():
        for layer_name, cka_score in cka_scores.items():
            cka_scores_over_checkpoint_flat[layer_name].append(cka_score)

    fig, ax = plt.subplots()
    for layer_name, cka_scores in cka_scores_over_checkpoint_flat.items():
        ax.plot(checkpoint_steps, cka_scores, label=layer_name)

    ax.set_xlabel("Checkpoint Step")
    ax.set_ylabel("CKA Score")
    ax.set_title("CKA Score Over Time")
    ax.legend()
    fig.savefig(f"cka_analysis/cka_plots/cka_{model_size}_scores_over_checkpoint.png")
