"""
Computed metrics from the extracted information of the model weights, activations and gradients
"""


from datasets import load_dataset 
import numpy as np
from lib import cka
import click 
import pickle
import os 
import time 
from tqdm import tqdm

cpu_count = os.cpu_count()

checkpoint_steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, ]
checkpoint_steps.extend([3000 + (i * 10000) for i in range(0, 15)])

def get_checkpoint_step_to_range_indices(dataset):
    """
    Get the range indices for each checkpoint step in the dataset
    """

    checkpoint_steps = dataset['checkpoint_step']

    checkpoint_step_to_range_indices = {}

    _target_checkpoint_step = checkpoint_steps[0]
    _target_checkpoint_start_idx = 0
    for idx, _curr_checkpoint_step in enumerate(checkpoint_steps):
        if _curr_checkpoint_step != _target_checkpoint_step:
            checkpoint_step_to_range_indices[_target_checkpoint_step] = (_target_checkpoint_start_idx, idx)
            _target_checkpoint_step = _curr_checkpoint_step
            _target_checkpoint_start_idx = idx
    else:
        checkpoint_step_to_range_indices[_target_checkpoint_step] = (_target_checkpoint_start_idx, idx+1)

    return checkpoint_step_to_range_indices

def compute_cka_sores(activation_dataset):
    """
    Computes the CKA scores of each model layer relative to the final layer's state after training
    """

    print("Computing CKA scores")

    last_checkpoint = activation_dataset.select(range(*get_checkpoint_step_to_range_indices(activation_dataset)[checkpoint_steps[-1]]))
    layer_names = last_checkpoint['layer_name']
    cka_scores_per_layer = {layer_name: [] for layer_name in layer_names}

    checkpoint_step_to_range_indices = get_checkpoint_step_to_range_indices(activation_dataset)

    for checkpoint_step in tqdm(checkpoint_steps):

        print("Processing checkpoint step: ", checkpoint_step)

        checkpoint_step_idx_range = checkpoint_step_to_range_indices[checkpoint_step]

        checkpoint_activations = activation_dataset.select(range(*checkpoint_step_idx_range))
        for checkpoint_layer, last_checkpoint_layer in zip(checkpoint_activations, last_checkpoint):
            layer_name = checkpoint_layer['layer_name']
            assert(layer_name == last_checkpoint_layer['layer_name'])

            layer_activation = np.array(checkpoint_layer['data'])

            last_batch_activation = np.array(last_checkpoint_layer['data'])

            cka_score = cka.feature_space_linear_cka(
                layer_activation, 
                last_batch_activation
            )

            cka_scores_per_layer[layer_name].append(cka_score)
    
    return cka_scores_per_layer

def compute_weight_magnitudes(weights_dataset):
    """
    Computes the magnitude of the weights in each layer of the model at the different timesteps.
    """

    print("Computing weight magnitudes")

    layer_names = set(weights_dataset['layer_name'])
    weight_magnitudes_per_layer = {layer_name: [] for layer_name in layer_names}

    checkpoint_step_to_range_indices = get_checkpoint_step_to_range_indices(weights_dataset)

    for checkpoint_step in tqdm(checkpoint_steps):

        print("Processing checkpoint step: ", checkpoint_step)
        checkpoint_step_idx_range = checkpoint_step_to_range_indices[checkpoint_step]

        checkpoint_weights = weights_dataset.select(range(*checkpoint_step_idx_range))
        for checkpoint_weight in checkpoint_weights: 
            layer_name = checkpoint_weight['layer_name']
            layer_weight = np.array(checkpoint_weight['data'])

            layer_weight_magnitude = np.linalg.norm(layer_weight)

            weight_magnitudes_per_layer[layer_name].append(layer_weight_magnitude)

    return weight_magnitudes_per_layer

def compute_grad_weight_magnitudes(gradient_dataset):
    """
    Computes the magnitude of the gradients in each layer of the model at the different timesteps.
    """

    print("Computing gradient weight magnitudes")

    layer_names = set(gradient_dataset['layer_name'])
    grad_weight_magnitudes_per_layer = {layer_name: [] for layer_name in layer_names}

    checkpoint_step_to_range_indices = get_checkpoint_step_to_range_indices(gradient_dataset)

    for checkpoint_step in tqdm(checkpoint_steps):

        print("Processing checkpoint step: ", checkpoint_step)
        checkpoint_step_idx_range = checkpoint_step_to_range_indices[checkpoint_step]

        checkpoint_grads = gradient_dataset.select(range(*checkpoint_step_idx_range))

        # group together indices of the same layer
        layer_name_to_indices = {}
        for idx, layer_name in enumerate(checkpoint_grads['layer_name']):
            if layer_name not in layer_name_to_indices:
                layer_name_to_indices[layer_name] = []
            layer_name_to_indices[layer_name].append(idx)

        for layer_name in layer_names:

            layer_name_indices = layer_name_to_indices[layer_name]

            layer_checkpoint_grads = checkpoint_grads.select(layer_name_indices)
            
            avg_layer_grad = None

            for layer_checkpoint_grad_data in layer_checkpoint_grads['data']:

                _grad_data = np.array(layer_checkpoint_grad_data)
                _grad_norm = np.linalg.norm(_grad_data)
            
                if avg_layer_grad is None:
                    avg_layer_grad = _grad_norm
                else: 
                    avg_layer_grad += _grad_norm

            avg_layer_grad /= len(layer_checkpoint_grads)

            grad_weight_magnitudes_per_layer[layer_name].append(avg_layer_grad)
    return grad_weight_magnitudes_per_layer


def compute_grad_sim_per_layer(gradient_dataset):
    """
    Computes the cosine similarity of the gradients in each layer of the model at the different timesteps.
    """

    print("Computing gradient similarity per layer")

    layer_names = set(gradient_dataset['layer_name'])
    grad_sim_per_layer = {layer_name: [] for layer_name in layer_names}

    checkpoint_step_to_range_indices = get_checkpoint_step_to_range_indices(gradient_dataset)

    for checkpoint_step in tqdm(checkpoint_steps):

        print("Processing checkpoint step: ", checkpoint_step)
        checkpoint_step_idx_range = checkpoint_step_to_range_indices[checkpoint_step]

        checkpoint_grads = gradient_dataset.select(range(*checkpoint_step_idx_range))

        # group together indices of the same layer
        layer_name_to_indices = {}
        for idx, layer_name in enumerate(checkpoint_grads['layer_name']):
            if layer_name not in layer_name_to_indices:
                layer_name_to_indices[layer_name] = []
            layer_name_to_indices[layer_name].append(idx)

        for layer_name in layer_names:

            layer_name_indices = layer_name_to_indices[layer_name]

            layer_checkpoint_grads = checkpoint_grads.select(layer_name_indices)
            
            avg_cosine_sim = None
            prev_grad_data = None

            for layer_checkpoint_grad_data in layer_checkpoint_grads['data']:

                _grad_data = np.array(layer_checkpoint_grad_data)
                _grad_data = _grad_data.flatten()

                if prev_grad_data is None:
                    prev_grad_data = _grad_data
                    continue

                # computing cosine similarity 
                cosine_sim = np.dot(prev_grad_data, _grad_data) / (np.linalg.norm(prev_grad_data) * np.linalg.norm(_grad_data))

                if avg_cosine_sim is None:
                    avg_cosine_sim = cosine_sim
                else: 
                    avg_cosine_sim += cosine_sim

                prev_grad_data = _grad_data

            avg_cosine_sim /= len(layer_checkpoint_grads)-1

            grad_sim_per_layer[layer_name].append(avg_cosine_sim)
    
    return grad_sim_per_layer

def get_dataset(subconfig: str):
    retry_count = 0
    sleep_time = 10

    while retry_count < 5:
        try: 
            dataset = load_dataset(
                "rdiehlmartinez/pythia-training-metrics", subconfig, split='default',
                cache_dir='/rds-d7/user/rd654/hpc-work/cache',
                writer_batch_size=100,
            )
            break
        except Exception as e:
            print("Failed to load dataset, retrying in 10 seconds. Exception:")
            print(e)
            time.sleep(sleep_time)
            retry_count += 1
            continue
    else:
        raise Exception("Failed to load dataset after 5 retries")

    return dataset


@click.command()
@click.option('--model_size') 
def main(model_size):

    activation_dataset = get_dataset(f"{model_size}__activations")
    weights_dataset = get_dataset(f"{model_size}__weights")
    gradient_dataset = get_dataset(f"{model_size}__gradients_mini")

    # save out the computed metrics to compiled_satistics/model_size
    os.makedirs(f"compiled_statistics/{model_size}", exist_ok=True)
 
    cka_scores_per_layer_fn = f"compiled_statistics/{model_size}/cka_scores_per_layer.pkl"
    if not os.path.exists(cka_scores_per_layer_fn):
        cka_scores_per_layer = compute_cka_sores(activation_dataset)
        with open(cka_scores_per_layer_fn, "wb") as f:
            pickle.dump(cka_scores_per_layer, f)

    weight_magnitudes_per_layer_fn = f"compiled_statistics/{model_size}/weight_magnitudes_per_layer.pkl"
    if not os.path.exists(weight_magnitudes_per_layer_fn):
        weight_magnitudes_per_layer = compute_weight_magnitudes(weights_dataset)
        with open(weight_magnitudes_per_layer_fn, "wb") as f:
            pickle.dump(weight_magnitudes_per_layer, f)

    grad_weight_magnitudes_per_layer_fn = f"compiled_statistics/{model_size}/grad_weight_magnitudes_per_layer.pkl"
    if not os.path.exists(grad_weight_magnitudes_per_layer_fn):
        grad_weight_magnitudes_per_layer = compute_grad_weight_magnitudes(gradient_dataset)
        with open(grad_weight_magnitudes_per_layer_fn, "wb") as f:
            pickle.dump(grad_weight_magnitudes_per_layer, f)

    grad_sim_per_layer_fn = f"compiled_statistics/{model_size}/grad_sim_per_layer.pkl"
    if not os.path.exists(grad_sim_per_layer_fn):
        grad_sim_per_layer = compute_grad_sim_per_layer(gradient_dataset)
        with open(grad_sim_per_layer_fn, "wb") as f:
            pickle.dump(grad_sim_per_layer, f)

    
if __name__ == '__main__':
    main()