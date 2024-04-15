from datasets import load_dataset 
import numpy as np
from lib import cka
import click 
import pickle
import os 
import time 

cpu_count = os.cpu_count()

checkpoint_steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, ]
checkpoint_steps.extend([3000 + (i * 10000) for i in range(0, 15)])

def compute_cka_sores(activation_dataset):
    """
    Computes the CKA scores of each model layer relative to the final layer's state after training
    """
    last_batch = activation_dataset.filter(lambda record: record['checkpoint_step'] == 143_000, num_proc=cpu_count)
    layer_names = last_batch['layer_name']
    cka_scores_per_layer = {layer_name: [] for layer_name in layer_names}

    for checkpoint_step in checkpoint_steps:

        checkpoint_activations = activation_dataset.filter(lambda record: record['checkpoint_step'] == checkpoint_step, num_proc=cpu_count)
        for layer_activations in checkpoint_activations: 
            layer_name = layer_activations['layer_name']
            layer_activation = np.array(layer_activations['data'])

            last_batch_activation = np.array(last_batch.filter(lambda record: record['layer_name'] == layer_name, num_proc=cpu_count)['data'][0])

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

    layer_names = set(weights_dataset['layer_name'])
    weight_magnitudes_per_layer = {layer_name: [] for layer_name in layer_names}

    for checkpoint_step in checkpoint_steps:

        checkpoint_weights = weights_dataset.filter(lambda record: record['checkpoint_step'] == checkpoint_step, num_proc=cpu_count)
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
    layer_names = set(gradient_dataset['layer_name'])
    grad_weight_magnitudes_per_layer = {layer_name: [] for layer_name in layer_names}

    for checkpoint_step in checkpoint_steps:
        checkpoint_grads = gradient_dataset.filter(lambda record: record['checkpoint_step'] == checkpoint_step, num_proc=cpu_count)

        for layer_name in layer_names:
            layer_checkpoint_grads = checkpoint_grads.filter(lambda record: record['layer_name'] == layer_name, num_proc=cpu_count)
            
            avg_layer_grad = None

            for layer_checkpoint_grad in layer_checkpoint_grads:

                _grad_data = np.array(layer_checkpoint_grad['data'])
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

    layer_names = set(gradient_dataset['layer_name'])
    grad_sim_per_layer = {layer_name: [] for layer_name in layer_names}

    for checkpoint_step in checkpoint_steps:
        checkpoint_grads = gradient_dataset.filter(lambda record: record['checkpoint_step'] == checkpoint_step, num_proc=cpu_count)

        for layer_name in layer_names:
            layer_checkpoint_grads = checkpoint_grads.filter(lambda record: record['layer_name'] == layer_name, num_proc=cpu_count)
            
            avg_cosine_sim = None
            prev_grad_data = None

            for layer_checkpoint_grad in layer_checkpoint_grads:

                _grad_data = np.array(layer_checkpoint_grad['data'])
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
            )
            break
        except:
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
 
    cka_scores_per_layer = compute_cka_sores(activation_dataset)
    with open(f"compiled_statistics/{model_size}/cka_scores_per_layer.pkl", "wb") as f:
        pickle.dump(cka_scores_per_layer, f)

    weight_magnitudes_per_layer = compute_weight_magnitudes(weights_dataset)
    with open(f"compiled_statistics/{model_size}/weight_magnitudes_per_layer.pkl", "wb") as f:
        pickle.dump(weight_magnitudes_per_layer, f)

    grad_weight_magnitudes_per_layer = compute_grad_weight_magnitudes(gradient_dataset)
    with open(f"compiled_statistics/{model_size}/grad_weight_magnitudes_per_layer.pkl", "wb") as f:
        pickle.dump(grad_weight_magnitudes_per_layer, f)

    grad_sim_per_layer = compute_grad_sim_per_layer(gradient_dataset)
    with open(f"compiled_statistics/{model_size}/grad_sim_per_layer.pkl", "wb") as f:
        pickle.dump(grad_sim_per_layer, f)

    
if __name__ == '__main__':
    main()