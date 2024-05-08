"""
Computed metrics from the extracted information of the model weights, activations and gradients
"""


from datasets import load_dataset 
from transformers import AutoConfig
import numpy as np
from lib import cka
import click 
import pickle
import os 
import time 
from tqdm import tqdm
from scipy.linalg import svdvals

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


def compute_cka_scores(activation_dataset, weights_dataset, model_size="70m"):
    """
    Computes the CKA scores of each model layer relative to the final layer's state after training
    """

    _model_config = AutoConfig.from_pretrained(f"EleutherAI/pythia-{model_size}-deduped")
    num_heads = _model_config.num_attention_heads

    def compute_checkpoint_ov_activation(checkpoint_activation, checkpoint_weights): 
        """
        Compute checkpoint-specific ov activations 
        """

        checkpoint_ov_activation = {}

        # for each checkpoint we 
        for layer_idx, i in enumerate(range(0, len(checkpoint_activation), 3)):
            value_activation = np.array(checkpoint_activation[i]['data'])
            output_projection = np.array(checkpoint_weights[i+1]['data']) 

            assert(checkpoint_activation[i]['layer_name'] == f"gpt_neox.layers.{layer_idx}.attention.query_key_value")
            assert(checkpoint_weights[i+1]['layer_name'] == f"gpt_neox.layers.{layer_idx}.attention.dense")

            for head_idx in range(num_heads):

                ov_activation_per_head = value_activation[:, head_idx*64:(head_idx+1)*64] @ output_projection[:, head_idx*64:(head_idx+1)*64].T
                checkpoint_ov_activation[f"gpt_neox.layers.{layer_idx}.attention.ov_activation.heads.{head_idx}"] = ov_activation_per_head

            checkpoint_ov_activation[f"gpt_neox.layers.{layer_idx}.attention.ov_activation"] = value_activation @ output_projection.T

        return checkpoint_ov_activation

    def get_mlp_activation(checkpoint_activation): 
        """
        Extracting just the MLP activations from the checkpoint activations. 
        """

        checkpoint_mlp_activation = {}
        for layer_idx, i in enumerate(range(2, len(checkpoint_activation), 3)):
            mlp_activation = np.array(checkpoint_activation[i]['data'])

            assert(checkpoint_activation[i]['layer_name'] == f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h")

            checkpoint_mlp_activation[f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h"] = mlp_activation
        
        return checkpoint_mlp_activation


    checkpoint_step_to_range_indices = get_checkpoint_step_to_range_indices(activation_dataset)

    last_checkpoint_activation = activation_dataset.select(range(*checkpoint_step_to_range_indices[checkpoint_steps[-1]]))
    last_checkpoint_weights = weights_dataset.select(range(*checkpoint_step_to_range_indices[checkpoint_steps[-1]]))

    last_checkpoint_ov_activation = compute_checkpoint_ov_activation(last_checkpoint_activation, last_checkpoint_weights)
    last_checkpoint_mlp_activation = get_mlp_activation(last_checkpoint_activation)

    cka_scores_per_activation = {layer_name: [] for layer_name in list(last_checkpoint_ov_activation.keys()) + list(last_checkpoint_mlp_activation.keys())} 

    for checkpoint_step in tqdm(checkpoint_steps):

        print("Processing checkpoint step: ", checkpoint_step)

        checkpoint_step_idx_range = checkpoint_step_to_range_indices[checkpoint_step]

        checkpoint_activations = activation_dataset.select(range(*checkpoint_step_idx_range))
        checkpoint_weights = weights_dataset.select(range(*checkpoint_step_idx_range))

        checkpoint_ov_activation = compute_checkpoint_ov_activation(checkpoint_activations, checkpoint_weights)

        for layer_name, last_checkpoint_layer_activation in last_checkpoint_ov_activation.items():
            checkpoint_layer_activation = checkpoint_ov_activation[layer_name]

            cka_score = cka.feature_space_linear_cka(
                last_checkpoint_layer_activation, 
                checkpoint_layer_activation
            )

            cka_scores_per_activation[layer_name].append(cka_score)
        
        checkpoint_mlp_activation = get_mlp_activation(checkpoint_activations)

        for layer_name, last_checkpoint_layer_activation in last_checkpoint_mlp_activation.items():
            checkpoint_layer_activation = checkpoint_mlp_activation[layer_name]

            cka_score = cka.feature_space_linear_cka(
                last_checkpoint_layer_activation, 
                checkpoint_layer_activation
            )

            cka_scores_per_activation[layer_name].append(cka_score)

    return cka_scores_per_activation

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


def compute_grad_sim(gradient_dataset):
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

def compute_svd(dataset):
    """
    For each layer, we compute the singular values.
    """

    print("Computing SVD directions per layer")

    layer_names = set(dataset['layer_name'])
    # NOTE: we are assuming we are extracting 3 projections per layer
    num_layers = len(layer_names)//3 

    svd_weight_per_layer = dict()

    checkpoint_step_to_range_indices = get_checkpoint_step_to_range_indices(dataset)

    for checkpoint_step in tqdm(checkpoint_steps):

        print("Processing checkpoint step: ", checkpoint_step)

        checkpoint_step_idx_range = checkpoint_step_to_range_indices[checkpoint_step]

        checkpoint_weights = dataset.select(range(*checkpoint_step_idx_range))

        # NOTE: Doing this because I want to be sure that we are extracting the correct matrices
        # (since the actual filter operation is slow)
        layer_name_to_indices = {}
        for idx, layer_name in enumerate(checkpoint_weights['layer_name']):
            if layer_name not in layer_name_to_indices:
                layer_name_to_indices[layer_name] = []
            layer_name_to_indices[layer_name].append(idx)
        
        # computing the OV projection 
        for layer_num in range(num_layers):

            qkv_idx = layer_name_to_indices[f"gpt_neox.layers.{layer_num}.attention.query_key_value"]
            output_idx = layer_name_to_indices[f"gpt_neox.layers.{layer_num}.attention.dense"]

            qkv_projection = np.array(checkpoint_weights.select(qkv_idx)['data'][0])
            value_projection = qkv_projection[-qkv_projection.shape[0]//3:,]
            output_projection = np.array(checkpoint_weights.select(output_idx)['data'][0])

            # these are the two matrices we are interested in
            ov_projection =  output_projection @ value_projection 
            mlp_dense = np.array(checkpoint_weights.select(layer_name_to_indices[f"gpt_neox.layers.{layer_num}.mlp.dense_4h_to_h"])['data'][0])

            layer_projection_matrices = dict()
            layer_projection_matrices[f"gpt_neox.layers.{layer_num}.attention.ov_projection"] = ov_projection   
            layer_projection_matrices[f"gpt_neox.layers.{layer_num}.mlp.dense_4h_to_h"] = mlp_dense

            for layer_name, proj_matrix in layer_projection_matrices.items():

                if layer_name not in svd_weight_per_layer:
                    svd_weight_per_layer[layer_name] = []

                S = svdvals(proj_matrix)
                svd_weight_per_layer[layer_name].append(S)

    return svd_weight_per_layer

def get_dataset(subconfig: str):
    retry_count = 0
    sleep_time = 10

    while retry_count < 5:
        try: 
            dataset = load_dataset(
                "rdiehlmartinez/pythia-training-metrics", subconfig, split='default',
                # cache_dir='/rds-d7/user/rd654/hpc-work/cache',
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
@click.argument('model_size') 
@click.option(
    '--metrics', '-m', multiple=True,
    default=[
        "cka",
        "weight_magnitudes", 
        "grad_weight_magnitudes", 
        "grad_sim",
        "svd_weight", 
        "svd_grad_weight"
    ])
def main(model_size, metrics):

    activation_dataset = None
    weights_dataset = None
    gradient_dataset = None

    # save out the computed metrics to computed_satistics/model_size
    os.makedirs(f"computed_statistics/{model_size}", exist_ok=True)
 
    cka_scores_per_layer_fn = f"computed_statistics/{model_size}/cka_scores_per_layer.pkl"
    if not os.path.exists(cka_scores_per_layer_fn) and "cka" in metrics:
        if activation_dataset is None:
            activation_dataset = get_dataset(f"{model_size}__activations")

        if weights_dataset is None: 
            weights_dataset = get_dataset(f"{model_size}__weights")

        cka_scores_per_layer = compute_cka_scores(activation_dataset, weights_dataset, model_size=model_size)
        with open(cka_scores_per_layer_fn, "wb") as f:
            pickle.dump(cka_scores_per_layer, f)

    weight_magnitudes_per_layer_fn = f"computed_statistics/{model_size}/weight_magnitudes_per_layer.pkl"
    if not os.path.exists(weight_magnitudes_per_layer_fn) and "weight_magnitudes" in metrics:
        if weights_dataset is None:
            weights_dataset = get_dataset(f"{model_size}__weights")

        weight_magnitudes_per_layer = compute_weight_magnitudes(weights_dataset)
        with open(weight_magnitudes_per_layer_fn, "wb") as f:
            pickle.dump(weight_magnitudes_per_layer, f)

    grad_weight_magnitudes_per_layer_fn = f"computed_statistics/{model_size}/grad_weight_magnitudes_per_layer.pkl"
    if not os.path.exists(grad_weight_magnitudes_per_layer_fn) and "grad_weight_magnitudes" in metrics:
        if gradient_dataset is None:
            gradient_dataset = get_dataset(f"{model_size}__gradients")

        grad_weight_magnitudes_per_layer = compute_grad_weight_magnitudes(gradient_dataset)
        with open(grad_weight_magnitudes_per_layer_fn, "wb") as f:
            pickle.dump(grad_weight_magnitudes_per_layer, f)

    grad_sim_per_layer_fn = f"computed_statistics/{model_size}/grad_sim_per_layer.pkl"
    if not os.path.exists(grad_sim_per_layer_fn) and "grad_sim" in metrics:
        if gradient_dataset is None:
            gradient_dataset = get_dataset(f"{model_size}__gradients")

        grad_sim_per_layer = compute_grad_sim(gradient_dataset)
        with open(grad_sim_per_layer_fn, "wb") as f:
            pickle.dump(grad_sim_per_layer, f)

    svd_weight_per_layer_fn = f"computed_statistics/{model_size}/svd_weight_per_layer.pkl"
    if not os.path.exists(svd_weight_per_layer_fn) and "svd_weight" in metrics:
        if weights_dataset is None:
            weights_dataset = get_dataset(f"{model_size}__weights")

        svd_weight_per_layer = compute_svd(weights_dataset)
        with open(svd_weight_per_layer_fn, "wb") as f:
            pickle.dump(svd_weight_per_layer, f) 

    
if __name__ == '__main__':
    main()