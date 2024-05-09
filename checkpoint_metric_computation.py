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


#### --- HELPER FUNCTIONS BEGIN --- #### 

def setup_metric_dictionary(model_config):
    """
    For a given metric that will be computed, set up the dictionary of layer names to store 
    the computed metric values over different checkpoints.  
    """

    layer_templates = [ 
        "gpt_neox.layers.{layer_idx}.attention.ov_activation",
        "gpt_neox.layers.{layer_idx}.attention.dense",
    ]

    metric_dictionary = {}

    for layer_idx in range(model_config.num_hidden_layers):
        for template in layer_templates:
            metric_dictionary[template.format(layer_idx=layer_idx)] = []
        
        for head_idx in range(model_config.num_attention_heads):
            metric_dictionary["gpt_neox.layers.{layer_idx}.attention.ov_circuit.heads.{head_idx}".format(layer_idx=layer_idx, head_idx=head_idx)] = []

    return metric_dictionary


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


def get_layer_name_to_checkpoint_idx(checkpoint_grads):
    """
    Create dictionary to lookup checkpoint indices of the same layer in a dictionary. 

    Only used by the gradient metric computation functions --> only for these do we compute 
    multiple gradients for the same checkpoint step, so there will be multiple of the same layer 
    when computing the given gradient-based metrics.

    The information we want is e.g.: 
    "layer.A" -> indices (0, 15) in checkpoint_grad

    """
    layer_name_to_indices = {}
    for idx, layer_name in enumerate(checkpoint_grads['layer_name']):
        if layer_name not in layer_name_to_indices:
            layer_name_to_indices[layer_name] = []
        layer_name_to_indices[layer_name].append(idx)
    return layer_name_to_indices

#### --- HELPER FUNCTIONS END --- #### 

#### --- 
#    COMPUTING CKA SCORES
#### ---

def compute_cka_scores(activation_dataset, weights_dataset, model_config):
    """
    Computes the CKA scores of each model layer relative to the final layer's state after training

    As part of the CKA scores, computes the ov activations per head.
    """

    num_heads = model_config.num_attention_heads
    attention_head_dim = model_config.hidden_size // model_config.num_attention_heads

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

                ov_activation_per_head = value_activation[:, head_idx*attention_head_dim:(head_idx+1)*attention_head_dim] @ output_projection[:, head_idx*attention_head_dim:(head_idx+1)*attention_head_dim].T
                checkpoint_ov_activation[f"gpt_neox.layers.{layer_idx}.attention.ov_circuit.heads.{head_idx}"] = ov_activation_per_head

            checkpoint_ov_activation[f"gpt_neox.layers.{layer_idx}.attention.ov_circuit"] = value_activation @ output_projection.T

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

    cka_scores_per_activation = setup_metric_dictionary(model_config)

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


#### --- 
#    COMPUTING WEIGHT MAGNITUDES
#### ---

def compute_weight_magnitudes(weights_dataset, model_config):
    """
    Computes the magnitude of the weights in each layer of the model at the different timesteps.
    For the OV activations, we compute the weight magnitudes of the OV activations per head.
    """

    num_heads = model_config.num_attention_heads
    attention_head_dim = model_config.hidden_size // model_config.num_attention_heads

    def compute_checkpoint_ov_weight_mag(checkpoint_weights): 
        """
        Compute checkpoint-specific ov weight magnitudes
        """

        checkpoint_ov_weight_mag = {}

        # for each checkpoint we 
        for layer_idx, i in enumerate(range(0, len(checkpoint_weights), 3)):
            qkv_projection = np.array(checkpoint_weights[i]['data']) 
            value_projection = qkv_projection[-qkv_projection.shape[0]//3:,]
            output_projection = np.array(checkpoint_weights[i+1]['data'])

            assert(checkpoint_weights[i]['layer_name'] == f"gpt_neox.layers.{layer_idx}.attention.query_key_value")
            assert(checkpoint_weights[i+1]['layer_name'] == f"gpt_neox.layers.{layer_idx}.attention.dense")

            for head_idx in range(num_heads):

                head_output_projection = output_projection[..., head_idx*attention_head_dim:(head_idx+1)*attention_head_dim]
                head_value_projection = value_projection[head_idx*attention_head_dim:(head_idx+1)*attention_head_dim, ...]
                head_ov_projection = head_output_projection @ head_value_projection
                checkpoint_ov_weight_mag[f"gpt_neox.layers.{layer_idx}.attention.ov_circuit.heads.{head_idx}"] = np.linalg.norm(head_ov_projection)

            checkpoint_ov_weight_mag[f"gpt_neox.layers.{layer_idx}.attention.ov_circuit.heads.{head_idx}"] = np.linalg.norm(output_projection @ value_projection)
            
        return checkpoint_ov_weight_mag

    def compute_checkpoint_mlp_weight_mag(checkpoint_weights): 
        """
        Extracting just the MLP activations from the checkpoint activations. 
        """

        checkpoint_mlp_weight_mag = {}

        for layer_idx, i in enumerate(range(2, len(checkpoint_weights), 3)):
            mlp_activation = np.array(checkpoint_weights[i]['data'])

            assert(checkpoint_weights[i]['layer_name'] == f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h")

            checkpoint_mlp_weight_mag[f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h"] = mlp_activation
        
        return checkpoint_mlp_weight_mag
 

    print("Computing weight magnitudes")

    weight_magnitudes_per_layer = setup_metric_dictionary(model_config)

    checkpoint_step_to_range_indices = get_checkpoint_step_to_range_indices(weights_dataset)

    for checkpoint_step in tqdm(checkpoint_steps):

        print("Processing checkpoint step: ", checkpoint_step)
        checkpoint_step_idx_range = checkpoint_step_to_range_indices[checkpoint_step]
        checkpoint_weights = weights_dataset.select(range(*checkpoint_step_idx_range))
        
        ov_weight_mag = compute_checkpoint_ov_weight_mag(checkpoint_weights)
        mlp_weight_mag = compute_checkpoint_mlp_weight_mag(checkpoint_weights)

        for layer_name, ov_weight_mag in ov_weight_mag.items():
            weight_magnitudes_per_layer[layer_name].append(ov_weight_mag)

        for layer_name, mlp_weight_mag in mlp_weight_mag.items():
            weight_magnitudes_per_layer[layer_name].append(mlp_weight_mag)

    return weight_magnitudes_per_layer
    

#### --- 
#    COMPUTING GRAD WEIGHT MAGNITUDES
#### ---

def compute_grad_weight_magnitudes(gradient_dataset, weights_dataset, model_config):
    """
    Computes the magnitude of the gradients in each layer of the model at the different timesteps.
    For the OV activations, we compute the weight magnitudes of the OV activations per head.
    """
    num_heads = model_config.num_attention_heads
    attention_head_dim = model_config.hidden_size // model_config.num_attention_heads 

    def compute_checkpoint_ov_grad_weight_mag(checkpoint_grads, checkpoint_weights): 
        """
        Compute checkpoint-specific ov grad weight magnitudes; including per-head grad weight 
        magnitudes.
        """

        checkpoint_ov_weight_mag = {}

        # NOTE: if we call the function below with weights_dataset we should get the same results
        grad_layer_name_to_indices = get_layer_name_to_checkpoint_idx(checkpoint_grads)
        weight_layer_name_to_indices = get_layer_name_to_checkpoint_idx(checkpoint_weights)

        for layer_idx in range(model_config.num_hidden_layers):
            qkv_projection_name =  f"gpt_neox.layers.{layer_idx}.attention.query_key_value"
            output_projection_name = f"gpt_neox.layers.{layer_idx}.attention.dense"

            qkv_grad_indices = grad_layer_name_to_indices[qkv_projection_name]
            output_grad_indices = grad_layer_name_to_indices[output_projection_name]

            # NOTE: we only have 1 weight matrix, so we can just use the first index
            qkv_projection_idx = weight_layer_name_to_indices[qkv_projection_name][0]
            output_projection_idx = weight_layer_name_to_indices[output_projection_name][0]

            qkv_projection = np.array(checkpoint_weights[qkv_projection_idx]['data']) 
            value_projection = qkv_projection[-qkv_projection.shape[0]//3:,]
            output_projection = np.array(checkpoint_weights[output_projection_idx]['data'])

            assert(checkpoint_weights[qkv_projection_idx]['layer_name'] == f"gpt_neox.layers.{layer_idx}.attention.query_key_value")
            assert(checkpoint_weights[output_projection_idx]['layer_name'] == f"gpt_neox.layers.{layer_idx}.attention.dense")

            avg_layer_grad = {}

            for qkv_idx, output_idx in zip(qkv_grad_indices, output_grad_indices):

                qkv_grad = np.array(checkpoint_grads[qkv_idx]['data']) 
                value_grad = qkv_grad[-qkv_grad.shape[0]//3:,]
                output_grad = np.array(checkpoint_grads[output_idx]['data'])

                assert(checkpoint_grads[qkv_idx]['layer_name'] == f"gpt_neox.layers.{layer_idx}.attention.query_key_value")
                assert(checkpoint_grads[output_idx]['layer_name'] == f"gpt_neox.layers.{layer_idx}.attention.dense")

                for head_idx in range(num_heads):

                    head_output_grad = output_grad[..., head_idx*attention_head_dim:(head_idx+1)*attention_head_dim]
                    head_value_grad = value_grad[head_idx*attention_head_dim:(head_idx+1)*attention_head_dim, ...]

                    head_output_projection = output_projection[..., head_idx*attention_head_dim:(head_idx+1)*attention_head_dim]
                    head_value_projection = value_projection[head_idx*attention_head_dim:(head_idx+1)*attention_head_dim, ...]

                    head_ov_grad = head_output_grad @ head_value_projection + head_output_projection @ head_value_grad

                    _head_norm = np.linalg.norm(head_ov_grad)

                    head_ov_key_name = f"gpt_neox.layers.{layer_idx}.attention.ov_circuit.heads.{head_idx}"

                    if head_ov_key_name not in avg_layer_grad:
                        avg_layer_grad[head_ov_key_name] = _head_norm
                    else: 
                        avg_layer_grad[head_ov_key_name] += _head_norm
                
                # computing the OV weight magnitude
                ov_weight_mag = np.linalg.norm(output_grad @ value_projection + output_projection @ value_grad)
                    
                if f"gpt_neox.layers.{layer_idx}.attention.ov_circuit" not in avg_layer_grad:
                    avg_layer_grad[f"gpt_neox.layers.{layer_idx}.attention.ov_circuit"] = ov_weight_mag
                else:
                    avg_layer_grad[f"gpt_neox.layers.{layer_idx}.attention.ov_circuit"] += ov_weight_mag
                
            for layer_name, avg_grad in avg_layer_grad.items():
                checkpoint_ov_weight_mag[layer_name].append(avg_grad/len(qkv_grad_indices))
        
        return checkpoint_ov_weight_mag


    def compute_checkpoint_mlp_grad_weight_mag(checkpoint_grads): 
        """
        Compute checkpoint-specific mlp grad weight magnitudes. 
        """

        checkpoint_mlp_weight_mag = {}

        layer_name_to_indices = get_layer_name_to_checkpoint_idx(checkpoint_grads)

        for layer_idx in range(model_config.num_hidden_layers):

            mlp_projection_name = f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h"

            mlp_indices = layer_name_to_indices[mlp_projection_name]

            avg_mlp_grad = None

            for mlp_idx in mlp_indices:
                mlp_projection = np.array(checkpoint_weights[mlp_idx]['data']) 

                ov_weight_mag = np.linalg.norm(mlp_projection)

                if avg_mlp_grad is None:
                    avg_mlp_grad = ov_weight_mag
                else:
                    avg_mlp_grad += ov_weight_mag
                
            checkpoint_mlp_weight_mag[layer_name].append(avg_mlp_grad/len(mlp_indices))
        
        return checkpoint_mlp_weight_mag


    print("Computing grad weight magnitudes")

    grad_weight_magnitudes_per_layer = setup_metric_dictionary(model_config)

    grad_checkpoint_step_to_range_indices = get_checkpoint_step_to_range_indices(gradient_dataset)
    weight_checkpoint_step_to_range_indices = get_checkpoint_step_to_range_indices(weights_dataset)

    for checkpoint_step in tqdm(checkpoint_steps):

        print("Processing checkpoint step: ", checkpoint_step)
        grad_checkpoint_step_idx_range = grad_checkpoint_step_to_range_indices[checkpoint_step]
        checkpoint_grads = gradient_dataset.select(range(*grad_checkpoint_step_idx_range))

        weight_checkpoint_step_to_range_indices = weight_checkpoint_step_to_range_indices[checkpoint_step]
        checkpoint_weights = weights_dataset.select(range(*weight_checkpoint_step_to_range_indices))
        
        ov_weight_mag = compute_checkpoint_ov_grad_weight_mag(checkpoint_grads, checkpoint_weights)
        mlp_weight_mag = compute_checkpoint_mlp_grad_weight_mag(checkpoint_grads)

        for layer_name, ov_weight_mag in ov_weight_mag.items():
            grad_weight_magnitudes_per_layer[layer_name].append(ov_weight_mag)

        for layer_name, mlp_weight_mag in mlp_weight_mag.items():
            grad_weight_magnitudes_per_layer[layer_name].append(mlp_weight_mag)

    return grad_weight_magnitudes_per_layer

#### --- 
#    COMPUTING GRAD COSINE SIM 
#### ---

def compute_grad_sim(gradient_dataset, model_config):
    """
    Computes the cosine similarity of the gradients in each layer of the model at the different timesteps.
    """

    num_heads = model_config.num_attention_heads
    attention_head_dim = model_config.hidden_size // model_config.num_attention_heads 

    def compute_checkpoint_ov_grad_weight_mag(checkpoint_grads): 
        """
        Compute checkpoint-specific ov weight magnitudes
        """

        checkpoint_ov_weight_mag = {}

        layer_name_to_indices = get_layer_name_to_checkpoint_idx(checkpoint_grads)

        for layer_idx in range(model_config.num_hidden_layers):
            qkv_projection_name =  f"gpt_neox.layers.{layer_idx}.attention.query_key_value"
            output_projection_name = f"gpt_neox.layers.{layer_idx}.attention.dense"

            qkv_indices = layer_name_to_indices[qkv_projection_name]
            output_indices = layer_name_to_indices[output_projection_name]

            avg_layer_grad = {}

            for qkv_idx, output_idx in zip(qkv_indices, output_indices):
                qkv_projection = np.array(checkpoint_weights[qkv_idx]['data']) 
                value_projection = qkv_projection[-qkv_projection.shape[0]//3:,]
                output_projection = np.array(checkpoint_weights[output_idx]['data'])

                assert(checkpoint_weights[qkv_idx]['layer_name'] == f"gpt_neox.layers.{layer_idx}.attention.query_key_value")
                assert(checkpoint_weights[output_idx]['layer_name'] == f"gpt_neox.layers.{layer_idx}.attention.dense")

                for head_idx in range(num_heads):

                    head_output_projection = output_projection[..., head_idx*attention_head_dim:(head_idx+1)*attention_head_dim]
                    head_value_projection = value_projection[head_idx*attention_head_dim:(head_idx+1)*attention_head_dim, ...]
                    head_ov_projection = head_output_projection @ head_value_projection

                    _head_norm = np.linalg.norm(head_ov_projection)

                    head_ov_key_name = f"gpt_neox.layers.{layer_idx}.attention.ov_circuit.heads.{head_idx}"

                    if head_ov_key_name not in avg_layer_grad:
                        avg_layer_grad[head_ov_key_name] = _head_norm
                    else: 
                        avg_layer_grad[head_ov_key_name] += _head_norm
                
                # computing the OV weight magnitude
                ov_weight_mag = np.linalg.norm(value_projection @ output_projection.T)

                if f"gpt_neox.layers.{layer_idx}.attention.ov_circuit" not in avg_layer_grad:
                    avg_layer_grad[f"gpt_neox.layers.{layer_idx}.attention.ov_circuit"] = ov_weight_mag
                else:
                    avg_layer_grad[f"gpt_neox.layers.{layer_idx}.attention.ov_circuit"] += ov_weight_mag
                
            for layer_name, avg_grad in avg_layer_grad.items():
                checkpoint_ov_weight_mag[layer_name].append(avg_grad/len(qkv_indices))
        
        return checkpoint_ov_weight_mag


    def compute_checkpoint_mlp_grad_weight_mag(checkpoint_grads): 
        """
        Extracting just the MLP activations from the checkpoint activations. 
        """

        checkpoint_mlp_weight_mag = {}

        layer_name_to_indices = get_layer_name_to_checkpoint_idx(checkpoint_grads)

        for layer_idx in range(model_config.num_hidden_layers):

            mlp_projection_name = f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h"

            mlp_indices = layer_name_to_indices[mlp_projection_name]

            avg_mlp_grad = None

            for mlp_idx in mlp_indices:
                mlp_projection = np.array(checkpoint_weights[mlp_idx]['data']) 

                ov_weight_mag = np.linalg.norm(mlp_projection)

                if avg_mlp_grad is None:
                    avg_mlp_grad = ov_weight_mag
                else:
                    avg_mlp_grad += ov_weight_mag
                
            checkpoint_mlp_weight_mag[layer_name].append(avg_mlp_grad/len(mlp_indices))
        
        return checkpoint_mlp_weight_mag


    print("Computing grad weight magnitudes")

    grad_weight_magnitudes_per_layer = setup_metric_dictionary(model_config)

    checkpoint_step_to_range_indices = get_checkpoint_step_to_range_indices(gradient_dataset)

    for checkpoint_step in tqdm(checkpoint_steps):

        print("Processing checkpoint step: ", checkpoint_step)
        checkpoint_step_idx_range = checkpoint_step_to_range_indices[checkpoint_step]
        checkpoint_weights = gradient_dataset.select(range(*checkpoint_step_idx_range))
        
        ov_weight_mag = compute_checkpoint_ov_grad_weight_mag(checkpoint_weights)
        mlp_weight_mag = compute_checkpoint_mlp_grad_weight_mag(checkpoint_weights)

        for layer_name, ov_weight_mag in ov_weight_mag.items():
            grad_weight_magnitudes_per_layer[layer_name].append(ov_weight_mag)

        for layer_name, mlp_weight_mag in mlp_weight_mag.items():
            grad_weight_magnitudes_per_layer[layer_name].append(mlp_weight_mag)

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

    model_config = AutoConfig.from_pretrained(f"EleutherAI/pythia-{model_size}-deduped")
 
    cka_scores_per_layer_fn = f"computed_statistics/{model_size}/cka_scores_per_layer.pkl"
    if not os.path.exists(cka_scores_per_layer_fn) and "cka" in metrics:
        if activation_dataset is None:
            activation_dataset = get_dataset(f"{model_size}__activations")

        if weights_dataset is None: 
            weights_dataset = get_dataset(f"{model_size}__weights")

        cka_scores_per_layer = compute_cka_scores(activation_dataset, weights_dataset, model_config)
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