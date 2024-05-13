""" 
Helper utility for the analysis of the evaluation results.
"""

import json
import os 
import pickle

# Basic constants for evaluation  
ALL_EVAL_METRICS = ['arc_challenge', 'arc_easy', 'lambada_openai', 'piqa', 'winogrande', 'wsc', 'sciq', 'logiqa',]
LIMITED_EVAL_METRICS = ['arc_easy', 'lambada_openai', ] 

MODEL_SIZES = ["70m", "160m", "410m", "1.4b", "2.8b",] 
METRICS = ['cka_scores', 'grad_sim', 'grad_weight_magnitudes', 'weight_magnitudes']

CHECKPOINT_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, ]
CHECKPOINT_STEPS.extend([3000 + (i * 10000) for i in range(0, 15)])


def get_checkpoint_evals(model_size, eval_metrics=LIMITED_EVAL_METRICS):
    """ 
    Get the evaluation results for each checkpoint for a given model size. 
    """ 

    eval_folder = f'/home/rd654/pretraining-playground/lib/pythia/evals/pythia-v1/pythia-{model_size}-deduped/zero-shot'

    # list all files in the folder 
    files = os.listdir(eval_folder)

    checkpoint_evals = {}

    for file in files:

        assert(".json" in file), "Evaluation file should be a json"

        # extract step number from file 
        step_number = file.split("_step")[-1].split(".json")[0]

        # read in json file
        with open(os.path.join(eval_folder, file), 'r') as f:
            data = json.load(f)
            results = data['results']

            for eval_metric in eval_metrics:
                eval_metric_result = results[eval_metric]

                if eval_metric not in checkpoint_evals:
                    checkpoint_evals[eval_metric] = []

                checkpoint_evals[eval_metric].append((eval_metric_result['acc'], step_number))
    
    # order the results by step number
    for eval_metric in eval_metrics:
        checkpoint_evals[eval_metric] = [x[0] for x in sorted(checkpoint_evals[eval_metric], key=lambda x: int(x[1]))]

    return checkpoint_evals


def sort_and_filter_metrics(metrics, filter_layer_name="attention.dense", remove_heads=False):
    """
    Given a dictionary of metrics, filters out the metrics by the filter_layer_name key and 
    possibly removed heads even if they share the 
    """
    metrics = {key: val for key, val in metrics.items() if filter_layer_name in key} 
    if remove_heads: 
        metrics = {key: val for key, val in metrics.items() if "heads" not in key}
    metrics = {key: val for key, val in sorted(metrics.items(), key=lambda x: int(x[0].split("layers.")[-1].split('.')[0]))}
    return metrics


def basic_data_sanity_check():
    """
    Simple sanity checks to verify the integrity of computed_metrics
    """
    for metric_name in METRICS: 
        for model_size in MODEL_SIZES: 
            # we want to compute the average metric for each of the model sizes and plot out 
            # the average metric as a function of the number of training steps
            try: 
                with open(f'/home/rd654/pretraining-playground/computed_statistics/{model_size}/{metric_name}_per_layer.pkl', 'rb') as f:
                    _metrics = pickle.load(f)
                
                for key, value in _metrics.items(): 
                    if value is None or len(value) == 0:
                        print(f"No data for  -- Model Size: {model_size} - Metric: {metric_name} - Layer Name: {key}")
            except:
                print(f"Could not open file for -- Model Size: {model_size} - Metric: {metric_name}")
