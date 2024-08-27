# Pretraining Playground 

A repository for analyzing the learning trajectories of pythia language models. The suite of models analyzed in this project ranges from 70M-7B parameter models. These models are all [open sourced](https://github.com/EleutherAI/pythia) by the Eleuther AI folks.

The main logic for extracting learning dynamic information about the models is in `checkpoint_metric_extractor.py` (which extracts the learning metrics that we analyze from the model checkpoints) and `checkpoint_metric_computation.py` (which applies additional computation on top of the 'raw' metrics that are extracted in the first script, e.g. compute CKA scores). 

The result of extracting these metrics is uploaded as a 'meta'-dataset (i.e. a dataset of learning dynamic metric) to [HF/pretraining-playground](https://huggingface.co/pretraining-playground). You might need to be added to this org - if so reach out to me. 

Finally, we run a set of different analyses in jupyter notebooks that are stored under the `notebooks/...` folder. 
