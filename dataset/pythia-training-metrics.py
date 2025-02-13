import datasets
import pickle

_DESCRIPTION = """\
    Dataset for storing training metrics of pythia models 
"""
LAST_STEP = 4091

class PythiaTrainingMetrics(datasets.GeneratorBasedBuilder):
    

    MODEL_SIZES = [ 
        "14m"
    ]

    _GRADIENTS_DESCRIPTION = """\
        Dataset for storing gradients of pythia models of the requested model size
    """

    _WEIGHTS_DESCRIPTION = """\
        Dataset for storing weights of pythia models  of the requested model size
    """

    _WEIGHTS_MINI_DESCRIPTION = """\
        Dataset for storing weights of pythia models (minimizes the amount of gradients per 
        checkpoint to only 2) of the requested model size
    """

    _ACTIVATIONS_DESCRIPTION = """\
        Dataset for storing activations of pythia models of the requested model size
    """
   
    BUILDER_CONFIGS = []
    for model_size in MODEL_SIZES:
        BUILDER_CONFIGS.extend([
            datasets.BuilderConfig(
                name=f"{model_size}__gradients",
                description=_WEIGHTS_DESCRIPTION,
                version="1.0.0",
            ),
            datasets.BuilderConfig(
                name=f"{model_size}__gradients_mini",
                description=_WEIGHTS_MINI_DESCRIPTION,
                version="1.0.0",
            ),
            datasets.BuilderConfig(
                name=f"{model_size}__activations",
                description=_ACTIVATIONS_DESCRIPTION,
                version="1.0.0",
            ),
            datasets.BuilderConfig(
                name=f"{model_size}__weights",
                description=_WEIGHTS_DESCRIPTION,
                version="1.0.0",
            ),
        ])

    def _info(self):
        """
        NOTE: we might want to specify features, but since the features are different for each
        model size it's annoying and kind of pointless since hf does it automatically 
        """

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
        )


    def _split_generators(self, dl_manager: datasets.DownloadManager):
        global LAST_STEP
        """ 
        Returns data for different splits - we define a split as a model size. 
        """

        to_download_files = []

        kwargs_checkpoint_steps = []
        kwargs_gradient_steps = [] 

        checkpoint_steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000,]
        checkpoint_steps.extend([1000 * i for i in range(2, (LAST_STEP // 1000) + 1)])
        checkpoint_steps.extend([LAST_STEP])

        def get_gradient_step(step: int): 
            """
            Return a list of the gradient steps that are used at a given checkpoint step. 
            """
            return list(range(max(0, step-5), min(step+6, LAST_STEP))) 

        model_size = self.config.name.split("__")[0]

        for checkpoint_step in checkpoint_steps:

            directory_path = f"./models/{model_size}/checkpoint_{checkpoint_step}"

            if "activations" in self.config.name:
                to_download_files.append(f"{directory_path}/checkpoint_activations.pickle")
                kwargs_checkpoint_steps.append(checkpoint_step)
            elif "weights" in self.config.name:
                to_download_files.append(f"{directory_path}/checkpoint_weights.pickle")
                kwargs_checkpoint_steps.append(checkpoint_step)
            elif "gradients" in self.config.name:
                gradient_steps = get_gradient_step(checkpoint_step)
                if "mini" in self.config.name:
                    gradient_steps = gradient_steps[:2]
                for gradient_step in gradient_steps:
                    to_download_files.append(f"{directory_path}/checkpoint_gradients_{gradient_step}.pickle")
                    kwargs_checkpoint_steps.append(checkpoint_step)
                    kwargs_gradient_steps.append(gradient_step)
            else: 
                raise Exception("Invalid config name")

            downloaded_files = dl_manager.download_and_extract(to_download_files)

        return [
            datasets.SplitGenerator(
                name='default',
                gen_kwargs={
                    "filepaths": downloaded_files,
                    "checkpoint_steps": kwargs_checkpoint_steps,
                    **({"gradient_steps": kwargs_gradient_steps} if "gradients" in self.config.name else {}),
                }
            )  
        ]

    def _generate_examples(self, filepaths, checkpoint_steps, **kwargs):

        # the filepaths should be a list of filepaths 
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        if "gradients" in self.config.name:
            gradient_steps = kwargs["gradient_steps"]

        global_idx = 0 # the unique identifier for the example 

        for idx, filepath in enumerate(filepaths):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

                for layer_name, layer_data in data.items():
                    record = {
                        "checkpoint_step": checkpoint_steps[idx],
                        "layer_name": layer_name,
                        "data": layer_data, 
                    }
                    if "gradients" in self.config.name:
                        record['gradient_step'] = gradient_steps[idx]
               
                    yield global_idx,  record
                    global_idx += 1
