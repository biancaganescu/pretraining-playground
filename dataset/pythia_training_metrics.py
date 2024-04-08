import datasets
import pickle


_DESCRIPTION = """\
    Dataset for storing training metrics of pythia models 
"""

class PythiaTrainingMetrics(datasets.GeneratorBasedBuilder):
    
    MODEL_SIZES = [ 
        "70m", 
        "160m", 
        "410m",
        "1b",
        "1.4b",
        "2.8b",
        "6.9b"
    ]

    _GRADIENTS_DESCRIPTION = """\
        Dataset for storing gradients of pythia models 
    """

    _WEIGHTS_DESCRIPTION = """\
        Dataset for storing weights of pythia models 
    """

    _WEIGHTS_MINI_DESCRIPTION = """\
        Dataset for storing weights of pythia models (minimizes the amount of gradients per 
        checkpoint to only 2)
    """

    _ACTIVATIONS_DESCRIPTION = """\
        Dataset for storing activations of pythia models 
    """
   
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="gradients",
            description=_WEIGHTS_DESCRIPTION,
            version="1.0.0",
        ),
        datasets.BuilderConfig(
            name="gradients_mini",
            description=_WEIGHTS_MINI_DESCRIPTION,
            version="1.0.0",
        ),
        datasets.BuilderConfig(
            name="activations ",
            description=_ACTIVATIONS_DESCRIPTION,
            version="1.0.0",
        ),
        datasets.BuilderConfig(
            name="weights",
            description=_WEIGHTS_DESCRIPTION,
            version="1.0.0",
        ),
        datasets.BuilderConfig(
            name="all",
            description="All the metrics",
            version="1.0.0",
        )
   ]

    def _info(self):
        """
        TODO: Got to figure out how to represent the features etc. 

        how do we do this if each feature is dependent on the model size?
        """

        features_dict = { 
            "checkpoint_step": datasets.Value('int32'),
            "layer_name": datasets.Value('string'),
        }

        if self.config.name in ["activations", "weights"]:
            features_dict['data'] = datasets.Sequence(datasets.Value('float32'))
        elif self.config_name in ["gradients", "gradients_mini"]:
            features_dict['gradient_step'] = datasets.Value('int32')
            features_dict['gradient'] = datasets.Sequence(datasets.Value('float32'))
            
        features = datasets.Features(features_dict)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,  
        )


    def _split_generators(self, dl_manager: datasets.DownloadManager):
        """ 
        Returns data for different splits - we define a split as a model size. 
        """

        model_size_to_fp = { model_size: [] for model_size in self.MODEL_SIZES }

        checkpoint_steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, ]
        checkpoint_steps.extend([3000 + (i * 10000) for i in range(0, 15)])

        def get_gradient_step(step: int): 
            """
            Return a list of the gradient steps that are used at a given checkpoint step. 
            """
            return list(range(max(0, step-5), min(step+6, 143_000))) 

        for model_size in self.MODEL_SIZES:
            for checkpoint_step in checkpoint_steps:

                directory_path = f"./models/{model_size}/checkpoint_{checkpoint_step}"

                if self.config.name == "activations": 
                    model_size_to_fp[model_size].append(f"{directory_path}/checkpoint_activations.pickle")
                elif self.config_name == "weights":
                    model_size_to_fp[model_size].append(f"{directory_path}/checkpoint_weights.pickle")
                elif self.config_name == "gradients":
                    for gradient_step in get_gradient_step(checkpoint_step):
                        model_size_to_fp[model_size].append(f"{directory_path}/checkpoint_gradients_{gradient_step}.pickle")
                elif self.config_name == "gradients_mini":
                    for gradient_step in get_gradient_step(checkpoint_step)[:2]:
                        model_size_to_fp[model_size].append(f"{directory_path}/checkpoint_gradients_mini_{gradient_step}.pickle")

        downloaded_files = dl_manager.download_and_extract(model_size_to_fp)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": downloaded_fps
                }
            )  for downloaded_fps in downloaded_files.values()
        ]

    def _generate_examples(self, filepaths):

        # the filepaths should be a list of filepaths 
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        
        global_idx = 0 # the unique identifier for the example 

        for filepath in filepaths:
            with open(filepath, encoding="utf-8") as f:
                data = pickle.load(f)

                # extract checkpoint step from the filepath
                checkpoint_step = int(filepath.split("/")[1].split("_")[-1])
                
                if self.config.name in ["activations", "weights"]:
                    for layer_name, layer_data in data.items():
                        for data in layer_data: 
                            yield global_idx, {"checkpoint_step": checkpoint_step, "layer_name": layer_name, "data": data}
                            global_idx += 1
                elif self.config.name in ["gradients", "gradients_mini"]:
                    for layer_name, layer_data in data.items():
                        for gradient_step, gradient in layer_data.items():
                            yield global_idx, {"checkpoint_step": checkpoint_step, "layer_name": layer_name, "gradient_step": gradient_step, "gradient": gradient}
                            global_idx += 1
