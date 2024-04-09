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
        "1.4b",
        "2.8b",
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
  ]

    def _info(self):
        """
        NOTE: we might want to specify features, but since the featuers are different for each
        model size it's annoying and kind of pointless since hf does it automatically 
        """

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
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
                elif self.config.name == "weights":
                    model_size_to_fp[model_size].append(f"{directory_path}/checkpoint_weights.pickle")
                elif self.config.name == "gradients":
                    for gradient_step in get_gradient_step(checkpoint_step):
                        model_size_to_fp[model_size].append(f"{directory_path}/checkpoint_gradients_{gradient_step}.pickle")
                elif self.config.name == "gradients_mini":
                    for gradient_step in get_gradient_step(checkpoint_step)[:2]:
                        model_size_to_fp[model_size].append(f"{directory_path}/checkpoint_gradients_mini_{gradient_step}.pickle")
                else: 
                    raise Exception("Invalid config name")

        downloaded_files = dl_manager.download_and_extract(model_size_to_fp)

        return [
            datasets.SplitGenerator(
                name=model_size_name,
                gen_kwargs={
                    "filepaths": downloaded_fps
                }
            )  for model_size_name, downloaded_fps in downloaded_files.items()
        ]

    def _generate_examples(self, filepaths):

        # the filepaths should be a list of filepaths 
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        global_idx = 0 # the unique identifier for the example 

        for filepath in filepaths:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

                # extract checkpoint step from the filepath
                checkpoint_step = int(filepath.split("/")[-2].split("_")[-1])
                
                if self.config.name in ["activations", "weights"]:
                    for layer_name, layer_data in data.items():
                        yield global_idx, {"checkpoint_step": checkpoint_step, "layer_name": layer_name, "data": layer_data}
                        global_idx += 1
                elif self.config.name in ["gradients", "gradients_mini"]:
                    gradient_step = int(filepath.split('/')[-1].split("_")[-1].split(".")[0])
                    for layer_name, layer_data in data.items():
                        yield global_idx, {"checkpoint_step": checkpoint_step, "layer_name": layer_name, "gradient_step": gradient_step, "data": layer_data}
                        global_idx += 1
