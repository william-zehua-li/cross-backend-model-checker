import torch
import torchvision.models as tv_models
import importlib
import os

class ModelLoader:
    def __init__(self):
        self.tv_models = {name: fn for name, fn in tv_models.__dict__.items() if callable(fn)}

    def from_library(self, model_name):
        if model_name not in self.tv_models:
            raise ValueError(f"Unknown TorchVision model: {model_name}")
        model = self.tv_models[model_name](weights="DEFAULT")
        model.eval()
        return model

    def from_repo(self, repo_path, class_path, **kwargs):
        if not os.path.exists(repo_path):
            raise FileNotFoundError(f"Repo path not found: {repo_path}")
        import sys
        sys.path.append(repo_path)
        module_name, class_name = class_path.rsplit(".", 1)
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        model = cls(**kwargs)
        model.eval()
        return model
