import torch
import torchvision
import importlib
import inspect
import os
import sys

class ModelLoader:
    def __init__(self):
        # 兼容：仍然保留一个顶层注册表（可选）
        import torchvision.models as tv_models
        self.top_level = {
            name: fn for name, fn in tv_models.__dict__.items() if callable(fn)
        }
        # 允许的子模块（按需可继续加）
        self._submodules = [
            "torchvision.models",
            "torchvision.models.detection",
            "torchvision.models.segmentation",
            "torchvision.models.video",
            "torchvision.models.quantization",
            "torchvision.models.optical_flow",
        ]

    def _resolve_callable(self, model_name: str):
        """
        支持三种写法：
        - "resnet50"
        - "detection.fasterrcnn_resnet50_fpn"
        - "torchvision.models.detection.fasterrcnn_resnet50_fpn"
        """
        # 1) 纯名字，先从顶层找（resnet50 等）
        if "." not in model_name:
            if model_name in self.top_level:
                return self.top_level[model_name]
            # 顶层没找到，再到各子模块找
            for modname in self._submodules:
                mod = importlib.import_module(modname)
                if hasattr(mod, model_name) and callable(getattr(mod, model_name)):
                    return getattr(mod, model_name)
            raise ValueError(f"Unknown TorchVision model: {model_name}")

        # 2) 带点号：尝试绝对导入（以 torchvision 开头）
        if model_name.startswith("torchvision."):
            module_path, attr = model_name.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            if not hasattr(mod, attr) or not callable(getattr(mod, attr)):
                raise ValueError(f"Unknown TorchVision model: {model_name}")
            return getattr(mod, attr)

        # 3) 相对写法（以 detection./segmentation. 等开头）
        #    自动补上 "torchvision.models."
        prefix_try = f"torchvision.models.{model_name}"
        module_path, attr = prefix_try.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        if not hasattr(mod, attr) or not callable(getattr(mod, attr)):
            raise ValueError(f"Unknown TorchVision model: {model_name}")
        return getattr(mod, attr)

    def _maybe_inject_default_weights(self, ctor, ctor_kwargs: dict):
        """
        如果调用签名里有 'weights' / 'weights_backbone' 且调用方没传，
        自动填 'DEFAULT'，这样 torchvision 会自动下载缺失的权重。
        调用方若显式传 None，则尊重 None（不下载）。
        """
        sig = inspect.signature(ctor)
        params = sig.parameters

        # weights
        if "weights" in params and ("weights" not in ctor_kwargs):
            ctor_kwargs["weights"] = "DEFAULT"  # 触发自动下载

        # detection/backbone 的额外权重
        if "weights_backbone" in params and ("weights_backbone" not in ctor_kwargs):
            # 注意：若 weights 已经为 None，很多 detection 构造器也允许 backbone 用 DEFAULT
            # 这里给默认 DEFAULT（可按需要改为跟随 weights）
            ctor_kwargs["weights_backbone"] = "DEFAULT"

        return ctor_kwargs

    def from_library(self, model_name, **ctor_kwargs):
        """
        用法示例：
        - from_library("resnet50")                      -> 自动下 DEFAULT
        - from_library("detection.fasterrcnn_resnet50_fpn") -> 自动下 DEFAULT + DEFAULT backbone
        - from_library("detection.fasterrcnn_resnet50_fpn", weights=None, weights_backbone=None) -> 禁止下载
        - from_library("torchvision.models.detection.fasterrcnn_resnet50_fpn", weights="DEFAULT")
        """
        ctor = self._resolve_callable(model_name)

        # 自动填充 DEFAULT（除非你显式传了 None/具体枚举）
        ctor_kwargs = dict(ctor_kwargs or {})
        ctor_kwargs = self._maybe_inject_default_weights(ctor, ctor_kwargs)

        model = ctor(**ctor_kwargs)
        model.eval()
        return model

    def from_repo(self, repo_path, class_path, *, params=None, weights=None, weight_format="auto"):
        if not os.path.exists(repo_path):
            raise FileNotFoundError(f"Repo path not found: {repo_path}")
        params = dict(params or {})
        cfg = params.get("cfg", None)
        sys.path.insert(0, repo_path)
        try:
            module_name, class_name = class_path.rsplit(".", 1)
            mod = importlib.import_module(module_name)
            cls = getattr(mod, class_name)
            model = cls(cfg)
            if weights:
                self.load_state_dict_smart(model, weights, weight_format)
            model.eval()
            return model
        finally:
            if repo_path in sys.path:
                sys.path.remove(repo_path)


    def from_hub(self, repo, entry, **kwargs):
        model = torch.hub.load(repo, entry, **kwargs)
        if hasattr(model, "eval"):
            model.eval()
        return model

    def from_builder(self, repo_path, builder, cfg=None, cfg_file=None, cfg_options=None, weights=None):
        """
        builder: "包.文件.函数名"，例如 "mmseg.models.builder.build_segmentor"
        cfg: 直接传入的python字典（优先级高）
        cfg_file: 指向一个配置文件路径（.py 或 .json/.yaml 由项目决定）
        cfg_options: 对 cfg/cfg_file 的覆盖项（字典）
        """
        if not os.path.isdir(repo_path):
            raise FileNotFoundError(f"Repo path not found: {repo_path}")
        sys.path.insert(0, repo_path)
        try:
            module_path, fn_name = builder.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            build_fn = getattr(mod, fn_name)

            # 1) 准备 cfg
            cfg_obj = None
            if cfg is not None:
                cfg_obj = cfg
            elif cfg_file is not None:
                # 这里按项目习惯加载cfg文件；
                # mmcv里的config通常是 .py 配置，需要 mmcv/omegaconf 等工具读取。
                # 为了和各种仓库解耦，给一个最简实现：直接 import 这个 cfg_file 的 py 模块拿到 "cfg" 变量，
                # 或者你自己用该项目的 Config loader(例如 mmcv.Config.fromfile)。
                if cfg_file.endswith(".py"):
                    # 简化：动态import一个包含 cfg 变量的python文件
                    cfg_mod_name = os.path.splitext(os.path.basename(cfg_file))[0]
                    cfg_dir = os.path.dirname(os.path.abspath(cfg_file))
                    sys.path.insert(0, cfg_dir)
                    try:
                        cfg_mod = importlib.import_module(cfg_mod_name)
                        cfg_obj = getattr(cfg_mod, "cfg", None)
                        if cfg_obj is None:
                            raise ValueError("Config .py must define a top-level 'cfg' object (dict).")
                    finally:
                        if cfg_dir in sys.path:
                            sys.path.remove(cfg_dir)
                else:
                    # 如果是 .json/.yaml，可以用 json/yaml 读取
                    import json, yaml
                    with open(cfg_file, "r", encoding="utf-8") as f:
                        if cfg_file.endswith(".json"):
                            cfg_obj = json.load(f)
                        else:
                            cfg_obj = yaml.safe_load(f)

            if cfg_options:
                # 浅覆盖（如需深覆盖可写一个merge函数）
                cfg_obj = {**cfg_obj, **cfg_options} if cfg_obj else cfg_options

            if cfg_obj is None:
                raise ValueError("No cfg or cfg_file provided for builder-mode loader.")

            # 2) 构建模型
            model = build_fn(cfg_obj)
            # 3) 加权重（尽量通用）
            if weights:
                ckpt = torch.load(weights, map_location="cpu")
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    ckpt = ckpt["state_dict"]
                model.load_state_dict(ckpt, strict=False)
            model.eval()
            return model
        finally:
            if repo_path in sys.path:
                sys.path.remove(repo_path)

    def load_state_dict_smart(self, model, weights_path, weight_format="auto"):
        ckpt = torch.load(weights_path, map_location="cpu")
        # YOLOv5 的 .pt 里通常是 ckpt["model"].state_dict() 或 ckpt["state_dict"]
        if weight_format == "yolov5" or (weight_format == "auto" and isinstance(ckpt, dict) and "model" in ckpt):
            if "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
                state = ckpt["model"].state_dict()
            elif "state_dict" in ckpt:
                state = ckpt["state_dict"]
            else:
                state = ckpt
            model.load_state_dict(state, strict=False)
        else:
            # 通用：字典/含 state_dict 的字典
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            model.load_state_dict(ckpt, strict=False)
