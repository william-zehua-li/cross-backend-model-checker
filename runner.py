import yaml
import torch
import json
import os
from tqdm import tqdm
from datetime import datetime
from model_loader import ModelLoader
from comparators import compare_outputs
from utils import set_global_seed, Timer

def run_once(model, device, inputs, use_compile=False):
    model = model.to(device)
    if use_compile:
        model = torch.compile(model)
    with torch.no_grad():
        outputs = model(inputs.to(device))
    return outputs

def run_pair(model, inputs, backend_pair, atol, rtol):
    (dev1, compile1), (dev2, compile2) = backend_pair
    out1 = run_once(model, dev1, inputs, compile1)
    out2 = run_once(model, dev2, inputs, compile2)
    passed, info = compare_outputs(out1, out2, atol, rtol)
    return passed, info

def main():
    set_global_seed(5)
    loader = ModelLoader()
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"results_{datetime.now().date()}.jsonl")

    for cfg_file in os.listdir("yaml_configs"):
        with open(os.path.join("yaml_configs", cfg_file), "r") as f:
            cfg = yaml.safe_load(f)

        if cfg["from"] == "library":
            model = loader.from_library(cfg["model"])
        elif cfg["from"] == "repo":
            model = loader.from_repo(cfg["repo"]["path"], cfg["repo"]["class"], **cfg.get("params", {}))
        else:
            raise ValueError("Unknown model source")

        # 准备输入
        from PIL import Image
        import torchvision.transforms as T
        preprocess = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=cfg["means"], std=cfg["stds"])
        ])
        inputs = [preprocess(Image.open(p).convert("RGB")) for p in cfg["inputs"]]
        inputs = torch.stack(inputs)

        # 后端组合
        backend_pairs = [
            (("cpu", False), ("cuda", False)),
            (("cpu", False), ("cuda", True)),
            (("cuda", False), ("cuda", True)),
        ]

        for atol in [1e-6, 1e-5, 1e-4, 1e-3]:
            rtol = 1e-5
            for bp in backend_pairs:
                with Timer() as t:
                    passed, info = run_pair(model, inputs, bp, atol, rtol)
                log = {
                    "config": cfg_file,
                    "model": cfg.get("model", cfg.get("repo",{}).get("class","")),
                    "atol": atol,
                    "rtol": rtol,
                    "backend_pair": str(bp),
                    "passed": passed,
                    "info": info,
                    "latency": t.interval
                }
                with open(log_file, "a") as lf:
                    lf.write(json.dumps(log) + "\n")

if __name__ == "__main__":
    main()
