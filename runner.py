import yaml
import torch
import json
import os
import time
from tqdm import tqdm
from datetime import datetime
from model_loader import ModelLoader
from comparators import compare_outputs
from cb_utils import set_global_seed, Timer

def run_once(model, device, inputs, use_compile=False):
    model = model.to(device)
    if use_compile:
        model = torch.compile(model, backend="aot_eager")
    with torch.no_grad():
        if device=="cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        outputs = model(inputs.to(device))
        if device=="cuda":
            torch.cuda.synchronize()
        t1 = time.time()
    latency_ms = (t1 - t0)*1000
    return outputs, latency_ms

def run_pair(model, inputs, backend_pair, atol, rtol):
    (dev1, compile1), (dev2, compile2) = backend_pair
    out1, latency1 = run_once(model, dev1, inputs, compile1)
    out2, latency2 = run_once(model, dev2, inputs, compile2)
    passed, info = compare_outputs(out1, out2, atol, rtol)
    return passed, info, latency1, latency2

def load_yaml_auto(path: str):
    """
    自动检测文件编码 (UTF-8 / GBK) 并加载 YAML 文件
    """
    encodings = ["utf-8", "gbk"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            continue  # 如果解码失败，换下一个编码
    raise UnicodeDecodeError("无法用 utf-8 或 gbk 解码，请确认文件编码。")

def main():
    set_global_seed(5)
    loader = ModelLoader()
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"results_{datetime.now().date()}.jsonl")

    for cfg_file in os.listdir("yaml_configs"):
        if not cfg_file.lower().endswith((".yaml", ".yml")):
            continue

        cfg_path = os.path.join("yaml_configs", cfg_file)
        cfg = load_yaml_auto(cfg_path)

        if cfg["from"] == "library":
            model = loader.from_library(cfg["model"])
        elif cfg["from"] == "repo":
            model = loader.from_repo(cfg["repo"]["path"], cfg["repo"]["class"], params=cfg.get("params", {}))
        elif cfg["from"] == "hub":
            hub = cfg["hub"]
            model = loader.from_hub(hub["repo"], hub["entry"], **(hub.get("kwargs", {})))
        elif cfg["from"] == "builder":
            b = cfg["builder"]
            model = loader.from_builder(
                repo_path=b["repo_path"],
                builder=b["func"],
                cfg=b.get("cfg"),
                cfg_file=b.get("cfg_file"),
                cfg_options=b.get("cfg_options"),
                weights=cfg.get("weights"),
            )
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
            (("cpu", False), ("cpu", True)),
            #(("cuda", False), ("cuda", True)),
        ]

        for atol in [1e-6, 1e-5, 1e-4, 1e-3]:
            rtol = 1e-5
            for bp in backend_pairs:
                with Timer() as t:
                    try:
                        passed, info, latency1, latency2 = run_pair(model, inputs, bp, atol, rtol)
                        if not passed:
                            if info.get("reason") in ("shape_mismatch","length_mismatch"):
                                failure_type = "B"
                            else:
                                failure_type = "C"
                        else:
                                failure_type = None
                    except Exception as e:
                        passed, info = False, {"error": str(e)}
                        failure_type = "A"
                        latency1, latency2 = None, None
                log = {
                    "config": cfg_file,
                    "model": cfg.get("model", cfg.get("repo",{}).get("class","")),
                    "atol": atol,
                    "rtol": rtol,
                    "backend_ref": bp[0][0] + ("+compile" if bp[0][1] else ""),
                    "backend_tgt": bp[1][0] + ("+compile" if bp[1][1] else ""),
                    "status": "PASS" if passed else "FAIL",
                    "info": info,
                    "latency": t.interval,
                    "failure_type": failure_type,
                    "latency_ref_ms": latency1,
                    "latency_tgt_ms": latency2
                }
                with open(log_file, "a") as lf:
                    lf.write(json.dumps(log) + "\n")

if __name__ == "__main__":
    main()
