from huggingface_hub import snapshot_download
from rkllm.api import RKLLM
import os

REPO_ID = "pnnbao-ump/VieNeu-TTS-0.3B"
LOCAL_DIR = "./hf_vieneu_tts_03b"
OUT = "vieneu_03b_w8a8_rk3588.rkllm"

hf_token = os.getenv("HF_TOKEN", None)

# 1) Download the HF repo to a local folder
path = snapshot_download(
    repo_id=REPO_ID,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,
    token=hf_token,
)

print("HF snapshot at:", path)

# 2) Load from LOCAL path (this is what rkllm-toolkit expects here)
llm = RKLLM()
ret = llm.load_huggingface(model=path, model_lora=None, device="cuda")
if ret != 0:
    raise RuntimeError(f"load_huggingface failed: {ret}")

# 3) Build + quantize
ret = llm.build(
    do_quantization=True,
    optimization_level=1,
    quantized_dtype="w8a8",
    quantized_algorithm="normal",
    target_platform="rk3588",
    num_npu_core=3,
    dataset="./data_quant.json",   # make sure this exists
)
if ret != 0:
    raise RuntimeError(f"build failed: {ret}")

ret = llm.export_rkllm(OUT)
if ret != 0:
    raise RuntimeError(f"export_rkllm failed: {ret}")

print("OK:", OUT)
