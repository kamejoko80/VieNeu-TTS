#!/usr/bin/env python3
import argparse
import ctypes
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
import soundfile as sf

# python3 e2e_vieneu_tts_rk3588_pyapi.py \
# --rkllm ./vieneu_03b_w8a8_rk3588.rkllm \
# --rkllmrt ./librkllmrt.so \
# --codec_onnx ./neucodec_int8/model.onnx \
# --ref_codes_pt "./vieneu/assets/samples/Vĩnh (nam miền Nam).pt" \
# --ref_text "Đến cuối thế kỷ 19, ngành đánh bắt cá được thương mại hóa." \
# --text "Ở một vương quốc xa xôi, có một vị vua già có bảy cô con gái công chúa. Bảy nàng công chúa này từng cãi nhau rất nhiều. Vì điều này, nhà vua thường mua quần áo giống nhau cho tất cả họ." \
# --out out.wav \
# --force_vi \
# --stop_on_end \
# --max_new_tokens 1024 \
# --max_context_len 4096 \
# --fade_ms 40 \
# --write_debug_wavs

SR = 24000
SAMPLES_PER_CODE = 480

SPEECH_RE = re.compile(r"<\|speech_(\d+)\|>")
END_TOK = "<|SPEECH_GENERATION_END|>"


def read_ref_codes_pt(path: str) -> np.ndarray:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "codes" in obj:
        obj = obj["codes"]
    if isinstance(obj, torch.Tensor):
        obj = obj.cpu().numpy()
    return np.asarray(obj, dtype=np.int32).reshape(-1)


def codes_to_tags(codes: np.ndarray) -> str:
    return "".join([f"<|speech_{int(c)}|>" for c in codes.tolist()])


def run_espeak_ipa(text: str, espeak_bin: str = "espeak-ng", voice: str = "vi") -> str:
    cmd = [espeak_bin, "-q", "-v", voice, "--ipa", text]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="ignore")
    return " ".join(out.strip().split())


def build_prompt_single_line(ipa_payload: str, ref_codes: np.ndarray) -> str:
    # single line (no '\n') is safe for rkllm prompt input
    return "user: Convert the text to speech: " + ipa_payload + " assistant:" + codes_to_tags(ref_codes)


def extract_codes(text: str) -> list[int]:
    return [int(x) for x in SPEECH_RE.findall(text)]


def decode_neucodec(codec_onnx: str, codes: np.ndarray, sess: ort.InferenceSession) -> np.ndarray:
    x = np.asarray(codes, dtype=np.int32)[None, None, :]  # (1,1,T) int32 required
    inp = sess.get_inputs()[0].name
    y = sess.run(None, {inp: x})[0]
    return y.squeeze().astype(np.float32)


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def fade_out(wav: np.ndarray, fade_ms: int) -> np.ndarray:
    if fade_ms <= 0 or wav.size == 0:
        return wav
    n = int(SR * (fade_ms / 1000.0))
    if n <= 1:
        return wav
    n = min(n, wav.size)
    win = np.linspace(1.0, 0.0, n, dtype=np.float32)
    out = wav.copy()
    out[-n:] *= win
    return out


def cap_codes_by_ratio(ref_codes: np.ndarray, gen_codes: list[int], ratio_pad: float, max_gen_codes: int) -> list[int]:
    # If model keeps generating extra speech codes (no END token or late END),
    # this prevents the “extra tail voice” by hard capping length.
    cap = int(len(ref_codes) * (1.0 + ratio_pad))
    cap = min(cap, max_gen_codes)
    cap = max(cap, 0)
    return gen_codes[:cap]


# ---------------- RKLLM ctypes binding (match flask_server.py ABI) ----------------
RKLLM_Handle_t = ctypes.c_void_p
userdata = ctypes.c_void_p(None)

LLMCallState = ctypes.c_int
LLMCallState.RKLLM_RUN_NORMAL = 0
LLMCallState.RKLLM_RUN_WAITING = 1
LLMCallState.RKLLM_RUN_FINISH = 2
LLMCallState.RKLLM_RUN_ERROR = 3

RKLLMInputType = ctypes.c_int
RKLLMInputType.RKLLM_INPUT_PROMPT = 0
RKLLMInputType.RKLLM_INPUT_TOKEN = 1
RKLLMInputType.RKLLM_INPUT_EMBED = 2
RKLLMInputType.RKLLM_INPUT_MULTIMODAL = 3

RKLLMInferMode = ctypes.c_int
RKLLMInferMode.RKLLM_INFER_GENERATE = 0
RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
RKLLMInferMode.RKLLM_INFER_GET_LOGITS = 2


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [("lora_adapter_name", ctypes.c_char_p)]


class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", RKLLMInferMode),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("keep_history", ctypes.c_int),
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat),
    ]


class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMMultiModalInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t),
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModalInput),
    ]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", RKLLMInputType),
        ("input_data", RKLLMInputUnion),
    ]


callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rkllm", required=True, help=".rkllm model path")
    ap.add_argument("--rkllmrt", default="./librkllmrt.so", help="path to librkllmrt.so")
    ap.add_argument("--target_platform", default="rk3588", help="rk3588/rk3576/rv1126b/rk3562")
    ap.add_argument("--codec_onnx", required=True)

    ap.add_argument("--ref_codes_pt", required=True)
    ap.add_argument("--ref_text", required=True)
    ap.add_argument("--text", required=True)

    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--dump_prefix", default="rk_py")

    ap.add_argument("--espeak_bin", default="espeak-ng")
    ap.add_argument("--force_vi", action="store_true")

    ap.add_argument("--stop_on_end", action="store_true")

    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_context_len", type=int, default=4096)
    ap.add_argument("--n_batch", type=int, default=1)

    ap.add_argument("--cap_by_ratio", action="store_true")
    ap.add_argument("--ratio_pad", type=float, default=0.05)
    ap.add_argument("--max_gen_codes", type=int, default=650)

    ap.add_argument("--fade_ms", type=int, default=40)

    ap.add_argument("--write_debug_wavs", action="store_true")
    args = ap.parse_args()

    bench = {}

    t0 = time.time()
    ref_codes = read_ref_codes_pt(args.ref_codes_pt)
    bench["ref_codes_load"] = time.time() - t0
    print(f"[DBG] ref_codes: n={len(ref_codes)} min={int(ref_codes.min())} max={int(ref_codes.max())}")

    joined_text = (args.ref_text.strip() + " " + args.text.strip()).strip()
    voice = "vi" if args.force_vi else "en"

    t0 = time.time()
    ipa_payload = run_espeak_ipa(joined_text, espeak_bin=args.espeak_bin, voice=voice)
    bench["espeak_ipa"] = time.time() - t0

    t0 = time.time()
    prompt_line = build_prompt_single_line(ipa_payload, ref_codes)
    bench["build_prompt"] = time.time() - t0

    Path(f"{args.dump_prefix}_prompt.txt").write_text(prompt_line, encoding="utf-8")
    print(f"[DBG] prompt_backend=espeak_ipa voice={voice}")
    print(f"[DBG] prompt_len={len(prompt_line)} newlines={prompt_line.count(chr(10))}")
    print(f"[DBG] wrote {args.dump_prefix}_prompt.txt")

    # init codec session once
    t0 = time.time()
    codec_sess = ort.InferenceSession(args.codec_onnx, providers=["CPUExecutionProvider"])
    bench["codec_init"] = time.time() - t0

    # load rkllm runtime
    rkllm_lib = ctypes.CDLL(str(Path(args.rkllmrt).resolve()))

    rkllm_init = rkllm_lib.rkllm_init
    rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
    rkllm_init.restype = ctypes.c_int

    rkllm_run = rkllm_lib.rkllm_run
    rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
    rkllm_run.restype = ctypes.c_int

    rkllm_destroy = rkllm_lib.rkllm_destroy
    rkllm_destroy.argtypes = [RKLLM_Handle_t]
    rkllm_destroy.restype = ctypes.c_int

    output_chunks: list[str] = []
    global_state = {"v": -1}

    def _cb(res_p, _userdata, state_i):
        global_state["v"] = int(state_i)
        if state_i == LLMCallState.RKLLM_RUN_NORMAL:
            res = res_p.contents
            if res.text:
                output_chunks.append(ctypes.cast(res.text, ctypes.c_char_p).value.decode("utf-8", errors="ignore"))
        return 0

    cb = callback_type(_cb)  # keep alive

    # rkllm init
    t0 = time.time()
    rk_param = RKLLMParam()
    rk_param.model_path = str(Path(args.rkllm).resolve()).encode("utf-8")
    rk_param.max_context_len = int(args.max_context_len)
    rk_param.max_new_tokens = int(args.max_new_tokens)

    # IMPORTANT for VieNeu: keep speech tokens visible
    rk_param.skip_special_token = False
    rk_param.is_async = False

    rk_param.n_keep = -1
    rk_param.top_k = 1
    rk_param.top_p = 0.9
    rk_param.temperature = 0.8
    rk_param.repeat_penalty = 1.1
    rk_param.frequency_penalty = 0.0
    rk_param.presence_penalty = 0.0
    rk_param.mirostat = 0
    rk_param.mirostat_tau = 5.0
    rk_param.mirostat_eta = 0.1

    rk_param.img_start = b""
    rk_param.img_end = b""
    rk_param.img_content = b""

    rk_param.extend_param.base_domain_id = 0
    rk_param.extend_param.embed_flash = 1
    rk_param.extend_param.use_cross_attn = 0

    nb = int(args.n_batch)
    nb = 1 if nb < 1 else (100 if nb > 100 else nb)
    rk_param.extend_param.n_batch = nb

    rk_param.extend_param.enabled_cpus_num = 4
    if args.target_platform.lower() in ["rk3576", "rk3588"]:
        rk_param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
    else:
        rk_param.extend_param.enabled_cpus_mask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)

    handle = RKLLM_Handle_t()
    ret = rkllm_init(ctypes.byref(handle), ctypes.byref(rk_param), cb)
    if ret != 0:
        raise RuntimeError(f"rkllm_init failed: {ret}")
    bench["rkllm_init"] = time.time() - t0

    # rkllm run (exclude init)
    t0 = time.time()

    # keep prompt + role buffers alive across the C call
    role_buf = ctypes.create_string_buffer(b"user")
    prompt_buf = ctypes.create_string_buffer(prompt_line.encode("utf-8", errors="ignore"))

    rk_in = RKLLMInput()
    rk_in.role = ctypes.cast(role_buf, ctypes.c_char_p)
    rk_in.enable_thinking = ctypes.c_bool(False)
    rk_in.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
    rk_in.input_data.prompt_input = ctypes.cast(prompt_buf, ctypes.c_char_p)

    rk_inf = RKLLMInferParam()
    ctypes.memset(ctypes.byref(rk_inf), 0, ctypes.sizeof(RKLLMInferParam))
    rk_inf.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
    rk_inf.lora_params = None
    rk_inf.prompt_cache_params = None
    rk_inf.keep_history = 0

    ret = rkllm_run(handle, ctypes.byref(rk_in), ctypes.byref(rk_inf), None)
    if ret != 0:
        rkllm_destroy(handle)
        raise RuntimeError(f"rkllm_run failed: {ret}")

    bench["rkllm_run"] = time.time() - t0

    rkllm_destroy(handle)

    # parse output
    t0 = time.time()
    out_text = "".join(output_chunks)
    Path(f"{args.dump_prefix}_rkllm_out.txt").write_text(out_text, encoding="utf-8")
    print(f"[DBG] wrote {args.dump_prefix}_rkllm_out.txt")

    if args.stop_on_end:
        p = out_text.find(END_TOK)
        if p != -1:
            out_text = out_text[:p]

    gen_codes = extract_codes(out_text)
    bench["parse_codes"] = time.time() - t0

    if not gen_codes:
        print("[ERR] No <|speech_...|> codes parsed from RKLLM output. Check *_rkllm_out.txt")
        return

    print(f"[DBG] gen_codes_raw: n={len(gen_codes)} min={min(gen_codes)} max={max(gen_codes)}")
    print(f"[DBG] gen_head={gen_codes[:10]}")
    print(f"[DBG] gen_tail={gen_codes[-10:]}")
    Path(f"{args.dump_prefix}_codes.txt").write_text("\n".join(map(str, gen_codes)) + "\n", encoding="utf-8")

    if args.cap_by_ratio:
        before = len(gen_codes)
        gen_codes = cap_codes_by_ratio(ref_codes, gen_codes, args.ratio_pad, args.max_gen_codes)
        print(f"[DBG] cap_by_ratio -> {len(gen_codes)} (was {before}), max_gen_codes={args.max_gen_codes} ratio_pad={args.ratio_pad}")

    # decode ref+gen and trim ref part
    t0 = time.time()
    all_codes = np.asarray(ref_codes.tolist() + gen_codes, dtype=np.int32)
    wav_full = decode_neucodec(args.codec_onnx, all_codes, sess=codec_sess)
    bench["decode_ref+gen"] = time.time() - t0

    t0 = time.time()
    cut = SAMPLES_PER_CODE * len(ref_codes) - SAMPLES_PER_CODE
    wav_trim = wav_full[cut:] if cut < wav_full.size else np.zeros((0,), np.float32)
    wav_trim = fade_out(wav_trim, args.fade_ms)
    bench["postprocess"] = time.time() - t0

    t0 = time.time()
    sf.write(args.out, wav_trim, SR)
    bench["write_wav"] = time.time() - t0

    est_audio = wav_trim.size / SR
    print(f"[DBG] wav_rms={rms(wav_trim):.6f}")
    if est_audio > 0:
        print(f"[OK] wrote {args.out} sec={est_audio:.3f}")

    total = sum(bench.values())
    print("\n========== BENCH (wall / RTF) ==========")
    for k in [
        "ref_codes_load",
        "espeak_ipa",
        "build_prompt",
        "rkllm_init",
        "rkllm_run",
        "parse_codes",
        "codec_init",
        "decode_ref+gen",
        "postprocess",
        "write_wav",
    ]:
        if k not in bench:
            continue
        dt = bench[k]
        rtf = (dt / est_audio) if est_audio > 0 else None
        if rtf is None:
            print(f"{k:12s}: {dt:.3f}s | RTF(n/a)")
        else:
            print(f"{k:12s}: {dt:.3f}s | RTF={rtf:.3f}")
    if est_audio > 0:
        print("----------------------------------------")
        print(f"{'TOTAL':12s}: {total:.3f}s | RTF={total/est_audio:.3f}")
    print("========================================")

    if args.write_debug_wavs:
        sf.write(f"{args.dump_prefix}_refgen_full.wav", wav_full, SR)
        wav_gen = decode_neucodec(args.codec_onnx, np.asarray(gen_codes, dtype=np.int32), sess=codec_sess)
        sf.write(f"{args.dump_prefix}_genonly.wav", wav_gen, SR)
        print(f"[DBG] wrote {args.dump_prefix}_refgen_full.wav and {args.dump_prefix}_genonly.wav")


if __name__ == "__main__":
    main()
