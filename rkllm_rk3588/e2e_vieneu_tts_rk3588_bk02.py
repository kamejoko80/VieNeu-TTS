#!/usr/bin/env python3
import argparse
import re
import time
import subprocess
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
import soundfile as sf

# export GRADIO_SERVER_NAME=0.0.0.0
# export GRADIO_SERVER_PORT=7860
# export VIENEU_DEBUG=1

# export VIENEU_DEBUG_LLM=1
# export VIENEU_DUMP_LLM=1
# export VIENEU_DUMP_DIR=./logs
# export VIENEU_DEBUG_LOG=./logs/vieneu_debug.log

# python3 e2e_vieneu_tts_rk3588.py \
# --llm_demo ./llm_demo \
# --rkllm ./vieneu_03b_w8a8_rk3588.rkllm \
# --codec_onnx ./neucodec_int8/model.onnx \
# --ref_codes_pt "./vieneu/assets/samples/Vĩnh (nam miền Nam).pt" \
# --ref_text "Đến cuối thế kỷ 19, ngành đánh bắt cá được thương mại hóa." \
# --text "Trên thực tế, các nghi ngờ đã bắt đầu xuất hiện." \
# --out out.wav \
# --force_vi \
# --stop_on_end \
# --max_new_tokens 1024 \
# --max_context_len 4096 \
# --cap_by_ratio \
# --ratio_pad 0.05 \
# --max_gen_codes 650 \
# --fade_ms 40 \
# --write_debug_wavs

SR = 24000
SAMPLES_PER_CODE = 480  # 20ms per code at 24kHz

SPEECH_RE = re.compile(r"<\|speech_(\d+)\|>")
ROBOT_RE = re.compile(r"robot:\s*(.*?)(?:\n\s*\nuser:|\Z)", re.DOTALL)

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
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="ignore")
    except FileNotFoundError:
        raise RuntimeError(f"Cannot find '{espeak_bin}'. Install espeak-ng or pass --espeak_bin PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "espeak-ng failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"OUT:\n{e.output.decode('utf-8', errors='ignore')}"
        )
    return " ".join(out.strip().split())

def build_prompt_single_line(ipa_payload: str, ref_codes: np.ndarray) -> str:
    return (
        "user: Convert the text to speech: "
        + ipa_payload
        + " assistant:"
        + codes_to_tags(ref_codes)
    )

def run_llm_demo(llm_demo: str, rkllm_model: str, max_new_tokens: int, max_context_len: int,
                prompt_line: str, timeout_s: float, save_stdout_path: str) -> str:
    cmd = [llm_demo, rkllm_model, str(max_new_tokens), str(max_context_len)]
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
        bufsize=1,
    )

    try:
        inp = prompt_line + "\nexit\n"
        out, _ = p.communicate(inp, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        p.kill()
        out, _ = p.communicate()

    Path(save_stdout_path).write_text(out, encoding="utf-8")
    return out

def extract_first_robot_block(stdout_text: str) -> str:
    m = ROBOT_RE.search(stdout_text)
    if not m:
        return ""
    return m.group(1)

def extract_codes(text: str) -> list[int]:
    return [int(x) for x in SPEECH_RE.findall(text)]

def decode_neucodec(codec_onnx: str, codes: np.ndarray) -> np.ndarray:
    x = np.asarray(codes, dtype=np.int32)[None, None, :]  # (1,1,T) int32
    sess = ort.InferenceSession(codec_onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    wav = sess.run(None, {inp: x})[0].squeeze().astype(np.float32)
    return wav

def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))

def estimate_max_gen_codes(ref_codes_len: int,
                           ipa_ref: str,
                           ipa_tgt: str,
                           ratio_pad: float,
                           hard_max: int | None) -> int:
    # Use token-like lengths (space split) rather than raw chars to be less sensitive.
    r = max(1, len(ipa_ref.split()))
    t = max(1, len(ipa_tgt.split()))
    ratio = t / r
    est = int(round(ref_codes_len * ratio * (1.0 + ratio_pad)))
    if hard_max is not None:
        est = min(est, int(hard_max))
    return max(1, est)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_demo", required=True)
    ap.add_argument("--rkllm", required=True)
    ap.add_argument("--codec_onnx", required=True)

    ap.add_argument("--ref_codes_pt", required=True)
    ap.add_argument("--ref_text", required=True)
    ap.add_argument("--text", required=True)

    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--dump_prefix", default="rk_case")

    ap.add_argument("--espeak_bin", default="espeak-ng")
    ap.add_argument("--force_vi", action="store_true")
    ap.add_argument("--stop_on_end", action="store_true")

    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_context_len", type=int, default=4096)
    ap.add_argument("--timeout_s", type=float, default=120.0)

    ap.add_argument("--write_debug_wavs", action="store_true")

    # NEW: cap generated codes by ref-vs-target phoneme ratio
    ap.add_argument("--cap_by_ratio", action="store_true",
                    help="Cap gen codes by (len(ipa_tgt)/len(ipa_ref)) * len(ref_codes).")
    ap.add_argument("--ratio_pad", type=float, default=0.15,
                    help="Extra padding for cap_by_ratio, e.g. 0.15 = +15%%.")
    ap.add_argument("--max_gen_codes", type=int, default=0,
                    help="Hard max gen codes (0 = unlimited). Useful safety clamp.")

    # Optional cosmetic: fade-out last N ms (doesn't remove voiced tail; only smooths cut)
    ap.add_argument("--fade_ms", type=int, default=0)

    args = ap.parse_args()

    ref_codes = read_ref_codes_pt(args.ref_codes_pt)
    print(f"[DBG] ref_codes: n={len(ref_codes)} min={int(ref_codes.min())} max={int(ref_codes.max())}")

    voice = "vi" if args.force_vi else "en"

    # IMPORTANT: generate IPA for ref and target separately (for ratio cap)
    ipa_ref = run_espeak_ipa(args.ref_text.strip(), espeak_bin=args.espeak_bin, voice=voice)
    ipa_tgt = run_espeak_ipa(args.text.strip(), espeak_bin=args.espeak_bin, voice=voice)

    ipa_payload = (ipa_ref + " " + ipa_tgt).strip()
    prompt_line = build_prompt_single_line(ipa_payload, ref_codes)

    prompt_path = f"{args.dump_prefix}_prompt.txt"
    Path(prompt_path).write_text(prompt_line, encoding="utf-8")
    print(f"[DBG] prompt_backend=espeak_ipa voice={voice}")
    print(f"[DBG] prompt_len={len(prompt_line)} newlines={prompt_line.count(chr(10))} (must be 0 for llm_demo)")
    print(f"[DBG] wrote {prompt_path}")

    # dump IPA parts for debugging
    Path(f"{args.dump_prefix}_ipa_ref.txt").write_text(ipa_ref + "\n", encoding="utf-8")
    Path(f"{args.dump_prefix}_ipa_tgt.txt").write_text(ipa_tgt + "\n", encoding="utf-8")

    t0 = time.time()
    rk_out_path = f"{args.dump_prefix}_rkllm_out.txt"
    stdout_text = run_llm_demo(
        llm_demo=args.llm_demo,
        rkllm_model=args.rkllm,
        max_new_tokens=args.max_new_tokens,
        max_context_len=args.max_context_len,
        prompt_line=prompt_line,
        timeout_s=args.timeout_s,
        save_stdout_path=rk_out_path,
    )
    llm_wall = time.time() - t0
    print(f"[DBG] saved llm stdout to {rk_out_path} (wall={llm_wall:.3f}s)")

    robot_block = extract_first_robot_block(stdout_text)
    if not robot_block:
        print("[ERR] Could not find first 'robot:' block in llm_demo output. Check *_rkllm_out.txt")
        return

    if args.stop_on_end:
        end_idx = robot_block.find("<|SPEECH_GENERATION_END|>")
        if end_idx != -1:
            robot_block = robot_block[:end_idx]

    gen_codes = extract_codes(robot_block)
    if not gen_codes:
        Path(f"{args.dump_prefix}_codes.txt").write_text("", encoding="utf-8")
        print("[ERR] No <|speech_...|> codes found in first robot block. Check *_rkllm_out.txt")
        return

    print(f"[DBG] gen_codes_raw: n={len(gen_codes)} min={min(gen_codes)} max={max(gen_codes)}")
    print(f"[DBG] gen_head={gen_codes[:10]}")
    print(f"[DBG] gen_tail={gen_codes[-10:]}")

    # NEW: cap by phoneme ratio (cuts voiced hallucinated tail)
    hard_max = None if args.max_gen_codes <= 0 else int(args.max_gen_codes)
    if args.cap_by_ratio:
        cap = estimate_max_gen_codes(
            ref_codes_len=len(ref_codes),
            ipa_ref=ipa_ref,
            ipa_tgt=ipa_tgt,
            ratio_pad=float(args.ratio_pad),
            hard_max=hard_max,
        )
        before = len(gen_codes)
        gen_codes = gen_codes[:cap]
        print(f"[DBG] cap_by_ratio=ON ratio_pad={args.ratio_pad} cap={cap} gen_codes: {before} -> {len(gen_codes)}")
    elif hard_max is not None:
        before = len(gen_codes)
        gen_codes = gen_codes[:hard_max]
        print(f"[DBG] hard_max_gen_codes cap={hard_max} gen_codes: {before} -> {len(gen_codes)}")

    Path(f"{args.dump_prefix}_codes.txt").write_text("\n".join(map(str, gen_codes)) + "\n", encoding="utf-8")

    all_codes = np.asarray(ref_codes.tolist() + gen_codes, dtype=np.int32)
    wav_full = decode_neucodec(args.codec_onnx, all_codes)

    cut = SAMPLES_PER_CODE * len(ref_codes) - SAMPLES_PER_CODE
    wav_trim = wav_full[cut:] if cut < wav_full.size else np.zeros((0,), np.float32)

    # Optional fade-out to avoid click after hard cut
    if args.fade_ms > 0 and wav_trim.size > 0:
        fade_n = int(SR * (args.fade_ms / 1000.0))
        fade_n = min(fade_n, wav_trim.size)
        if fade_n > 1:
            wav_trim[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
        print(f"[DBG] applied fade_ms={args.fade_ms} (samples={fade_n})")

    sf.write(args.out, wav_trim, SR)
    est_audio = wav_trim.size / SR
    print(f"[DBG] wav_rms={rms(wav_trim):.6f}")
    if est_audio > 0:
        print(f"[OK] wrote {args.out} sec={est_audio:.3f} RTF={llm_wall/est_audio:.3f}")
    else:
        print(f"[OK] wrote {args.out} sec=0.000")

    if args.write_debug_wavs:
        sf.write(f"{args.dump_prefix}_refgen_full.wav", wav_full, SR)
        wav_gen = decode_neucodec(args.codec_onnx, np.asarray(gen_codes, dtype=np.int32))
        sf.write(f"{args.dump_prefix}_genonly.wav", wav_gen, SR)
        print(f"[DBG] wrote {args.dump_prefix}_refgen_full.wav and {args.dump_prefix}_genonly.wav")

if __name__ == "__main__":
    main()
