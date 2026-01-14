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

# TTS flow: (espeak → rkllm → parse → codec decode → trim/write)

# python3 e2e_vieneu_tts_rk3588.py \
# --llm_demo ./llm_demo \
# --rkllm ./vieneu_03b_w8a8_rk3588.rkllm \
# --codec_onnx ./neucodec_int8/model.onnx \
# --ref_codes_pt "./vieneu/assets/samples/Vĩnh (nam miền Nam).pt" \
# --ref_text "Đến cuối thế kỷ 19, ngành đánh bắt cá được thương mại hóa." \
# --text "Ở một vương quốc xa xôi, có một vị vua già có bảy cô con gái công chúa. Bảy nàng công chúa này từng cãi nhau rất nhiều. Vì điều này, nhà vua thường mua quần áo giống nhau cho tất cả họ." \
# --out out.wav \
# --force_vi \
# --stop_on_end \
# --max_new_tokens 1024 \
# --max_context_len 4096 \
# --max_gen_codes 650 \
# --fade_ms 40 \
# --write_debug_wavs

SR = 24000
SAMPLES_PER_CODE = 480
CODES_PER_SEC = SR / SAMPLES_PER_CODE  # 50.0 codes/sec

SPEECH_RE = re.compile(r"<\|speech_(\d+)\|>")
ROBOT_RE = re.compile(r"robot:\s*(.*?)(?:\n\s*\nuser:|\Z)", re.DOTALL)

def now():
    return time.perf_counter()

def fmt(x: float) -> str:
    return f"{x:.3f}"

def safe_rtf(wall_s: float, audio_s: float) -> str:
    if audio_s <= 1e-9:
        return "inf"
    return f"{wall_s / audio_s:.3f}"

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

class NeuCodecDecoder:
    def __init__(self, codec_onnx: str):
        self.codec_onnx = codec_onnx
        self.sess = ort.InferenceSession(codec_onnx, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def decode(self, codes: np.ndarray) -> np.ndarray:
        x = np.asarray(codes, dtype=np.int32)[None, None, :]  # (1,1,T)
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        return y.squeeze().astype(np.float32)

def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))

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

    # optional caps / cleanup
    ap.add_argument("--max_gen_codes", type=int, default=0, help="Hard cap on generated codes (0=disable).")
    ap.add_argument("--drop_last_codes", type=int, default=0, help="Drop last N generated codes (0=disable).")
    ap.add_argument("--fade_ms", type=float, default=0.0, help="Fade-out length in ms (0=disable).")

    args = ap.parse_args()

    t_all0 = now()

    # -------- stage: load ref codes
    t0 = now()
    ref_codes = read_ref_codes_pt(args.ref_codes_pt)
    t_ref = now() - t0
    ref_audio_s = len(ref_codes) / CODES_PER_SEC
    print(f"[DBG] ref_codes: n={len(ref_codes)} min={int(ref_codes.min())} max={int(ref_codes.max())} ref_audio≈{ref_audio_s:.3f}s")

    # -------- stage: espeak ipa
    t0 = now()
    joined_text = (args.ref_text.strip() + " " + args.text.strip()).strip()
    voice = "vi" if args.force_vi else "en"
    ipa_payload = run_espeak_ipa(joined_text, espeak_bin=args.espeak_bin, voice=voice)
    t_ipa = now() - t0

    # -------- stage: build prompt + dump prompt
    t0 = now()
    prompt_line = build_prompt_single_line(ipa_payload, ref_codes)
    prompt_path = f"{args.dump_prefix}_prompt.txt"
    Path(prompt_path).write_text(prompt_line, encoding="utf-8")
    t_prompt = now() - t0

    print(f"[DBG] prompt_backend=espeak_ipa voice={voice}")
    print(f"[DBG] prompt_len={len(prompt_line)} newlines={prompt_line.count(chr(10))} (must be 0 for llm_demo)")
    print(f"[DBG] wrote {prompt_path}")

    # -------- stage: rkllm generate
    t0 = now()
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
    t_llm = now() - t0
    print(f"[DBG] saved llm stdout to {rk_out_path} (wall={t_llm:.3f}s)")

    # -------- stage: parse codes
    t0 = now()
    robot_block = extract_first_robot_block(stdout_text)
    if not robot_block:
        print("[ERR] Could not find first 'robot:' block in llm_demo output. Check *_rkllm_out.txt")
        return

    if args.stop_on_end:
        end_idx = robot_block.find("<|SPEECH_GENERATION_END|>")
        if end_idx != -1:
            robot_block = robot_block[:end_idx]

    gen_codes = extract_codes(robot_block)
    t_parse = now() - t0

    if not gen_codes:
        print("[ERR] No <|speech_...|> codes found in first robot block. Check *_rkllm_out.txt")
        return

    # caps / cleanup
    gen_raw_n = len(gen_codes)
    if args.max_gen_codes and len(gen_codes) > args.max_gen_codes:
        gen_codes = gen_codes[: args.max_gen_codes]
    if args.drop_last_codes and len(gen_codes) > args.drop_last_codes:
        gen_codes = gen_codes[:-args.drop_last_codes]

    Path(f"{args.dump_prefix}_codes.txt").write_text("\n".join(map(str, gen_codes)) + "\n", encoding="utf-8")

    gen_audio_s = len(gen_codes) / CODES_PER_SEC
    print(f"[DBG] gen_codes_raw: n={gen_raw_n}")
    print(f"[DBG] gen_codes_now: n={len(gen_codes)} min={min(gen_codes)} max={max(gen_codes)} gen_audio≈{gen_audio_s:.3f}s")
    print(f"[DBG] gen_head={gen_codes[:10]}")
    print(f"[DBG] gen_tail={gen_codes[-10:]}")

    # -------- stage: init codec session
    t0 = now()
    dec = NeuCodecDecoder(args.codec_onnx)
    t_codec_init = now() - t0

    # -------- stage: decode ref+gen
    t0 = now()
    all_codes = np.asarray(ref_codes.tolist() + gen_codes, dtype=np.int32)
    wav_full = dec.decode(all_codes)
    t_decode_full = now() - t0

    # trim ref portion
    t0 = now()
    cut = SAMPLES_PER_CODE * len(ref_codes) - SAMPLES_PER_CODE
    wav_trim = wav_full[cut:] if cut < wav_full.size else np.zeros((0,), np.float32)

    # optional fade out
    if args.fade_ms and wav_trim.size > 0:
        fade_n = int(SR * (args.fade_ms / 1000.0))
        fade_n = min(fade_n, wav_trim.size)
        if fade_n > 1:
            wav_trim[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)

    t_post = now() - t0

    # -------- stage: write wav
    t0 = now()
    sf.write(args.out, wav_trim, SR)
    t_write = now() - t0

    out_audio_s = wav_trim.size / SR
    total_wall = now() - t_all0

    print(f"[DBG] wav_rms={rms(wav_trim):.6f}")
    print(f"[OK] wrote {args.out} sec={out_audio_s:.3f}")

    # ===================== BENCHMARK SUMMARY =====================
    print("\n========== BENCH (wall / RTF) ==========")
    print(f"ref_codes_load : {fmt(t_ref)}s | RTF(n/a)")
    print(f"espeak_ipa     : {fmt(t_ipa)}s | RTF={safe_rtf(t_ipa, out_audio_s)}")
    print(f"build_prompt   : {fmt(t_prompt)}s | RTF={safe_rtf(t_prompt, out_audio_s)}")
    print(f"rkllm_generate : {fmt(t_llm)}s | RTF={safe_rtf(t_llm, out_audio_s)}")
    print(f"parse_codes    : {fmt(t_parse)}s | RTF={safe_rtf(t_parse, out_audio_s)}")
    print(f"codec_init     : {fmt(t_codec_init)}s | RTF={safe_rtf(t_codec_init, out_audio_s)}")
    print(f"decode_ref+gen : {fmt(t_decode_full)}s | RTF={safe_rtf(t_decode_full, out_audio_s)}")
    print(f"postprocess    : {fmt(t_post)}s | RTF={safe_rtf(t_post, out_audio_s)}")
    print(f"write_wav      : {fmt(t_write)}s | RTF={safe_rtf(t_write, out_audio_s)}")
    print("----------------------------------------")
    print(f"TOTAL          : {fmt(total_wall)}s | RTF={safe_rtf(total_wall, out_audio_s)}")
    print("========================================\n")

    if args.write_debug_wavs:
        t0 = now()
        sf.write(f"{args.dump_prefix}_refgen_full.wav", wav_full, SR)
        wav_gen = dec.decode(np.asarray(gen_codes, dtype=np.int32))
        sf.write(f"{args.dump_prefix}_genonly.wav", wav_gen, SR)
        t_dbg = now() - t0
        print(f"[DBG] wrote {args.dump_prefix}_refgen_full.wav and {args.dump_prefix}_genonly.wav (wall={t_dbg:.3f}s)")

if __name__ == "__main__":
    main()
