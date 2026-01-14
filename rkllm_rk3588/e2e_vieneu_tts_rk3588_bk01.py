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
# --write_debug_wavs

SR = 24000
SAMPLES_PER_CODE = 480

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
    # IMPORTANT: single line (llm_demo reads line-by-line)
    # Matches your working dump style except newline is replaced by a space.
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
        # Send exactly one line prompt + newline + exit + newline
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
    x = np.asarray(codes, dtype=np.int32)[None, None, :]  # (1,1,T)
    sess = ort.InferenceSession(codec_onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    wav = sess.run(None, {inp: x})[0].squeeze().astype(np.float32)
    return wav


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
    args = ap.parse_args()

    ref_codes = read_ref_codes_pt(args.ref_codes_pt)
    print(f"[DBG] ref_codes: n={len(ref_codes)} min={int(ref_codes.min())} max={int(ref_codes.max())}")

    joined_text = (args.ref_text.strip() + " " + args.text.strip()).strip()
    voice = "vi" if args.force_vi else "en"
    ipa_payload = run_espeak_ipa(joined_text, espeak_bin=args.espeak_bin, voice=voice)

    prompt_line = build_prompt_single_line(ipa_payload, ref_codes)

    prompt_path = f"{args.dump_prefix}_prompt.txt"
    Path(prompt_path).write_text(prompt_line, encoding="utf-8")
    print(f"[DBG] prompt_backend=espeak_ipa voice={voice}")
    print(f"[DBG] prompt_len={len(prompt_line)} newlines={prompt_line.count(chr(10))} (must be 0 for llm_demo)")
    print(f"[DBG] wrote {prompt_path}")

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

    Path(f"{args.dump_prefix}_codes.txt").write_text("\n".join(map(str, gen_codes)) + ("\n" if gen_codes else ""), encoding="utf-8")

    if not gen_codes:
        print("[ERR] No <|speech_...|> codes found in first robot block. Check *_rkllm_out.txt")
        return

    print(f"[DBG] gen_codes: n={len(gen_codes)} min={min(gen_codes)} max={max(gen_codes)}")
    print(f"[DBG] gen_head={gen_codes[:10]}")
    print(f"[DBG] gen_tail={gen_codes[-10:]}")

    # Decode ref+gen then trim ref portion (best quality)
    all_codes = np.asarray(ref_codes.tolist() + gen_codes, dtype=np.int32)
    wav_full = decode_neucodec(args.codec_onnx, all_codes)

    cut = SAMPLES_PER_CODE * len(ref_codes) - SAMPLES_PER_CODE
    wav_trim = wav_full[cut:] if cut < wav_full.size else np.zeros((0,), np.float32)

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
