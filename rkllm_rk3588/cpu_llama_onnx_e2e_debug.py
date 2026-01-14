#!/usr/bin/env python3
import argparse, re, time, json, subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import onnxruntime as ort
import torch

# python3 cpu_llama_onnx_e2e_debug.py \
# --gguf hf_vieneu_03b_q4_gguf/VieNeu-TTS-0_3B-Q4_0.gguf \
# --codec_onnx ./neucodec_int8/model.onnx \
# --ref_codes_pt "./vieneu/assets/samples/Vĩnh (nam miền Nam).pt" \
# --ref_text "Đến cuối thế kỷ 19, ngành đánh bắt cá được thương mại hóa." \
# --text "Trên thực tế, các nghi ngờ đã bắt đầu xuất hiện." \
# --out out_cpu.wav \
# --dump_prefix cpu_cmp2 \
# --force_vi \
# --stop_on_end \
# --max_tokens 1024 \
# --n_ctx 4096

SAMPLES_PER_CODE = 480
SR = 24000
SPEECH_RE = re.compile(r"<\|speech_(\d+)\|>")

def read_ref_codes_pt(path: str) -> np.ndarray:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "codes" in obj:
        obj = obj["codes"]
    if isinstance(obj, torch.Tensor):
        obj = obj.cpu().numpy()
    return np.asarray(obj, dtype=np.int32).reshape(-1)

def codes_to_tags(codes: np.ndarray) -> str:
    return "".join([f"<|speech_{int(c)}|>" for c in codes.tolist()])

def _run(cmd):
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="ignore")

def espeak_ipa(text: str, voice="vi", espeak_bin="espeak-ng") -> str:
    cmd = [espeak_bin, "-q", "-v", voice, "--ipa", text]
    out = _run(cmd)
    return " ".join(out.strip().split())

def espeak_x(text: str, voice="vi", espeak_bin="espeak-ng") -> str:
    cmd = [espeak_bin, "-q", "-v", voice, "-x", text]
    out = _run(cmd)
    return " ".join(out.strip().split())

def build_prompt(ref_text: str, text: str, ref_codes: np.ndarray,
                 text_mode: str, espeak_bin: str, force_vi: bool):
    joined = (ref_text.strip() + " " + text.strip()).strip()
    voice = "vi" if force_vi else "en"

    if text_mode == "raw":
        payload = joined
        backend = "raw"
    elif text_mode == "espeak_x":
        payload = espeak_x(joined, voice=voice, espeak_bin=espeak_bin)
        backend = "espeak_x"
    elif text_mode == "espeak_ipa":
        payload = espeak_ipa(joined, voice=voice, espeak_bin=espeak_bin)
        backend = "espeak_ipa"
    else:
        raise ValueError("text_mode must be raw|espeak_x|espeak_ipa")

    prompt = "user: Convert the text to speech: " + payload + "\nassistant:" + codes_to_tags(ref_codes)
    return prompt, payload, backend

def llama_generate(gguf_path: str, prompt: str, max_tokens: int, n_ctx: int,
                   threads: int, temperature: float, top_p: float,
                   stop_on_end: bool):
    from llama_cpp import Llama

    t0 = time.time()
    llm = Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_threads=threads if threads > 0 else None,
        n_batch=512,
        logits_all=False,
        verbose=False,
    )
    t_load = time.time() - t0

    stops = []
    if stop_on_end:
        stops.append("<|SPEECH_GENERATION_END|>")
    stops.append("\nuser:")
    stops.append("user:")

    t1 = time.time()
    out = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stops if stops else None,
    )
    dt = time.time() - t1
    text_out = out["choices"][0]["text"]
    return text_out, t_load, dt

def extract_codes(text: str) -> list[int]:
    return [int(x) for x in SPEECH_RE.findall(text)]

def decode_neucodec(codec_onnx: str, codes: np.ndarray, sess: ort.InferenceSession | None = None) -> np.ndarray:
    x = np.asarray(codes, dtype=np.int32)[None, None, :]
    if sess is None:
        sess = ort.InferenceSession(codec_onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    wav = sess.run(None, {inp: x})[0].squeeze().astype(np.float32)
    return wav

def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))

def _rtf(dt: float, audio_s: float) -> float:
    if audio_s <= 0:
        return 0.0
    return dt / audio_s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--codec_onnx", required=True)
    ap.add_argument("--ref_codes_pt", required=True)
    ap.add_argument("--ref_text", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="out_cpu.wav")
    ap.add_argument("--dump_prefix", default="cpu_dbg")

    ap.add_argument("--text_mode", choices=["raw","espeak_x","espeak_ipa"], default="espeak_ipa")
    ap.add_argument("--espeak_bin", default="espeak-ng")
    ap.add_argument("--force_vi", action="store_true")

    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--n_ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--stop_on_end", action="store_true")
    args = ap.parse_args()

    bench = {}

    t0_all = time.time()

    t0 = time.time()
    ref_codes = read_ref_codes_pt(args.ref_codes_pt)
    bench["ref_codes_load"] = time.time() - t0
    print(f"[DBG] ref_codes: n={len(ref_codes)} min={int(ref_codes.min())} max={int(ref_codes.max())}")

    t0 = time.time()
    prompt, payload, backend = build_prompt(
        ref_text=args.ref_text,
        text=args.text,
        ref_codes=ref_codes,
        text_mode=args.text_mode,
        espeak_bin=args.espeak_bin,
        force_vi=args.force_vi,
    )
    bench["build_prompt_total"] = time.time() - t0

    # (optional) split build prompt timing: espeak vs string formatting
    # We can infer espeak time by re-running only when needed; but we keep it simple and measure total.

    prompt_path = f"{args.dump_prefix}_prompt.txt"
    Path(prompt_path).write_text(prompt, encoding="utf-8")
    Path(f"{args.dump_prefix}_meta.json").write_text(
        json.dumps({"backend": backend, "payload": payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[DBG] prompt_backend={backend}")
    print(f"[DBG] prompt_len={len(prompt)} newlines={prompt.count(chr(10))} (should be 1)")
    print(f"[DBG] wrote {prompt_path}")

    t0 = time.time()
    llm_out, t_load, t_gen = llama_generate(
        gguf_path=args.gguf,
        prompt=prompt,
        max_tokens=args.max_tokens,
        n_ctx=args.n_ctx,
        threads=args.threads,
        temperature=args.temperature,
        top_p=args.top_p,
        stop_on_end=args.stop_on_end,
    )
    bench["llama_load"] = t_load
    bench["llama_generate"] = t_gen
    bench["llama_total_call"] = time.time() - t0  # includes small python overhead

    llm_out_path = f"{args.dump_prefix}_llm_out.txt"
    Path(llm_out_path).write_text(llm_out, encoding="utf-8")
    print(f"[DBG] loaded llama-cpp in {t_load:.3f}s")
    print(f"[DBG] wrote {llm_out_path} (llm_wall={t_gen:.3f}s)")

    t0 = time.time()
    gen_codes = extract_codes(llm_out)
    bench["parse_codes"] = time.time() - t0

    print(f"[DBG] gen_codes: n={len(gen_codes)}" + (f" min={min(gen_codes)} max={max(gen_codes)}" if gen_codes else " (NONE)"))
    if gen_codes:
        print(f"[DBG] gen_head={gen_codes[:10]}")
        print(f"[DBG] gen_tail={gen_codes[-10:]}")

    codes_path = f"{args.dump_prefix}_codes.txt"
    Path(codes_path).write_text("\n".join(map(str, gen_codes)) + ("\n" if gen_codes else ""), encoding="utf-8")
    print(f"[DBG] wrote {codes_path}")

    # Init codec once (so you can see init cost separately)
    t0 = time.time()
    codec_sess = ort.InferenceSession(args.codec_onnx, providers=["CPUExecutionProvider"])
    bench["codec_init"] = time.time() - t0

    # Decode ref+gen then trim reference, plus gen-only for comparison
    t0 = time.time()
    all_codes = np.asarray(ref_codes.tolist() + gen_codes, dtype=np.int32)
    wav_full = decode_neucodec(args.codec_onnx, all_codes, sess=codec_sess)
    bench["decode_ref+gen"] = time.time() - t0

    t0 = time.time()
    cut = SAMPLES_PER_CODE * len(ref_codes) - SAMPLES_PER_CODE
    wav_trim = wav_full[cut:] if cut < wav_full.size else np.zeros((0,), np.float32)
    bench["trim_ref"] = time.time() - t0

    t0 = time.time()
    wav_gen = decode_neucodec(args.codec_onnx, np.asarray(gen_codes, dtype=np.int32), sess=codec_sess) if gen_codes else np.zeros((0,), np.float32)
    bench["decode_genonly"] = time.time() - t0

    t0 = time.time()
    out_full = f"{args.dump_prefix}_refgen_full.wav"
    out_gen = f"{args.dump_prefix}_genonly.wav"
    sf.write(out_full, wav_full, SR)
    sf.write(out_gen, wav_gen, SR)
    sf.write(args.out, wav_trim, SR)
    bench["write_wavs"] = time.time() - t0

    full_s = wav_full.size / SR
    gen_s = wav_gen.size / SR
    out_s = wav_trim.size / SR

    print(f"[OK] wrote {out_full} sec={full_s:.3f} rms={rms(wav_full):.6f}")
    print(f"[OK] wrote {out_gen} sec={gen_s:.3f} rms={rms(wav_gen):.6f}")
    print(f"[OK] wrote {args.out} sec={out_s:.3f} rms={rms(wav_trim):.6f}")

    if len(gen_codes) < 80:
        print("[WARN] Few generated codes → likely prompt/phoneme mismatch. Compare *_prompt.txt with your working llm_dump PROMPT.")

    total = time.time() - t0_all

    # RTF should be computed against the FINAL output duration
    audio_s = out_s if out_s > 0 else 0.0

    def _print_line(name: str, dt: float):
        if audio_s > 0:
            print(f"{name:16s}: {dt:7.3f}s | RTF={_rtf(dt, audio_s):.3f}")
        else:
            print(f"{name:16s}: {dt:7.3f}s | RTF(n/a)")

    print("\n========== BENCH (wall / RTF) ==========")
    # Prompt build is grouped; if you want espeak-only split, measure inside build_prompt() similarly.
    _print_line("ref_codes_load", bench.get("ref_codes_load", 0.0))
    _print_line("build_prompt",   bench.get("build_prompt_total", 0.0))
    _print_line("llama_load",     bench.get("llama_load", 0.0))
    _print_line("llama_generate", bench.get("llama_generate", 0.0))
    _print_line("parse_codes",    bench.get("parse_codes", 0.0))
    _print_line("codec_init",     bench.get("codec_init", 0.0))
    _print_line("decode_ref+gen", bench.get("decode_ref+gen", 0.0))
    _print_line("trim_ref",       bench.get("trim_ref", 0.0))
    _print_line("decode_genonly", bench.get("decode_genonly", 0.0))
    _print_line("write_wavs",     bench.get("write_wavs", 0.0))
    if audio_s > 0:
        print("----------------------------------------")
        print(f"{'TOTAL':16s}: {total:7.3f}s | RTF={total/audio_s:.3f}")
    print("========================================")

if __name__ == "__main__":
    main()
