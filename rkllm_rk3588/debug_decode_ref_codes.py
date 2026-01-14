import argparse, re
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

# python3 debug_decode_ref_codes.py --codec_onnx ./neucodec_int8/model.onnx --ref_codes_pt "./vieneu/assets/samples/Vĩnh (nam miền Nam).pt" --out ref_recon.wav

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec_onnx", required=True)
    ap.add_argument("--ref_codes_pt", required=True)  # the preset .pt path
    ap.add_argument("--out", default="ref_recon.wav")
    ap.add_argument("--sr", type=int, default=24000)
    args = ap.parse_args()

    codes = torch.load(args.ref_codes_pt, map_location="cpu", weights_only=True)
    if isinstance(codes, torch.Tensor):
        codes = codes.cpu().numpy()
    codes = np.asarray(codes, dtype=np.int32).reshape(1, 1, -1)

    sess = ort.InferenceSession(args.codec_onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    wav = sess.run(None, {inp.name: codes})[0].squeeze().astype(np.float32)

    sf.write(args.out, wav, args.sr)
    print("Wrote", args.out, "codes", codes.shape[-1], "wav_sec", wav.size / args.sr)

if __name__ == "__main__":
    main()
