import argparse
import numpy as np
import torch

# python3 make_rkllm_prompt.py \
#  --ref_codes_pt "./vieneu/assets/samples/Vĩnh (nam miền Nam).pt" \
#  --ref_text "Đến cuối thế kỷ 19, ngành đánh bắt cá được thương mại hóa." \
#  --text "Trên thực tế, các nghi ngờ đã bắt đầu xuất hiện." \
#  --out prompt.txt

def codes_to_tokens(codes: np.ndarray) -> str:
    return "".join([f"<|speech_{int(x)}|>" for x in codes.tolist()])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_codes_pt", required=True)
    ap.add_argument("--ref_text", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="prompt.txt")
    args = ap.parse_args()

    codes = torch.load(args.ref_codes_pt, map_location="cpu", weights_only=True)
    if isinstance(codes, torch.Tensor):
        codes = codes.cpu().numpy()
    codes = np.asarray(codes, dtype=np.int32).reshape(-1)

    ref_tokens = codes_to_tokens(codes)

    prompt = (
        "user: Convert the text to speech:"
        "<|TEXT_PROMPT_START|>"
        + args.ref_text.strip()
        + "\n"
        + args.text.strip()
        + "<|TEXT_PROMPT_END|>\n"
        "assistant:<|SPEECH_PROMPT_START|>"
        + ref_tokens
        + "<|SPEECH_PROMPT_END|><|SPEECH_GENERATION_START|>\n"
    )

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(prompt)

    print(f"Wrote {args.out}")
    print(f"ref_codes_len={len(codes)}")

if __name__ == "__main__":
    main()
