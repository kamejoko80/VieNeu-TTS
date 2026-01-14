import json, random

base = [
  "Xin chào", "Chào bạn", "Hôm nay", "Ngày mai", "Bây giờ", "Tôi", "Bạn", "Chúng ta",
  "đang kiểm tra", "muốn thử", "cần đánh giá", "hiệu năng", "chất lượng", "giọng nói",
  "mô hình", "trên RK3588", "với NPU", "và CPU", "độ trễ", "tốc độ", "thời gian thực"
]
tails = [
  ".", "!", "?", "…",
  ". Hãy đọc câu này thật tự nhiên.",
  ". Bạn có thể nói chậm lại một chút không?",
  ". Xin hãy nhấn giọng ở các từ quan trọng."
]

def gen_sentence():
  n = random.randint(6, 18)
  s = " ".join(random.choice(base) for _ in range(n))
  s = s[0].upper() + s[1:]
  return s + random.choice(tails)

random.seed(0)
data = [{"input": gen_sentence(), "target": ""} for _ in range(1200)]

with open("data_quant.json", "w", encoding="utf-8") as f:
  json.dump(data, f, ensure_ascii=False, indent=2)

print("Wrote data_quant.json with", len(data), "samples")
