---------------------------------------------------------------------
VieNeu-TTS (Linux X86 setup)

mkdir VieNeu-TTS
cd VieNeu-TTS
python -m venv venv
source venv/bin/activate
pip install -U pip

pip install gradio==4.44.1

git clone https://github.com/pnnbao97/VieNeu-TTS.git
pip install -e .


Option 2: The Manual Fix (Edit the Code)
If you want to keep your current Gradio version, you can manually remove the offending argument from the script.

Open /home/henry/Workspace/Rockchip/VieNeu-TTS/VieNeu-TTS/gradio_app.py in your text editor.

Go to line 1050 (as indicated in your traceback).

Look for the gr.Textbox definition. It will look something like this:

Python

status_output = gr.Textbox(
    label="Status",
    show_copy_button=True,  # <--- Remove or comment out this line
    ...
)

python3 gradio_app.py 


--------------------------------------------------------------
mkdir RKNN-LLM
cd RKNN-LLM
wget -c https://mirrors.bfsu.edu.cn/github-release/condaforge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh
chmod 777 Miniforge3-Linux-x86_64.sh

Run bash Miniforge3-Linux-x86_64.sh and install in path = $PWD/env
Every time we open a new console me must activate the env:

> source env/bin/activate

Create a Conda environment named "RKLLM-Toolkit" with Python 3.8 version (recommended version):

> conda create -n RKLLM-Toolkit python=3.8

Activate RKNN-Toolkit:

> conda activate RKLLM-Toolkit

To deactivate:

$ conda deactivate

Clone the rknn-llm from github repo:

> git clone https://github.com/airockchip/rknn-llm.git
> pip3 install rknn-llm/rkllm-toolkit/rkllm_toolkit-1.2.1b1-cp38-cp38-linux_x86_64.whl

----------------------------------------------------------------------------------------

RK3588 linux setup

----------------------------------------------------------------------------------------
mkdir VieNeu-TTS

sudo apt update
sudo apt install -y git cmake g++ make

git clone https://github.com/airockchip/rknn-llm.git
cd rknn-llm/examples/rkllm_api_demo/deploy

edit build-linux.sh as bellow:


C_COMPILER=gcc
CXX_COMPILER=g++
STRIP_COMPILER=strip

./build-linux.sh

# go to output dir
cd build/build_linux_aarch64_Release

# make runtime library visible (option A: LD_LIBRARY_PATH)
export LD_LIBRARY_PATH=../../../../../rkllm-runtime/Linux/librkllm_api/aarch64/:$LD_LIBRARY_PATH

# Adjust debugging log
export RKLLM_LOG_LEVEL=1

# recommended by RKLLM docs
ulimit -HSn 10240

# run your model
./llm_demo ./vieneu_03b_w8a8_rk3588.rkllm 512 2048


-----------------------------------------------------------
NeuCodec on RK3588


Run bash Miniforge3-25.11.0-0-Linux-aarch64.sh and install in path = $PWD/env

Every time we open a new console we must activate the env:

```bash
source env/bin/activate
```

Create a Conda environment named "RKNN-Toolkit2" with Python 3.10 version:

```bash
conda create -n RKNN-Toolkit2 python=3.10
```

Activate RKNN-Toolkit2:

```bash
> conda activate RKNN-Toolkit2
```

To deactivate:

```bash
> conda deactivate
```


sudo apt-get update
sudo apt-get install -y espeak-ng libespeak-ng1



pip install numpy onnxruntime soundfile librosa torch neucodec

pip install -U "huggingface_hub[cli]"

hf download neuphonic/neucodec-onnx-decoder-int8 model.onnx --local-dir neucodec_int8

hf download pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf --local-dir hf_vieneu_03b_q4_gguf




The download onnx model will be in:

neucodec_int8/model.onnx

python3 bench_neucodec_onnx.py


python3 make_ref_codes.py --ref_wav example.wav --out ref_codes.txt


-------------------------------------------------
Debugging


# Fix NPU frequency:

sudo bash ./fix_freq_rk3588.sh


export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=7860
export VIENEU_DEBUG=1

export VIENEU_DEBUG_LLM=1
export VIENEU_DUMP_LLM=1
export VIENEU_DUMP_DIR=./logs
export VIENEU_DEBUG_LOG=./logs/vieneu_debug.log

python3 gradio_app_llmdebug.py



The correct prompt format (what you should generate)

Make the prompt exactly like this:

user: Convert the text to speech:<|TEXT_PROMPT_START|>
<REF_PHONEMES>
<TARGET_PHONEMES>
<|TEXT_PROMPT_END|>
assistant:<|SPEECH_GENERATION_START|>
<REF_CODES_AS_SPEECH_TAGS>


And then you let the model generate new <|speech_x|> codes until it emits <|SPEECH_GENERATION_END|> (or you stop by max tokens).

This formatting matches what you were doing manually with rkllm earlier and is what vieneu/core.py expects.




python3 cpu_llama_onnx_e2e_debug.py \
  --gguf hf_vieneu_03b_q4_gguf/VieNeu-TTS-0_3B-Q4_0.gguf \
  --codec_onnx ./neucodec_int8/model.onnx \
  --ref_codes_pt "./vieneu/assets/samples/Vĩnh (nam miền Nam).pt" \
  --ref_text "Đến cuối thế kỷ 19, ngành đánh bắt cá được thương mại hóa." \
  --text "Trên thực tế, các nghi ngờ đã bắt đầu xuất hiện." \
  --out out_cpu.wav \
  --dump_prefix cpu_cmp2 \
  --force_vi \
  --stop_on_end \
  --max_tokens 1024 \
  --n_ctx 4096

python3 e2e_vieneu_tts_rk3588.py \
  --llm_demo ./llm_demo \
  --rkllm ./vieneu_03b_w8a8_rk3588.rkllm \
  --codec_onnx ./neucodec_int8/model.onnx \
  --ref_codes_pt "./vieneu/assets/samples/Vĩnh (nam miền Nam).pt" \
  --ref_text "Đến cuối thế kỷ 19, ngành đánh bắt cá được thương mại hóa." \
  --text "Trên thực tế, các nghi ngờ đã bắt đầu xuất hiện." \
  --out out.wav \
  --force_vi \
  --stop_on_end \
  --max_new_tokens 1024 \
  --max_context_len 4096 \
  --cap_by_ratio \
  --ratio_pad 0.05 \
  --max_gen_codes 650 \
  --fade_ms 40 \
  --write_debug_wavs

python3 e2e_vieneu_tts_rk3588_pyapi.py \
  --rkllm ./vieneu_03b_w8a8_rk3588.rkllm \
  --rkllmrt ./librkllmrt.so \
  --codec_onnx ./neucodec_int8/model.onnx \
  --ref_codes_pt "./vieneu/assets/samples/Vĩnh (nam miền Nam).pt" \
  --ref_text "Đến cuối thế kỷ 19, ngành đánh bắt cá được thương mại hóa." \
  --text "Ở một vương quốc xa xôi, có một vị vua già có bảy cô con gái công chúa. Bảy nàng công chúa này từng cãi nhau rất nhiều. Vì điều này, nhà vua thường mua quần áo giống nhau cho tất cả họ." \
  --out out.wav \
  --force_vi \
  --stop_on_end \
  --max_new_tokens 1024 \
  --max_context_len 4096 \
  --fade_ms 40 \
  --write_debug_wavs


./llm_demo ./vieneu_03b_w8a8_rk3588.rkllm 2048 2048


python3 e2e_vieneu_tts_rk3588_bk01.py --llm_demo ./llm_demo --rkllm ./vieneu_03b_w8a8_rk3588.rkllm --codec_onnx ./neucodec_int8/model.onnx --ref_codes_pt "./vieneu/assets/samples/Vĩnh (nam miền Nam).pt" --ref_text "Đến cuối thế kỷ 19, ngành đánh bắt cá được thương mại hóa." --out out.wav --force_vi --stop_on_end --max_new_tokens 1024 --max_context_len 4096 --write_debug_wavs --text "Ở một vương quốc xa xôi, có một vị vua già có bảy cô con gái công chúa."


python3 cpu_llama_onnx_e2e_debug.py \
  --gguf hf_vieneu_03b_q4_gguf/VieNeu-TTS-0_3B-Q4_0.gguf \
  --codec_onnx ./neucodec_int8/model.onnx \
  --ref_codes_pt "./vieneu/assets/samples/Đoan (nữ miền Nam).pt" \
  --ref_text "Đến cuối thế kỷ 19, ngành đánh bắt cá được thương mại hóa." \
  --text "Ở một vương quốc xa xôi, có một vị vua già có bảy cô con gái công chúa. Bảy nàng công chúa này từng cãi nhau rất nhiều. Vì điều này, nhà vua thường mua quần áo giống nhau cho tất cả họ." \
  --out out_cpu.wav \
  --dump_prefix cpu_cmp2 \
  --force_vi \
  --stop_on_end \
  --max_tokens 1024 \
  --n_ctx 4096


http://192.168.0.172:7860/

Đến cuối thế kỷ 19, ngành đánh bắt cá được thương mại hóa.

Trên thực tế, các nghi ngờ đã bắt đầu xuất hiện.


export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH



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