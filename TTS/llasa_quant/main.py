import json
import os
import random

import numpy as np
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer, attn, ExLlamaV2Cache
from exllamav2.generator import (
    ExLlamaV2DynamicGenerator,
    ExLlamaV2DynamicJob,
    ExLlamaV2Sampler,
)
from jinja2 import Template
import torch
import torchaudio
import transformers
import soundfile as sf
from huggingface_hub import snapshot_download
from pyexpat.errors import messages
from tqdm import tqdm
from xcodec2.modeling_xcodec2 import XCodec2Model
import whisper

WP = None

def extract_word_timestamps(audio_in):
    global WP
    if WP is None:
        WP = whisper.load_model("turbo")
    result = WP.transcribe(audio_in, word_timestamps=True, verbose=None)
    word_segments = []
    for segment in result["segments"]:
        ws = ((w["word"], w["start"], w["end"]) for w in segment["words"])
        word_segments.extend(ws)
    return word_segments


def ids_to_speech_tokens(speech_ids):
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str


def extract_speech_ids(speech_tokens_str):
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]

            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids


def read_text(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        return ' '.join([line.strip() for line in lines])


def get_model():
    torch.cuda.empty_cache()
    model_path = snapshot_download("Annuvin/Llasa-8B-8.0bpw-h8-exl2")
    config = ExLlamaV2Config(model_path)
    config.max_seq_len = 2048
    model = ExLlamaV2(config, lazy_load=True)
    cache = ExLlamaV2Cache(model, lazy=True)
    paged = attn.has_flash_attn
    model.load_autosplit(cache, callback=lambda x,y: print(f'Loaded {x}/{y}'))
    tokenizer = ExLlamaV2Tokenizer(config, lazy_init=True)
    generator = ExLlamaV2DynamicGenerator(model, cache, tokenizer, paged=paged)
    template = Template(
        generator.tokenizer.tokenizer_config_dict.get("chat_template", "")
    )
    gen_settings = ExLlamaV2Sampler.Settings.greedy()
    gen_settings.allow_tokens(
        tokenizer=generator.tokenizer,
        tokens=[generator.tokenizer.single_id("<|SPEECH_GENERATION_END|>")]+
               list(range(generator.tokenizer.single_id("<|s_0|>"), generator.tokenizer.single_id("<|s_65535|>")+1))
    )
    gen_settings.temperature = 0.8
    gen_settings.top_p = 1.0
    gen_settings.token_repetition_penalty = 1.0
    gen_settings.top_k = 50

    print("Model loaded")
    return {"generator": generator, "template": template, "gen_settings": gen_settings}


def geet_codec_model():
    codec_model = XCodec2Model.from_pretrained("hkustaudio/xcodec2")
    print("XCodec2 model loaded")
    codec_model.eval().to("cuda")
    return {"codec_model": codec_model}

def tts(sample_audio_path, prompt_text, target_text, regen, *, generator, template, gen_settings, codec_model):
    waveform, sample_rate = torchaudio.load(sample_audio_path)

    if waveform.size(0) > 1 or sample_rate != 16000:
        raise ValueError(
            "Only 16kHz mono audio is supported, please resample the audio, you can use resample_audio function")

    assert prompt_text is not None, "Prompt text is required for TTS"
    prompt_wav, sr = sf.read(sample_audio_path)  # English prompt
    prompt_wav = torch.from_numpy(prompt_wav).float().unsqueeze(0)

    input_text = prompt_text.strip() + " " + target_text.strip()

    with torch.no_grad():
        vq_code_prompt = codec_model.encode_code(input_waveform=prompt_wav)
        vq_code_prompt = vq_code_prompt[0,0,:]
        speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)

        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

        # Tokenize the text and the speech prefix
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
        ]
        inp = template.render(messages=chat)
        input_ids = generator.tokenizer.encode(inp, add_bos=True)[:,:-1]
        max_new_tokens = 2048 - input_ids.shape[-1]
        job = ExLlamaV2DynamicJob(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=2,
            gen_settings=gen_settings,
            seed=random.randint(128, 65535) if regen > 0 else None,
            stop_conditions=[generator.tokenizer.single_id("<|SPEECH_GENERATION_END|>")],
        )
        generator.enqueue(job)
        output = []
        while generator.num_remaining_jobs():
            for res in generator.iterate():
                txt = res.get("text")
                if txt:
                    output.append(txt)
        if not output:
            raise ValueError("No output from generator")
        output = extract_speech_ids(output)
        speech_tokens = torch.tensor(output).cuda().unsqueeze(0).unsqueeze(0)
        gen_wav = codec_model.decode_code(speech_tokens)
        # gen_wav = gen_wav[:,:,prompt_wav.shape[1]:]
    return gen_wav[0,0,:].cpu().numpy()

def save_audio(gen_wav, output_path, sample_rate=16000):
    sf.write(output_path, gen_wav, sample_rate)

def save_word_timestamps(word_segments, output_path):
    with open(output_path, "w") as f:
        json.dump(word_segments, f)

def is_broken_tts(audio_in, target_text, sample_rate=16000):
    try:
        word_segments = extract_word_timestamps(audio_in)
        rx = lambda w: 0.6 + 0.12*len(w)
        # remove whisper hallucination (short words)
        word_segments = [w for w in word_segments if w[2] - w[1] > 0.1]
        # Check if end of last word is more than N second before the end of the audio
        audio_len_s = len(audio_in) / sample_rate
        last_word_end = word_segments[-1][2]
        if audio_len_s - last_word_end > 3.0:
            print(f"last word end is {audio_len_s - last_word_end} seconds before the end of the audio")
            return True, None
        if word_segments[-1][1] < audio_len_s - 3.0 - rx(word_segments[-1][0]):
            print(f"last word `{word_segments[-1][0]}` start is {word_segments[-1][1]}({audio_len_s}) seconds")
            return True, None
        # Check if the beginning of the first word is more than N seconds after the start of the audio
        first_word_start = word_segments[0][1]
        if first_word_start > 2.5:
            print(f"fist word start is {first_word_start} seconds after the start of the audio")
            return True, None
        if word_segments[0][2] > 2.5 + rx(word_segments[0][0]):
            print(f"first word `{word_segments[0][0]}` end is {word_segments[0][2]} seconds")
            return True, None
        # Check if the gap between 2 consecutive words is more than N seconds
        for i in range(1, len(word_segments)):
            if word_segments[i][1] - word_segments[i-1][2] > 1.5:
                print(f"gap between {word_segments[i-1][0]} and {word_segments[i][0]} is {word_segments[i][1] - word_segments[i-1][2]} seconds")
                return True, None
        # Check if the duration of one word is more than X+N*len(word) seconds (except for the first and last word)
        for w in word_segments[1:-1]:
            if w[2] - w[1] > rx(w) + 1.5:
                print(f"word {w[0]} duration is {w[2] - w[1]} seconds")
                return True, None
        # Check if the wpm is too extreme
        slow_talk_rpm = 100
        fast_talk_rpm = 250
        wpm = len(word_segments) / (word_segments[-1][2] - word_segments[0][1]) * 60
        if not slow_talk_rpm < wpm < fast_talk_rpm:
            print(f"wpm is {wpm}, which is not between {slow_talk_rpm} and {fast_talk_rpm}")
            return True, None
    except Exception as e:
        print(f"Error: {e}")
        return True, None
    return False, word_segments


def main():
    resources_base = os.path.join(os.path.dirname(__file__), "..", "resources")
    models = get_model()
    models.update(geet_codec_model())
    sample_audio_path = os.path.join(resources_base, "new_intro2.wav")  # Seems to work better with this sample
    prompt_text = read_text(os.path.join(resources_base, "new_intro.txt"))
    output_base = os.path.join(resources_base, "new_out2")
    os.makedirs(output_base, exist_ok=True)
    with open(os.path.join(resources_base, "..", "..", "output", "text.json"), "r") as f:
        data = json.load(f)
    sentences = [d[0].strip() for d in data["sentences"]]
    max_retry = 5
    final_out = []
    word_ts_list = []
    with tqdm(total=len(sentences), desc="Processing Sentences") as pbar:
        for sentence in sentences:
            for attempt in range(max_retry + 1):  # Try max_retry times
                output = tts(sample_audio_path, prompt_text, sentence, regen=attempt, **models)
                flag, word_timestamps = is_broken_tts(output, sentence)
                if not flag:
                    final_out.append(output)
                    word_ts_list.append(word_timestamps)
                    pbar.update(1)  # Only update tqdm if successful
                    break
            else:
                pbar.close()  # Ensure tqdm closes before raising error
                raise RuntimeError(f"TTS failed after {max_retry} retries for: {sentence}")
    # Concatenate all the generated speech waveforms ([T]) to a single waveform
    for i, (a_out, wts) in enumerate(zip(final_out, word_ts_list)):
        a_out_path = os.path.join(output_base, f"output_{i}.wav")
        wts_out_path = os.path.join(output_base, f"output_{i}.json")
        save_audio(a_out, a_out_path)
        save_word_timestamps(wts, wts_out_path)


if __name__ == "__main__":
    main()
