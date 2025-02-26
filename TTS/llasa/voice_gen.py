import os
import json

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import soundfile as sf
from xcodec2.modeling_xcodec2 import XCodec2Model
import torchaudio


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


def resample_audio(audio_path, output_path, target_sr=16000):
    waveform, sample_rate = torchaudio.load(audio_path)
    # Check if the audio is stereo (i.e., has more than one channel)
    if waveform.size(0) > 1:
        # Convert stereo to mono by averaging the channels
        waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
    else:
        # If already mono, just use the original waveform
        waveform_mono = waveform
    waveform_16k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(waveform_mono)
    torchaudio.save(output_path, waveform_16k, target_sr)


def save_audio(gen_wav, output_path, sample_rate=16000):
    sf.write(output_path, gen_wav, sample_rate)


def text_to_speech(sample_audio_path, prompt_text, target_text,
                   *, model, tokenizer, xcodec2_model):
    waveform, sample_rate = torchaudio.load(sample_audio_path)

    if waveform.size(0) > 1 or sample_rate != 16000:
        raise ValueError("Only 16kHz mono audio is supported, please resample the audio, you can use resample_audio function")

    assert prompt_text is not None, "Prompt text is required for TTS"
    prompt_wav, sr = sf.read(sample_audio_path)  # English prompt
    prompt_wav = torch.from_numpy(prompt_wav).float().unsqueeze(0)

    input_text = prompt_text.strip() + " " + target_text.strip()
    # TTS start!
    with torch.no_grad():
        # Encode the prompt wav
        vq_code_prompt = xcodec2_model.encode_code(input_waveform=prompt_wav)
        # print("Prompt Vq Code Shape:", vq_code_prompt.shape)

        vq_code_prompt = vq_code_prompt[0, 0, :]
        # Convert int 12345 to token <|s_12345|>
        speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)

        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

        # Tokenize the text and the speech prefix
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
        ]

        input_ids = tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors='pt',
            continue_final_message=True
        )
        input_ids = input_ids.to('cuda')
        speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

        # Generate the speech autoregressively
        outputs = model.generate(
            input_ids,
            max_length=2048,  # We trained our model with a max length of 2048
            eos_token_id=speech_end_id,
            do_sample=True,
            top_p=1,
            temperature=0.8,
        )
        # Extract the speech tokens
        generated_ids = outputs[0][input_ids.shape[1] - len(speech_ids_prefix):-1]

        speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Convert  token <|s_23456|> to int 23456
        speech_tokens = extract_speech_ids(speech_tokens)

        speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)

        # Decode the speech tokens to speech waveform
        gen_wav = xcodec2_model.decode_code(speech_tokens)

        # if only need the generated part
        gen_wav = gen_wav[:,:,prompt_wav.shape[1]:]

    return gen_wav[0,0,:].cpu().numpy()


def get_models():
    llasa = 'HKUST-Audio/Llasa-3B'
    # llasa = 'HKUST-Audio/Llasa-8B'
    xcodec2_model_path = 'HKUST-Audio/xcodec2'
    tokenizers = AutoTokenizer.from_pretrained(llasa)
    model = AutoModelForCausalLM.from_pretrained(llasa)
    model.eval()
    model.to('cuda')
    xcodec2_model = XCodec2Model.from_pretrained(xcodec2_model_path)
    xcodec2_model.eval().cuda()
    return {'model': model, 'tokenizer': tokenizers, 'xcodec2_model': xcodec2_model}


def read_text(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        return ' '.join([line.strip() for line in lines])


def main():
    resources_base = os.path.join(os.path.dirname(__file__), "..", "resources")
    models = get_models()
    sample_audio_path = os.path.join(resources_base, f"intro2.wav")  # Seems to work better with this sample
    prompt_text = read_text(os.path.join(resources_base, "intro.txt"))
    with open(os.path.join(os.path.dirname(__file__), "..", "..", "output", "text.json"), "r") as f:
        data = json.load(f)
    sentences = [d[0].strip() for d in data["sentences"]]
    output_path = os.path.join(resources_base, f"output_all.wav")
    final_out = []
    for sentence in tqdm(sentences, desc="Processing Sentences"):
        output = text_to_speech(sample_audio_path, prompt_text, sentence, **models)
        final_out.append(output)
    # Concatenate all the generated speech waveforms ([T]) to a single waveform
    final_out = np.concatenate(final_out)
    save_audio(final_out, output_path)


if __name__ == "__main__":
    main()
