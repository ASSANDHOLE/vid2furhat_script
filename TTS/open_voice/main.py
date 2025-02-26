import json
import os
from os.path import join as oj
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

from melo.api import TTS


BASE_DIR = os.path.dirname(__file__)

def main():
    ckpt_converter = oj(BASE_DIR, 'checkpoints_v2', 'converter')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = oj(BASE_DIR, 'outputs_v2')

    tone_color_converter = ToneColorConverter(oj(ckpt_converter, 'config.json'), device=device)
    tone_color_converter.watermark_model = None
    tone_color_converter.load_ckpt(oj(ckpt_converter, 'checkpoint.pth'))

    os.makedirs(output_dir, exist_ok=True)

    resources_base = oj(BASE_DIR, '..', 'resources')
    ref_voice_path = oj(resources_base, 'intro2.wav')
    target_se, audio_name = se_extractor.get_se(ref_voice_path, tone_color_converter, vad=True)
    with open(oj(BASE_DIR, "..", "..", "output", "text.json"), "r") as f:
        data = json.load(f)
    sentences = ' '.join([d[0].strip() for d in data["sentences"]])
    # output_path = oj(resources_base, f"output_opv.wav")
    lan_code = 'EN_NEWEST'
    tmp_path = oj(output_dir, 'tmp.wav')
    speed = 1.0
    model = TTS(language=lan_code, device=device)
    speaker_ids = model.hps.data.spk2id
    for speaker_key in speaker_ids.keys():
        speaker_id = speaker_ids[speaker_key]
        speaker_key = speaker_key.lower().replace('_', '-')
        source_se = torch.load(oj(BASE_DIR, 'checkpoints_v2', 'base_speakers', 'ses', f'{speaker_key}.pth'), map_location=device)
        model.tts_to_file(sentences, speaker_id, tmp_path, speed=speed)
        save_path = oj(output_dir, f'output_v2_{speaker_key}.wav')
        tone_color_converter.convert(
            audio_src_path=tmp_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=save_path
        )


if __name__ == "__main__":
    main()