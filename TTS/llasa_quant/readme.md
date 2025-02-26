# Install Deps

Create a python 3.12 env:
```sh
conda create -n <name> python=3.12
conda activate <name>
```

Install dependencies, ignore any `xcodec2` errors:
```sh
pip install -r requirements.txt
pip install xcodec2 --no-deps
```

### Also see
- [Llasa 8B](https://huggingface.co/HKUSTAudio/Llasa-8B)
- [Llasa 8B Quant](https://huggingface.co/Annuvin/Llasa-8B-8.0bpw-h8-exl2)
- [Llasa Web UI (Inspired by)](https://github.com/Zuellni/LLaSA-WebUI)