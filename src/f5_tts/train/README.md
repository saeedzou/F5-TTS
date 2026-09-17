# Training

Check your FFmpeg installation:
```bash
ffmpeg -version
```
If not found, install it first (or skip assuming you know of other backends available).

## Prepare Dataset

Example data processing scripts, and you may tailor your own one along with a Dataset class in `src/f5_tts/model/dataset.py`.

### 1. Some specific Datasets preparing scripts
Download corresponding dataset first, and fill in the path in scripts.

```bash
# Prepare the Emilia dataset
python src/f5_tts/train/datasets/prepare_emilia.py

# Prepare the Wenetspeech4TTS dataset
python src/f5_tts/train/datasets/prepare_wenetspeech4tts.py

# Prepare the LibriTTS dataset
python src/f5_tts/train/datasets/prepare_libritts.py

# Prepare the LJSpeech dataset
python src/f5_tts/train/datasets/prepare_ljspeech.py
```

### 2. Create custom dataset with CSV
Prepare a CSV with two columns using a required header: `audio_file|text`. Audio paths must be absolute.
Use guidance see [#57 here](https://github.com/SWivid/F5-TTS/discussions/57#discussioncomment-10959029).

```bash
python src/f5_tts/train/datasets/prepare_csv_wavs.py /path/to/metadata.csv /path/to/output
```

### 3. Create custom dataset with JSONL manifest
Prepare a JSONL manifest with one object per line. Each object must contain
`audio_filepath` (an absolute audio path), `duration` (in seconds), and `text`.

For BPE training, this preparation step and BPE training are independent: the
manifest preparation stores the original text, and the model applies BPE when
it batches text for training or inference. You may train BPE before or after
this step, but it must be available before model training starts.

```json
{"audio_filepath": "/path/to/audio.wav", "duration": 2.5, "text": "Hello."}
```

```bash
python src/f5_tts/train/datasets/prepare_manifest.py /path/to/manifest.jsonl /path/to/output
```

Use `--pretrain` to build `vocab.txt` from the manifest text instead of using the pretrained vocabulary.

### 4. Train a BPE tokenizer
Train a SentencePiece BPE tokenizer directly from the `text` field in the JSONL manifest.
Normalization is disabled (`identity`) and byte fallback is disabled.

When using BPE, the practical order is to train this tokenizer first, then run
`prepare_manifest.py`, and finally start model training. `prepare_manifest.py`
does not convert text into character or BPE IDs.

```bash
python src/f5_tts/train/datasets/train_bpe.py \
	/path/to/manifest.jsonl /path/to/tokenizer \
	--vocab-size 4096
```

The command writes `/path/to/tokenizer.model` and `/path/to/tokenizer.vocab`.
Use `--help` for corpus sampling, sentence length, character coverage, and reserved-symbol options.

## Training & Finetuning

Once your datasets are prepared, you can start the training process.

For BPE training, set `model.tokenizer: bpe` and
`model.tokenizer_path: /path/to/tokenizer.model` in the YAML config. The
dataset directory must use the corresponding `_bpe` suffix, for example
`data/MyDataset_bpe`.

### 1. Training script used for pretrained model

```bash
# setup accelerate config, e.g. use multi-gpu ddp, fp16
# will be to: ~/.cache/huggingface/accelerate/default_config.yaml     
accelerate config

# .yaml files are under src/f5_tts/configs directory
accelerate launch src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml

# possible to overwrite accelerate and hydra config
accelerate launch --mixed_precision=fp16 src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml ++datasets.batch_size_per_gpu=19200
```

### 2. Finetuning practice
Discussion board for Finetuning [#57](https://github.com/SWivid/F5-TTS/discussions/57).

Gradio UI training/finetuning with `src/f5_tts/train/finetune_gradio.py` see [#143](https://github.com/SWivid/F5-TTS/discussions/143).

If want to finetune with a variant version e.g. *F5TTS_v1_Base_no_zero_init*, manually download pretrained checkpoint from model weight repository and fill in the path correspondingly on web interface.

If use tensorboard as logger, install it first with `pip install tensorboard`.

<ins>The `use_ema = True` might be harmful for early-stage finetuned checkpoints</ins> (which goes just few updates, thus ema weights still dominated by pretrained ones), try turn it off with finetune gradio option or `load_model(..., use_ema=False)`, see if offer better results.

### 3. W&B Logging

The `wandb/` dir will be created under path you run training/finetuning scripts.

By default, the training script does NOT use logging (assuming you didn't manually log in using `wandb login`).

To turn on wandb logging, you can either:

1. Manually login with `wandb login`: Learn more [here](https://docs.wandb.ai/ref/cli/wandb-login)
2. Automatically login programmatically by setting an environment variable: Get an API KEY at https://wandb.ai/authorize and set the environment variable as follows:

On Mac & Linux:

```
export WANDB_API_KEY=<YOUR WANDB API KEY>
```

On Windows:

```
set WANDB_API_KEY=<YOUR WANDB API KEY>
```
Moreover, if you couldn't access W&B and want to log metrics offline, you can set the environment variable as follows:

```
export WANDB_MODE=offline
```
