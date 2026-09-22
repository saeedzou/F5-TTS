"""
Resize a pretrained checkpoint's text embedding to a new vocab size (e.g. for BPE).

The old and new vocabularies don't share row indices (BPE ids are unrelated to
pinyin/char ids), so the embedding table can't be copied over -- it's just
re-initialized at the new size. Everything else in the checkpoint (attention,
conv, mel layers) is left untouched and still gets to warm-start from the
pretrained weights.

Usage:
    python resize_pretrained_vocab.py pretrained.safetensors resized.safetensors --vocab-size 4096
"""

import argparse

import torch


def resize(ckpt_path, out_path, new_vocab_size):
    is_safetensors = ckpt_path.endswith(".safetensors")
    if is_safetensors:
        from safetensors.torch import load_file, save_file

        sd = load_file(ckpt_path, device="cpu")
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = ckpt["ema_model_state_dict"]

    resized = 0
    for key in list(sd.keys()):
        if key.endswith("text_embed.text_embed.weight"):
            old = sd[key]
            new = torch.empty(new_vocab_size + 1, old.shape[1])  # +1 for filler token, see list_str_to_idx()
            torch.nn.init.normal_(new)
            sd[key] = new
            resized += 1
            print(f"resized {key}: {tuple(old.shape)} -> {tuple(new.shape)}")

    if resized == 0:
        raise RuntimeError("No text_embed.text_embed.weight key found; check the checkpoint format.")

    if is_safetensors:
        save_file(sd, out_path)
    else:
        torch.save(ckpt, out_path)
    print(f"Saved: {out_path}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path")
    parser.add_argument("out_path")
    parser.add_argument("--vocab-size", type=int, required=True, help="New (BPE) vocab size, e.g. sp.get_piece_size()")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    resize(args.ckpt_path, args.out_path, args.vocab_size)