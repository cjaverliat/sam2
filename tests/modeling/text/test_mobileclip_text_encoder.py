import torch
from sam.modeling.text.tokenizer import Sam3Tokenizer
from sam.modeling.text.mobileclip_text_encoder import MobileClipTextEncoder


def test_mobileclip_text_encoder_contract():
    tok = Sam3Tokenizer()
    enc = MobileClipTextEncoder(tokenizer=tok, variant="MobileCLIP-S0",
                                context_length=16, output_dim=256).eval()
    mask, memory, embeds = enc(["dog", "a red car"], device=torch.device("cpu"))
    # contract matches Sam3TextEncoder: (seq, batch, ...) layout, projected to output_dim
    assert memory.shape[-1] == 256
    assert memory.shape[1] == 2  # batch
    assert mask.dtype == torch.bool


def test_encode_returns_language_features():
    tok = Sam3Tokenizer()
    enc = MobileClipTextEncoder(tokenizer=tok, variant="MobileCLIP-S0",
                                context_length=16, output_dim=256).eval()
    feats = enc.encode(["dog"])
    assert feats.shape[-1] == 256
