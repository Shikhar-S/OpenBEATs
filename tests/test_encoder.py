import torch

from openbeats.beats_encoder import BeatsEncoder
from openbeats.model import OpenBeats

TINY = {"encoder_layers": 2, "encoder_embed_dim": 64,
        "encoder_ffn_embed_dim": 128, "encoder_attention_heads": 4}


def test_variable_length_masking():
    """A tiny encoder runs and the padding mask shortens the padded clip."""
    torch.manual_seed(0)
    enc = BeatsEncoder(input_size=0, beats_config=TINY, is_pretraining=False).eval()
    assert enc.output_size() == 64

    T = 16000
    xs = torch.randn(2, T)
    ilens = torch.tensor([T, T // 2])  # one full clip, one half-length
    with torch.no_grad():
        rep, olens, _ = enc(xs, ilens, waveform_input=True)

    assert rep.shape[0] == 2 and rep.shape[2] == 64
    assert olens[0] > olens[1]  # shorter input -> fewer valid patches


def test_chunked_encoding():
    """chunk_seconds yields patch embeddings (num_patches, D) like a single pass."""
    torch.manual_seed(0)
    enc = BeatsEncoder(input_size=0, beats_config=TINY, is_pretraining=False).eval()
    model = OpenBeats(enc)  # cpu, no classifier
    wav = torch.randn(16000 * 5).numpy()  # 5 s

    whole = model.encode(wav)["patch_embeddings"]
    chunked = model.encode(wav, chunk_seconds=2)["patch_embeddings"]
    assert whole.ndim == 2 and whole.shape[1] == 64
    assert chunked.ndim == 2 and chunked.shape[1] == 64
    assert chunked.shape[0] > 0
