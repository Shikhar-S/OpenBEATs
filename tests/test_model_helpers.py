import torch

from openbeats.utils import _derive_base_repo, build_classifier, encoder_state_dict


def test_derive_base_repo():
    assert _derive_base_repo("espnet/OpenBEATS-Large-i1-as20k") == "shikhar7ssu/OpenBEATs-Large-i1"
    assert _derive_base_repo("espnet/OpenBEATS-Base-i3-bats") == "shikhar7ssu/OpenBEATs-Base-i3"
    # ESPnet beats_ckpt_path style (iter0 -> i1)
    assert _derive_base_repo("ear_large/beats_iter0_large.tune/epoch59.pt") == \
        "shikhar7ssu/OpenBEATs-Large-i1"
    assert _derive_base_repo("something-unrelated") is None


def test_encoder_state_dict_strips_wrapper_prefix():
    direct = {"patch_embedding.weight": torch.zeros(1)}
    assert encoder_state_dict(direct) is direct  # already unwrapped

    wrapped = {
        "encoder.patch_embedding.weight": torch.zeros(1),
        "encoder.layer_norm.weight": torch.zeros(1),
        "decoder.linear_out.weight": torch.zeros(1),  # head stays out of encoder
    }
    out = encoder_state_dict(wrapped)
    assert set(out) == {"patch_embedding.weight", "layer_norm.weight"}


def test_build_classifier():
    sd = {"decoder.linear_out.weight": torch.randn(5, 8),
          "decoder.linear_out.bias": torch.randn(5)}
    head = build_classifier(sd, in_features=8)
    assert head is not None and head.out_features == 5

    # input dim must match the encoder, else it's not a class head (e.g. MLM decoder)
    assert build_classifier(sd, in_features=16) is None
    assert build_classifier({}, in_features=8) is None
