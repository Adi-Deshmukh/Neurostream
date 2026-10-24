from pathlib import Path

import pytest

from neurostream.configuration import load_config


ROOT = Path(__file__).parents[1]


def test_default_configuration_is_valid():
    config = load_config(ROOT / "configs" / "config.yaml")
    assert config.dataset_name == "BNCI2014_001"
    assert config.sampling_frequency_hz == 250.0
    assert config.encoder_timesteps == 20
    assert config.encoder_channel_count == 22
    assert config.output_checkpoint_dir.name == "checkpoints"


def test_configuration_rejects_invalid_filter_cutoffs(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        (ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")
        .replace("filter_h_freq_hz: 100.0", "filter_h_freq_hz: 130.0"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Nyquist"):
        load_config(path)


def test_configuration_rejects_missing_required_section(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("data: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="preprocessing"):
        load_config(path)


def test_configuration_rejects_full_validation_split(tmp_path):
    path = tmp_path / "config.yaml"
    text = (ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")
    path.write_text(text.replace("validation_fraction: 0.2", "validation_fraction: 1.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="less than 1"):
        load_config(path)
