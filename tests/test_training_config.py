"""Smoke tests for training configuration and API compatibility."""
import tempfile
from pathlib import Path

import pytest
from transformers import TrainingArguments

from swe_scaffold.config import LoRAConfig, TrainingConfig


def test_training_config_has_eval_strategy():
    """Verify TrainingConfig includes eval_strategy field."""
    config = TrainingConfig()
    
    assert hasattr(config, "eval_strategy")
    assert config.eval_strategy in ["no", "steps", "epoch"]


def test_training_args_accepts_eval_strategy():
    """Verify TrainingArguments accepts eval_strategy parameter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with Transformers 4.57.1 API
        args = TrainingArguments(
            output_dir=tmpdir,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            per_device_train_batch_size=1,
            num_train_epochs=1,
        )
        
        assert args.eval_strategy == "steps"
        assert args.save_strategy == "steps"


def test_lora_config_defaults():
    """Verify LoRAConfig has sensible defaults."""
    config = LoRAConfig()
    
    assert config.r == 8
    assert config.alpha == 16
    assert config.dropout == 0.05
    assert "q_proj" in config.target_modules
    assert "v_proj" in config.target_modules


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
