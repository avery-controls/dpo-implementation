"""Tests for DPO Implementation"""

import json
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import DPOTrainer, TrainingConfig


@pytest.fixture
def temp_project():
    """Temporary project directory for testing."""
    tmpdir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    os.chdir(tmpdir)

    yield Path(tmpdir)

    os.chdir(original_cwd)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def trainer(temp_project):
    """DPOTrainer instance."""
    return DPOTrainer()


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = TrainingConfig(
            model_name="meta-llama/Llama-3.1-8B",
            dataset_source="hh-rlhf"
        )

        assert config.model_name == "meta-llama/Llama-3.1-8B"
        assert config.dataset_source == "hh-rlhf"
        assert config.lora_rank == 8
        assert config.lora_alpha == 16
        assert config.beta == 0.1
        assert config.learning_rate == 5e-7
        assert config.epochs == 3

    def test_config_from_file(self, temp_project):
        """Test loading config from file."""
        config_data = {
            "model_name": "meta-llama/Llama-3.1-8B",
            "dataset_source": "hh-rlhf",
            "lora_rank": 16,
            "beta": 0.2
        }

        config_file = temp_project / ".dpo_config.json"
        config_file.write_text(json.dumps(config_data))

        config = TrainingConfig.from_file(config_file)
        assert config.model_name == "meta-llama/Llama-3.1-8B"
        assert config.lora_rank == 16
        assert config.beta == 0.2

    def test_config_to_file(self, temp_project):
        """Test saving config to file."""
        config = TrainingConfig(
            model_name="mistralai/Mistral-7B",
            dataset_source="webgpt",
            lora_rank=16
        )

        config_file = temp_project / ".dpo_config.json"
        config.to_file(config_file)

        assert config_file.exists()

        loaded_data = json.loads(config_file.read_text())
        assert loaded_data["model_name"] == "mistralai/Mistral-7B"
        assert loaded_data["lora_rank"] == 16


class TestInitialization:
    """Test DPO project initialization."""

    def test_init_creates_config_file(self, trainer, temp_project):
        """Test that init creates config file."""
        trainer.init(
            model_name="meta-llama/Llama-3.1-8B",
            dataset_source="hh-rlhf"
        )

        config_file = temp_project / ".dpo_config.json"
        assert config_file.exists()

    def test_init_creates_requirements(self, trainer, temp_project):
        """Test that init creates requirements file."""
        trainer.init(
            model_name="meta-llama/Llama-3.1-8B",
            dataset_source="hh-rlhf"
        )

        req_file = temp_project / "dpo_requirements.txt"
        assert req_file.exists()

        content = req_file.read_text()
        assert "trl>=0.7.0" in content
        assert "peft>=0.6.0" in content
        assert "bitsandbytes>=0.41.0" in content

    def test_init_creates_train_script(self, trainer, temp_project):
        """Test that init creates training script."""
        trainer.init(
            model_name="meta-llama/Llama-3.1-8B",
            dataset_source="hh-rlhf"
        )

        train_script = temp_project / "dpo_train.py"
        assert train_script.exists()

        content = train_script.read_text()
        assert "DPOTrainer" in content
        assert "LoraConfig" in content
        assert "args.epochs" in content


class TestDatasetPreparation:
    """Test dataset preparation."""

    def test_prepare_creates_data_dir(self, trainer, temp_project):
        """Test that prepare-dataset creates data directory."""
        assert trainer.data_dir.exists()

    def test_prepare_creates_output_dir(self, trainer, temp_project):
        """Test that prepare-dataset creates output directory."""
        assert trainer.output_dir.exists()

    def test_prepare_validates_source(self, trainer, temp_project, capsys):
        """Test that prepare-dataset validates dataset source."""
        # Use invalid source
        with patch('sys.exit'):
            trainer.prepare_dataset(source="invalid_source")

        captured = capsys.readouterr()
        assert "Unknown dataset source" in captured.out


class TestConfiguration:
    """Test configuration management."""

    def test_status_with_config(self, trainer, temp_project, capsys):
        """Test status with valid config."""
        # Create config
        config_data = {
            "model_name": "meta-llama/Llama-3.1-8B",
            "dataset_source": "hh-rlhf",
            "lora_rank": 8,
            "epochs": 3
        }

        config_file = temp_project / ".dpo_config.json"
        config_file.write_text(json.dumps(config_data))

        trainer.status()

        captured = capsys.readouterr()
        assert "DPO Implementation Status" in captured.out
        assert "meta-llama/Llama-3.1-8B" in captured.out
        assert "hh-rlhf" in captured.out

    def test_status_without_config(self, trainer, temp_project, capsys):
        """Test status without config."""
        trainer.status()

        captured = capsys.readouterr()
        assert "Not set" in captured.out

    def test_status_shows_directories(self, trainer, temp_project, capsys):
        """Test status shows directory information."""
        # Create data file
        (trainer.data_dir / "hh-rlhf_sample.jsonl").write_text("{}")

        trainer.status()

        captured = capsys.readouterr()
        assert "Data directory" in captured.out
        assert "hh-rlhf_sample.jsonl" in captured.out


class TestQLoRAParameters:
    """Test QLoRA parameter handling."""

    def test_default_qlora_params(self):
        """Test default QLoRA parameters."""
        config = TrainingConfig(
            model_name="test",
            dataset_source="test"
        )

        assert config.lora_rank == 8
        assert config.lora_alpha == 16
        assert config.quantization_bits == 4

    def test_custom_qlora_params(self):
        """Test custom QLoRA parameters."""
        config = TrainingConfig(
            model_name="test",
            dataset_source="test",
            lora_rank=16,
            lora_alpha=32,
            quantization_bits=8
        )

        assert config.lora_rank == 16
        assert config.lora_alpha == 32
        assert config.quantization_bits == 8


class TestDPOParameters:
    """Test DPO parameter handling."""

    def test_default_dpo_params(self):
        """Test default DPO parameters."""
        config = TrainingConfig(
            model_name="test",
            dataset_source="test"
        )

        assert config.beta == 0.1
        assert config.learning_rate == 5e-7
        assert config.epochs == 3

    def test_custom_dpo_params(self):
        """Test custom DPO parameters."""
        config = TrainingConfig(
            model_name="test",
            dataset_source="test",
            beta=0.2,
            learning_rate=1e-6,
            epochs=5
        )

        assert config.beta == 0.2
        assert config.learning_rate == 1e-6
        assert config.epochs == 5


class TestTrainingParameters:
    """Test training parameter handling."""

    def test_default_training_params(self):
        """Test default training parameters."""
        config = TrainingConfig(
            model_name="test",
            dataset_source="test"
        )

        assert config.batch_size == 4
        assert config.max_length == 512
        assert config.output_dir == "./dpo_output"

    def test_custom_training_params(self):
        """Test custom training parameters."""
        config = TrainingConfig(
            model_name="test",
            dataset_source="test",
            batch_size=8,
            max_length=1024,
            output_dir="./custom_output"
        )

        assert config.batch_size == 8
        assert config.max_length == 1024
        assert config.output_dir == "./custom_output"


class TestIntegration:
    """Integration tests."""

    def test_full_init_workflow(self, trainer, temp_project):
        """Test full initialization workflow."""
        # Init
        trainer.init(
            model_name="meta-llama/Llama-3.1-8B",
            dataset_source="hh-rlhf"
        )

        # Check all files created
        assert (temp_project / ".dpo_config.json").exists()
        assert (temp_project / "dpo_requirements.txt").exists()
        assert (temp_project / "dpo_train.py").exists()

        # Check config
        config = TrainingConfig.from_file(temp_project / ".dpo_config.json")
        assert config.model_name == "meta-llama/Llama-3.1-8B"
        assert config.dataset_source == "hh-rlhf"

        # Check requirements
        req_content = (temp_project / "dpo_requirements.txt").read_text()
        assert "trl>=0.7.0" in req_content

        # Check training script
        train_content = (temp_project / "dpo_train.py").read_text()
        assert "DPOTrainer" in train_content

    def test_config_persistence(self, trainer, temp_project):
        """Test config persistence across operations."""
        # Init with custom config
        trainer.init(
            model_name="mistralai/Mistral-7B",
            dataset_source="webgpt"
        )

        # Reload
        trainer2 = DPOTrainer()
        assert trainer2.config.model_name == "mistralai/Mistral-7B"
        assert trainer2.config.dataset_source == "webgpt"

        # Update config
        trainer2.config.lora_rank = 16
        trainer2.config.to_file(temp_project / ".dpo_config.json")

        # Reload again
        trainer3 = DPOTrainer()
        assert trainer3.config.lora_rank == 16
