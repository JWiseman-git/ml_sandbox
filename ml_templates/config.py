# src/classifier/config.py
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    random_state: int = 42


@dataclass
class TrainingConfig:
    test_size: float = 0.2
    cv_folds: int = 5
    model: ModelConfig = field(default_factory=ModelConfig)
    artifact_dir: Path = Path("artifacts")