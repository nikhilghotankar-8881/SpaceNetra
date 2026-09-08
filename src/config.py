import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


@dataclass
class ProjectMetaConfig:
    name: str = "SpaceNetra"
    version: str = "0.1.0"
    description: str = ""
    author: str = ""


@dataclass
class SystemConfig:
    seed: int = 42
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class PathsConfig:
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(default_factory=lambda: Path("data"))
    raw_data_dir: Path = field(default_factory=lambda: Path("data/raw"))
    processed_data_dir: Path = field(default_factory=lambda: Path("data/processed"))
    levir_cd_dir: Path = field(default_factory=lambda: Path("data/levir_cd"))
    embeddings_dir: Path = field(default_factory=lambda: Path("data/embeddings"))
    checkpoints_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    outputs_dir: Path = field(default_factory=lambda: Path("outputs"))
    predictions_dir: Path = field(default_factory=lambda: Path("outputs/predictions"))
    metrics_dir: Path = field(default_factory=lambda: Path("outputs/metrics"))
    visualizations_dir: Path = field(default_factory=lambda: Path("outputs/visualizations"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))


@dataclass
class ChangeDetectionConfig:
    default_model: str = "siamese_unet"
    input_channels: int = 3
    input_size: List[int] = field(default_factory=lambda: [256, 256])
    patch_size: int = 256
    stride: int = 256
    models: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticRetrievalConfig:
    default_model: str = "remoteclip"
    embedding_dim: int = 512
    top_k: int = 5
    similarity_threshold: float = 0.7


@dataclass
class CredibilityEngineConfig:
    weights: Dict[str, float] = field(default_factory=dict)
    min_confidence_threshold: float = 0.50


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = ""


@dataclass
class AppConfig:
    project: ProjectMetaConfig = field(default_factory=ProjectMetaConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    change_detection: ChangeDetectionConfig = field(default_factory=ChangeDetectionConfig)
    semantic_retrieval: SemanticRetrievalConfig = field(default_factory=SemanticRetrievalConfig)
    credibility_engine: CredibilityEngineConfig = field(default_factory=CredibilityEngineConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "AppConfig":
        """Load configuration from YAML file and apply environment variable overrides."""
        if config_path is None:
            root = Path(__file__).resolve().parent.parent
            config_path = str(root / "configs" / "project_config.yaml")

        path = Path(config_path)
        raw: Dict[str, Any] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

        # Parse sections
        project = ProjectMetaConfig(**raw.get("project", {}))

        system_raw = raw.get("system", {})
        if os.getenv("DEVICE"):
            system_raw["device"] = os.getenv("DEVICE")
        if os.getenv("NUM_WORKERS"):
            system_raw["num_workers"] = int(os.getenv("NUM_WORKERS"))
        if os.getenv("SEED"):
            system_raw["seed"] = int(os.getenv("SEED"))
        system = SystemConfig(**system_raw)

        paths_raw = raw.get("paths", {})
        root_dir = Path(__file__).resolve().parent.parent
        paths = PathsConfig(
            root_dir=root_dir,
            data_dir=root_dir / paths_raw.get("data_dir", "data"),
            raw_data_dir=root_dir / paths_raw.get("raw_data_dir", "data/raw"),
            processed_data_dir=root_dir / paths_raw.get("processed_data_dir", "data/processed"),
            levir_cd_dir=root_dir / paths_raw.get("levir_cd_dir", "data/levir_cd"),
            embeddings_dir=root_dir / paths_raw.get("embeddings_dir", "data/embeddings"),
            checkpoints_dir=root_dir / paths_raw.get("checkpoints_dir", "checkpoints"),
            outputs_dir=root_dir / paths_raw.get("outputs_dir", "outputs"),
            predictions_dir=root_dir / paths_raw.get("predictions_dir", "outputs/predictions"),
            metrics_dir=root_dir / paths_raw.get("metrics_dir", "outputs/metrics"),
            visualizations_dir=root_dir / paths_raw.get("visualizations_dir", "outputs/visualizations"),
            logs_dir=root_dir / paths_raw.get("logs_dir", "logs"),
        )

        cd = ChangeDetectionConfig(**raw.get("change_detection", {}))
        sr = SemanticRetrievalConfig(**raw.get("semantic_retrieval", {}))
        ce = CredibilityEngineConfig(**raw.get("credibility_engine", {}))

        log_raw = raw.get("logging", {})
        if os.getenv("LOG_LEVEL"):
            log_raw["level"] = os.getenv("LOG_LEVEL")
        logging_cfg = LoggingConfig(**log_raw)

        return cls(
            project=project,
            system=system,
            paths=paths,
            change_detection=cd,
            semantic_retrieval=sr,
            credibility_engine=ce,
            logging=logging_cfg,
        )


# Global configuration instance singleton
config = AppConfig.load()
