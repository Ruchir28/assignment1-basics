from dataclasses import dataclass, asdict
import yaml

@dataclass
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000.0

@dataclass
class DataConfig:
    batch_size: int = 32
    train_data_path: str = "path/to/your/train_data.npy"
    val_data_path: str = "path/to/your/val_data.npy"

@dataclass
class AdamWConfig:
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.01
    eps: float = 1e-8

@dataclass
class TrainingConfig:
    total_epochs: int = 10
    max_l2_norm: float = 1.0
    lr: float = 1e-4
    min_lr: float = 1e-6
    warmup_steps: int = 100
    eval_interval: int = 500
    save_interval: int = 1000
    log_interval: int = 10
    checkpoint_path: str = "checkpoints/"
    optimizer: AdamWConfig = AdamWConfig()

@dataclass
class FullConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig

def save_config(config: FullConfig, path: str):
    with open(path, 'w') as f:
        yaml.dump(asdict(config), f)

def load_config(path: str) -> FullConfig:
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return FullConfig(
        model=ModelConfig(**config_dict['model']),
        data=DataConfig(**config_dict['data']),
        training=TrainingConfig(
            **{k: v for k, v in config_dict['training'].items() if k != 'optimizer'},
            optimizer=AdamWConfig(**config_dict['training']['optimizer'])
        )
    )
