# threestudio/utils/config.py
from typing import Dict, Any, Union, Type, Optional
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, asdict
from .registry import (
    Registry,
    MODEL_REGISTRY,
    INVERSION_REGISTRY,
    SYSTEM_REGISTRY,
    SDEDIT_REGISTRY,
)

@dataclass
class BaseConfig:
    """Base class for all configurations."""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class Configurable:
    """Base class for configurable objects."""
    def __init__(self, cfg: BaseConfig):
        self.cfg = cfg

def instantiate_from_config(
    cfg: Union[DictConfig, Dict[str, Any]],
    registry: Registry,
    **kwargs
) -> Any:
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    else:
        cfg = cfg.copy()

    if "_target_" in cfg:
        name = cfg.pop("_target_")
    elif "type" in cfg:
        name = cfg.pop("type")
    else:
        raise ValueError("Config must contain '_target_' or 'type' key")
    
    # Pass kwargs to registry.instantiate so they can be merged with cfg
    return registry.instantiate(name, cfg, **kwargs)


def get_obj_from_config(cfg: DictConfig, **kwargs) -> Any:
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    if "type" in config_dict:
        obj_type = config_dict["type"]
        
        for registry in [MODEL_REGISTRY, INVERSION_REGISTRY, SYSTEM_REGISTRY, SDEDIT_REGISTRY]:
            try:
                cls = registry.get(obj_type)
                return instantiate_from_config(cfg, registry, **kwargs)
            except KeyError:
                continue
        
        raise ValueError(f"Unknown object type: {obj_type}")
    
    raise ValueError("Config must contain 'type' key")