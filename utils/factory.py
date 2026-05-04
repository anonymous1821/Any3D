# threestudio/utils/factory.py
from typing import Dict, Any, Union, Optional
from omegaconf import DictConfig, OmegaConf
from .registry import (
    MODEL_REGISTRY, 
    INVERSION_REGISTRY,
    SYSTEM_REGISTRY,
    SDEDIT_REGISTRY, 
    EVALUATOR_REGISTRY,
    RENDERER_REGISTRY,
    GUIDANCE_REGISTRY,
    Registry
)
from .import_utils import load_all_registries

def instantiate_from_config(
    cfg: Union[DictConfig, Dict[str, Any]],
    registry: Optional[Registry] = None,
    target_key: str = "_target_",
    **kwargs
) -> Any:
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    
    if target_key not in cfg:
        for key in ["type", "name", "_name"]:
            if key in cfg:
                target = cfg.pop(key)
                break
        else:
            raise ValueError(
                f"Config must contain one of: {target_key}, 'type', 'name', '_name'"
            )
    else:
        target = cfg.pop(target_key)
    
    if registry is None:
        registry = _select_registry_by_target(target)
    
    return registry.instantiate(target, cfg, **kwargs)


def _select_registry_by_target(target: str) -> Registry:
    for registry in [MODEL_REGISTRY, INVERSION_REGISTRY, SYSTEM_REGISTRY, SDEDIT_REGISTRY, EVALUATOR_REGISTRY]:
        try:
            registry.get_class(target)
            return registry
        except KeyError:
            continue
    
    raise ValueError(f"Cannot find target '{target}' in any registry")


def instantiate_model(cfg: Union[DictConfig, Dict[str, Any]], **kwargs):
    load_all_registries()
    return instantiate_from_config(cfg, MODEL_REGISTRY, **kwargs)

def instantiate_inversion(cfg: Union[DictConfig, Dict[str, Any]], **kwargs):
    load_all_registries()
    return instantiate_from_config(cfg, INVERSION_REGISTRY, **kwargs)

def instantiate_system(cfg: Union[DictConfig, Dict[str, Any]], **kwargs):
    load_all_registries()
    return instantiate_from_config(cfg, SYSTEM_REGISTRY, **kwargs)

def instantiate_sdedit(cfg: Union[DictConfig, Dict[str, Any]], **kwargs):
    load_all_registries()
    return instantiate_from_config(cfg, SDEDIT_REGISTRY, **kwargs)

def instantiate_evaluator(cfg: Union[DictConfig, Dict[str, Any]], **kwargs):
    load_all_registries()
    return instantiate_from_config(cfg, EVALUATOR_REGISTRY, **kwargs)

def instantiate_renderer(cfg: Union[DictConfig, Dict[str, Any]], **kwargs):
    load_all_registries()
    return instantiate_from_config(cfg, RENDERER_REGISTRY, **kwargs)

def instantiate_guidance(cfg: Union[DictConfig, Dict[str, Any]], **kwargs):
    load_all_registries()
    return instantiate_from_config(cfg, GUIDANCE_REGISTRY, **kwargs)