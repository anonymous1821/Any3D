from typing import Dict, Type, Any, Optional, Union, Callable, get_type_hints
from dataclasses import is_dataclass, fields, asdict
import inspect
import warnings
from dataclasses import make_dataclass, field

class Registry:
    def __init__(self, name: str, base_class: Optional[Type] = None):
        self._name = name
        self._obj_map: Dict[str, Type] = {}
        self._base_class = base_class
        self._config_schemas: Dict[str, Any] = {} 
    
    def register(
        self, 
        name: Optional[str] = None,
        config_class: Optional[Type] = None
    ):
        def decorator(cls):
            nonlocal name
            if name is None:
                name = cls.__name__
            
            if self._base_class and not issubclass(cls, self._base_class):
                raise TypeError(
                    f"{cls.__name__} must be a subclass of {self._base_class.__name__}"
                )
            
            if name in self._obj_map:
                raise KeyError(f"'{name}' already registered in '{self._name}'")
            
            self._obj_map[name] = cls
            
            if config_class:
                self._config_schemas[name] = config_class
            elif hasattr(cls, 'Config'):
                self._config_schemas[name] = cls.Config
            else:
                self._config_schemas[name] = self._create_config_class(cls)
            
            return cls
        
        return decorator
    
    def _create_config_class(self, cls: Type) -> Type:
        init_signature = inspect.signature(cls.__init__)
        parameters = list(init_signature.parameters.values())[1:]
        
        
        fields = []
        for param in parameters:
            field_name = param.name
            
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            if param.default == inspect.Parameter.empty:
                fields.append((field_name, param.annotation))
            else:
                fields.append((field_name, param.annotation, 
                             field(default=param.default)))
        
        config_cls = make_dataclass(
            f"{cls.__name__}Config",
            fields,
            namespace={
                "__doc__": f"Configuration for {cls.__name__}"
            }
        )
        
        return config_cls
    
    def get_class(self, name: str) -> Type:
        if name not in self._obj_map:
            available = ", ".join(self._obj_map.keys())
            raise KeyError(
                f"'{name}' not found in registry '{self._name}'. "
                f"Available: {available}"
            )
        return self._obj_map[name]
    
    def get_config_class(self, name: str) -> Optional[Type]:
        return self._config_schemas.get(name)
    
    def instantiate(
        self, 
        name: str, 
        cfg: Union[Dict, 'DictConfig'], 
        **kwargs
    ) -> Any:
        cls = self.get_class(name)
        config_cls = self.get_config_class(name)
        
        if hasattr(cfg, '_metadata'): 
            from omegaconf import OmegaConf
            cfg = OmegaConf.to_container(cfg, resolve=True)
        
        config_dict = {**cfg, **kwargs}
        
        try:
            if config_cls and is_dataclass(config_cls):
                # Filter arguments for config class
                field_names = {f.name for f in fields(config_cls)}
                filtered = {k: v for k, v in config_dict.items() if k in field_names}
                
                config_instance = config_cls(**filtered)
                
                if hasattr(config_instance, 'validate'):
                    config_instance.validate()
                
                # Check if the class expects a config object
                # We check if the first parameter (after self) is named 'cfg' 
                # or annotated with the config class
                sig = inspect.signature(cls.__init__)
                params = list(sig.parameters.values())
                # Any kwargs from the original cfg/kwargs that are not part of the
                # config dataclass fields should be considered extra and may be
                # intended for the target class constructor (e.g., `model=`).
                extra = {k: v for k, v in config_dict.items() if k not in field_names}

                if len(params) > 1 and (params[1].name == 'cfg' or params[1].annotation == config_cls):
                    # Pass config instance as first arg and forward matching extra kwargs
                    # Filter extra to only those parameters the constructor accepts
                    ctor_param_names = {p.name for p in params[2:]}  # skip self and cfg
                    ctor_kwargs = {k: v for k, v in extra.items() if k in ctor_param_names}
                    return cls(config_instance, **ctor_kwargs)

                # Otherwise, unpack as kwargs (legacy support) and forward matching extra kwargs
                ctor_param_names = {p.name for p in params[1:]}  # skip self
                ctor_kwargs = {k: v for k, v in {**asdict(config_instance), **extra}.items() if k in ctor_param_names}
                return cls(**ctor_kwargs)
            else:
                return cls(**config_dict)
                
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate '{name}' from registry '{self._name}': {e}"
            ) from e

    def list_available(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        for name, cls in self._obj_map.items():
            config_cls = self._config_schemas.get(name)
            result[name] = {
                "class": cls,
                "config_class": config_cls,
                "config_schema": self._get_config_schema(config_cls),
                "docstring": cls.__doc__
            }
        return result
    
    def _get_config_schema(self, config_cls: Optional[Type]) -> Dict[str, Any]:
        if not config_cls:
            return {}
        
        schema = {}
        if is_dataclass(config_cls):
            for field in fields(config_cls):
                schema[field.name] = {
                    "type": str(field.type),
                    "default": field.default if field.default != inspect.Parameter.empty else "REQUIRED",
                    "description": field.metadata.get("description", "")
                }
        
        return schema

MODEL_REGISTRY = Registry('model')
INVERSION_REGISTRY = Registry('inversion')
SYSTEM_REGISTRY = Registry('system')
SDEDIT_REGISTRY = Registry('sdedit')
EVALUATOR_REGISTRY = Registry('evaluator')
RENDERER_REGISTRY = Registry('renderer')
GUIDANCE_REGISTRY = Registry('guidance')