"""
This is a high level abstract wrapper for rectified flow model, to unify 
the interface of different rectified flow models (both 2D and 3D).
"""
from utils.config import instantiate_from_config, MODEL_REGISTRY
from omegaconf import OmegaConf, DictConfig 
class RFModel:

    @classmethod
    def from_config(cls, cfg: DictConfig, **kwargs):
        model_cfg = cfg.get("model", {})
        
        if isinstance(model_cfg, DictConfig):
            model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
        
        model = instantiate_from_config(model_cfg, MODEL_REGISTRY, **kwargs)
        return model

    def __init__(self):
        pass 

    def to(self):
        """
        Move all the tensors, models and others to the desired device 
        """
        pass 
    def get_v_prediction(self):
        """
        Return the velocity prediction given required input (x_t, t, cond etc.)
        """
        pass 
    def sample(self):
        """
        Run the full pipeline for generation/editing single pass 
        """
        pass 
    def get_schedule(self):
        """
        Return the timesteps, sigmas of sampling 
        """
        pass 
    def set_prompt(self):
        """
        Set condition (text, image etc.)
        """
        pass 
    def encode(self):
        """
        Encode native representation (e.g. image) into latent 
        """
        pass 
    def decode(self):
        """
        Decode latent into native representation (e.g. image)
        """