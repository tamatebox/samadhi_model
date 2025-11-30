from typing import Dict, Any, Union
import torch
import torch.nn as nn

from src.components.vicara.base import BaseVicara
from src.components.refiners.base import BaseRefiner
from src.configs.vicara import StandardVicaraConfig
from src.configs.factory import create_vicara_config  # Add import for factory


class StandardVicara(BaseVicara):
    """
    Standard Vicāra.
    Performs purification using a single refiner network.
    """

    def __init__(self, config: StandardVicaraConfig, refiner: BaseRefiner):
        if isinstance(config, dict):
            config = create_vicara_config(config)
        super().__init__(config, nn.ModuleList([refiner]))
        # StandardVicara expects a single refiner, passed as a list to the base class

    def _refine_step(self, s_t: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        # Use the single shared refiner
        return self.refiners[0](s_t)
