#from .humman import HuMMan
from .mirage import MIRAGE
from .manus import Manus

def get_dataset(cfg, name):
    dataset_type = cfg.data.dataset_type
    if dataset_type=="MIRAGE":
        return MIRAGE(cfg, name)
    if dataset_type=="MANUS":
        return Manus(cfg, name)
    else:
        raise ValueError(f"Dataset {dataset_type} does not exist")