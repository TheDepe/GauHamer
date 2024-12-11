#from .humman import HuMMan
from .mirage import MIRAGE

def get_dataset(cfg, name):
    dataset_type = cfg.data.dataset_type
    if dataset_type=="MIRAGE":
        return MIRAGE(cfg, name)
    else:
        raise ValueError(f"Dataset {dataset_type} does not exist")