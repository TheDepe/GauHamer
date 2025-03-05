import numpy as np

path='/graphics/scratch2/students/perrettde/MODELS/GausHamer/.cache/HaMeR_Obj/data/mano_mean_params_manus.npz'

cam_data = np.load(path)

for key, val in cam_data.items():
    print(key, val)
    
    
#CACHE_DIR = os.path.join('/graphics/scratch2/students/perrettde/MODELS/GausHamer/', ".cache")
#CACHE_DIR_HAMER = os.path.join(CACHE_DIR, "HaMeR") # RENAME
#DEFAULT_CHECKPOINT = f'{CACHE_DIR_HAMER}/hamer_ckpts/checkpoints/hamer.ckpt'
path='/graphics/scratch2/students/perrettde/MODELS/GausHamer/.cache/HaMeR/data/mano_mean_params_manus.npz'

cam_data = np.load(path)

for key, val in cam_data.items():
    print(key, val)