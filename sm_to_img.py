import os
import pickle
from PIL import Image
import numpy as np

from config_files.config_iit_train import ConfigIitTrain

cfg = ConfigIitTrain()

affordances = np.array([0])

if __name__ == '__main__':
    # TODO: create GTsegmask_VOC_2012_train_images if it doesn't exist
    print(cfg.DATA_MASKS_DIR)
    list_dir = os.listdir(cfg.DATA_MASKS_DIR)
    for file in list_dir:
        # read only mask files
        if os.path.isfile(os.path.join(cfg.DATA_MASKS_DIR, file)):
            with open(os.path.join(cfg.DATA_MASKS_DIR, file), 'rb') as f:
                mask_im = pickle.load(f)

            # save as image
            im = Image.fromarray(np.array(mask_im))

            mask_path = os.path.join(cfg.DATA_MASKS_DIR_IMGS, file).split('.')[0]
            im.save(mask_path + '.png')