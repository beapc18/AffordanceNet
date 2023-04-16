import os
import sys
import pickle
import time
from PIL import Image
import numpy as np

from config_files.config_iit_train import ConfigIitTrain

cfg = ConfigIitTrain()

affordances = np.array([0])

if __name__ == '__main__':
    # TODO: create GTsegmask_VOC_2012_train_images if it doesn't exist
    print(cfg.DATA_MASKS_DIR)
    list_dir = os.listdir(cfg.DATA_MASKS_DIR)
    print(list_dir)
    np.set_printoptions(threshold=sys.maxsize)
    for file in list_dir:
        # read only mask files
        if os.path.isfile(os.path.join(cfg.DATA_MASKS_DIR, file)):
            with open(os.path.join(cfg.DATA_MASKS_DIR, file), 'rb') as f:
                mask_im = pickle.load(f)
                print("asdfökljsdaölkjföalsdfölsdakjasf")
                time.sleep(10)
                for elem in mask_im.flatten():
                    if elem != 0:
                        print(elem)
                

            # save as image
            im = Image.fromarray(np.array(mask_im))
            array = np.array(im)
            for elem in array.flatten():
                if elem != 0:
                    print(elem)

            mask_path = os.path.join(cfg.DATA_MASKS_DIR_IMGS, file).split('.')[0]
            print(mask_path)
            im.save(mask_path + '.png')