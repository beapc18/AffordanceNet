# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {} #global param

from utils.datasets.pascal_voc import pascal_voc
# from lib.datasets.coco import coco

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        for dataset in ['iit', 'umd']:
            name = 'voc_{}_{}_{}'.format(year, split, dataset)
            __sets[name] = (lambda split=split, year=year, dataset=dataset: pascal_voc(split, year, dataset))

# # Set up coco_2014_<split>
# for year in ['2014']:
#     for split in ['train', 'val', 'minival', 'valminusminival']:
#         name = 'coco_{}_{}'.format(year, split)
#         __sets[name] = (lambda split=split, year=year: coco(split, year))
#
# # Set up coco_2015_<split>
# for year in ['2015']:
#     for split in ['test', 'test-dev']:
#         name = 'coco_{}_{}'.format(year, split)
#         __sets[name] = (lambda split=split, year=year: coco(split, year))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
