# path to compile libdarknet.so file
darknet_libso_path = '/path/to/darknet/libdark.so'

# model requirements path
model_req = {
    'mask':{
        'configPath': '/path/to/darknet/cfg/yolov4-mask.cfg', # Path to mask model config
        'weightPath': '/home/rishab/repos/darknet/backup_mask/yolov4-mask_last.weights', # Path to trained weights
        'metaPath': '/home/rishab/repos/darknet/data/mask.data', # Path to meta info file
    },
    'helmet':{
        'configPath': '/home/rishab/repos/darknet/cfg/yolov4-helmet.cfg', # Path to helmet model config
        'weightPath': '/home/rishab/repos/darknet/backup_helmet/yolov4-helmet_last.weights', # Path to trained weights
        'metaPath': '/home/rishab/repos/darknet/data/helmet.data', # Path to meta info file
    }
}

# path to image dataset
img_dir = 'images'

# path to ground truth xml
xml_annotation_path = 'path/to/annotations.xml'

# directory path to parsed ground truth to yolo format
labels_save_dir = {
                    'helmet': 'save/path/to/labels-helmet', 
                    'mask': 'save/path/to/labels-mask'
                    }
# directory path to parsed ground truth to evaluation format
gt_save_dir = {
                'helmet': 'save/path/to/gt-helmet', 
                'mask': 'save/path/to/gt-mask'
                }
# directory path to parsed ground truth to evaluation format
pred_save_dir = {
                 'helmet': 'save/path/to/pred-helmet', 
                 'mask': 'save/path/to/pred-mask'
                }
# supported image format
SUPPORTED_IMG_FORMATS = ['.jpg', 'jpeg', '.png']

# labels parsing meta information
labels = {
    'helmet':{
        'index':{ # class label index
            'yes': 0, 
            'no': 1
        },
        'path_train': 'save/path/to/helmet_train.txt', # path to train labels
        'path_val': 'save/path/to/helmet_val.txt', # path to validation labels
        'path_test': 'save/path/to/helmet_test.txt', # path to test labels
        'train_split': 0.9, # train split
        'validation_split': 0.05, # validation split
        'test_split': 0.05 # test split
    },
    'mask':{
        'index':{ # class label index
            'yes': 0,
            'no': 1,
            'invisible': 2,
            'wrong': 3
        },
        'path_train': 'save/path/to/mask_train.txt', # path to train labels
        'path_val': 'save/path/to/mask_val.txt', # path to validation labels
        'path_test': 'save/path/to/mask_test.txt', # path to test labels
        'train_split': 0.9, # train split
        'validation_split': 0.05, # validation split
        'test_split': 0.05 # test split
    }
}