import os
import shutil
import glob
import urllib.request
import tarfile
import re

def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes = 90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())

MODELS_CONFIG = {
    'ssd_mobilenet_v2': {
        'model_name': 'ssd_mobilenet_v2_coco_2018_03_29',
        'pipeline_file': 'ssd_mobilenet_v2_coco.config',
        'batch_size': 12
    },
    'faster_rcnn_inception_v2': {
        'model_name': 'faster_rcnn_inception_v2_coco_2018_01_28',
        'pipeline_file': 'faster_rcnn_inception_v2_pets.config',
        'batch_size': 12
    },
    'rfcn_resnet101': {
        'model_name': 'rfcn_resnet101_coco_2018_01_28',
        'pipeline_file': 'rfcn_resnet101_pets.config',
        'batch_size': 8
    }
}

num_steps = 1000
num_eval_steps = 50

selected_model = 'ssd_mobilenet_v2'

MODEL = MODELS_CONFIG[selected_model]['model_name']

pipeline_file = MODELS_CONFIG[selected_model]['pipeline_file']

batch_size = MODELS_CONFIG[selected_model]['batch_size']

import os
os.environ['PYTHONPATH'] += ':/content/models/research/:/content/models/research/slim/'

test_record_fname = 'annotations/test.record'
train_record_fname = 'annotations/train.record'
label_map_pbtxt_fname = 'annotations/label_map.pbtxt'

MODEL_FILE = MODEL + '.tar.gz'

#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
DEST_DIR = 'model/'

#tar = tarfile.open(MODEL_FILE)
#tar.extractall()
#tar.close()

pipeline_fname = os.path.join('configs/', pipeline_file)

fine_tune_checkpoint = os.path.join(DEST_DIR, "model.ckpt")

num_classes = get_num_classes(label_map_pbtxt_fname)
with open(pipeline_fname) as f:
    s = f.read()
with open(pipeline_fname, 'w') as f:
    # fine_tune_checkpoint
    s = re.sub('fine_tune_checkpoint: ".*?"',
               'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)
    
    # tfrecord files train and test.
    s = re.sub(
        '(input_path: ".*?)(train.record)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
    s = re.sub(
        '(input_path: ".*?)(val.record)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

    # label_map_path
    s = re.sub(
        'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

    # Set training batch_size.
    s = re.sub('batch_size: [0-9]+',
               'batch_size: {}'.format(batch_size), s)

    # Set training steps, num_steps
    s = re.sub('num_steps: [0-9]+',
               'num_steps: {}'.format(num_steps), s)
    
    # Set number of classes num_classes.
    s = re.sub('num_classes: [0-9]+',
               'num_classes: {}'.format(num_classes), s)
    f.write(s)


model_dir = 'training/'
#either i create the directory, or have it created
#os.makedirs(model_dir, exist_ok=True)

# !python model_main.py \
#     --pipeline_config_path={pipeline_fname} \
#     --model_dir={model_dir} \
#     --alsologtostderr \
#     --num_train_steps={num_steps} \
#     --num_eval_steps={num_eval_steps}

