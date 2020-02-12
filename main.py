from process import from_image, train_or_load_model
import os

TRAIN_DATASET_PATH = '.' + os.path.sep + 'dataset'  + os.path.sep

train_image_paths = []
for image_name in os.listdir(TRAIN_DATASET_PATH):
    if '.jpg' in image_name:
        train_image_paths.append(os.path.join(TRAIN_DATASET_PATH, image_name))
       
model = train_or_load_model(train_image_paths)

VALIDATION_DATASET_PATH=os.path.abspath("validation")
VALIDATION_DATASET_PATH+=os.path.sep
for image_path in os.listdir(VALIDATION_DATASET_PATH):
    from_image(model, VALIDATION_DATASET_PATH+image_path)
