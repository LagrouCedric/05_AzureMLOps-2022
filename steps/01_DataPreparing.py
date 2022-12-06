from ctypes import resize
from glob import glob
import json
import os
from datetime import datetime
import math
import random
import shutil

from utils import connectWithAzure

import cv2
from dotenv import load_dotenv
from azureml.core import Dataset
from azureml.data.datapath import DataPath



# When you work locally, you can use a .env file to store all your environment variables.
# This line read those in.
load_dotenv()

OPTIONS = os.environ.get('OPTIONS').split(',')
SEED = int(os.environ.get('RANDOM_SEED'))
TRAIN_TEST_SPLIT_FACTOR = float(os.environ.get('TRAIN_TEST_SPLIT_FACTOR'))

def processAndUploadAnimalImages(datasets, data_path, processed_path, ws, options_name):

    # We can't use mount on these machines, so we'll have to download them

    option_path = os.path.join(data_path, 'options', options_name)

    # Get the dataset name for this animal, then download to the directory
    datasets[options_name].download(option_path, overwrite=True) # Overwriting means we don't have to delete if they already exist, in case something goes wrong.
    print('Downloading all the images')

    # Get all the image paths with the `glob()` method.
    print(f'Resizing all images for {options_name} ...')
    image_paths = glob(f"{option_path}/*.png") # CHANGE THIS LINE IF YOU NEED TO GET YOUR options_nameS IN THERE IF NEEDED!

    # Process all the images with OpenCV. Reading them, then resizing them to 64x64 and saving them once more.
    print(f"Processing {len(image_paths)} images")
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (64, 64)) # Resize to a square of 64, 64
        cv2.imwrite(os.path.join(processed_path, options_name, image_path.split('/')[-1]), image)
    print(f'... done resizing. Stopping context now...')
    
    # Upload the directory as a new dataset
    print(f'Uploading directory now ...')
    resized_dataset = Dataset.File.upload_directory(
                        # Enter the sourece directory on our machine where the resized pictures are
                        src_dir = os.path.join(processed_path, options_name),
                        # Create a DataPath reference where to store our images to. We'll use the default datastore for our workspace.
                        target = DataPath(datastore=ws.get_default_datastore(), path_on_datastore=f'processed_options/{options_name}'),
                        overwrite=True)

    print('... uploaded images, now creating a dataset ...')

    # Make sure to register the dataset whenever everything is uploaded.
    new_dataset = resized_dataset.register(ws,
                            name=f'resized_{options_name}',
                            description=f'{options_name} images resized tot 64, 64',
                            tags={'options': options_name, 'AI-Model': 'CNN', 'GIT-SHA': os.environ.get('GIT_SHA')}, # Optional tags, can always be interesting to keep track of these!
                            create_new_version=True)
    print(f" ... Dataset id {new_dataset.id} | Dataset version {new_dataset.version}")
    print(f'... Done. Now freeing the space by deleting all the images, both original and processed.')
    emptyDirectory(option_path)
    print(f'... done with the original images ...')
    emptyDirectory(os.path.join(processed_path, options_name))
    print(f'... done with the processed images. On to the next Animal, if there are still!')

def emptyDirectory(directory_path):
    shutil.rmtree(directory_path)

def prepareDataset(ws):
    data_folder = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_folder, exist_ok=True)

    for options_name in OPTIONS:
        os.makedirs(os.path.join(data_folder, 'options', options_name), exist_ok=True)

    # Define a path to store the animal images onto. We'll choose for `data/processed/animals` this time. Again, create subdirectories for all the animals
    processed_path = os.path.join(os.getcwd(), 'data', 'processed', 'options')
    os.makedirs(processed_path, exist_ok=True)
    for options_name in OPTIONS:
        os.makedirs(os.path.join(processed_path, options_name), exist_ok=True)

    datasets = Dataset.get_all(workspace=ws) # Make sure to give our workspace with it
    for options_name in OPTIONS:
        processAndUploadAnimalImages(datasets, data_folder, processed_path, ws, options_name)

def trainTestSplitData(ws):

    training_datapaths = []
    testing_datapaths = []
    default_datastore = ws.get_default_datastore()
    for options_name in OPTIONS:
        # Get the dataset by name
        option_dataset = Dataset.get_by_name(ws, f"resized_{options_name}")
        print(f'Starting to process {options_name} images.')

        # Get only the .JPG images
        option_images = [img for img in option_dataset.to_path() if img.split('.')[-1] == 'png']

        print(f'... there are about {len(option_images)} images to process.')

        ## Concatenate the names for the options_name and the img_path. Don't put a / between, because the img_path already contains that
        option_images = [(default_datastore, f'processed_options/{options_name}{img_path}') for img_path in option_images] # Make sure the paths are actual DataPaths
        
        random.seed(SEED) # Use the same random seed as I use and defined in the earlier cells
        random.shuffle(option_images) # Shuffle the data so it's randomized
        
        ## Testing images
        amount_of_test_images = math.ceil(len(option_images) * TRAIN_TEST_SPLIT_FACTOR) # Get a small percentage of testing images

        option_test_images = option_images[:amount_of_test_images]
        option_training_images = option_images[amount_of_test_images:]
        
        # Add them all to the other ones
        testing_datapaths.extend(option_test_images)
        training_datapaths.extend(option_training_images)

        print(f'We already have {len(testing_datapaths)} testing images and {len(training_datapaths)} training images, on to process more animals if necessary!')

    training_dataset = Dataset.File.from_files(path=training_datapaths[:200]+training_datapaths[-200:])
    testing_dataset = Dataset.File.from_files(path=training_datapaths[:200]+training_datapaths[-200:])

    training_dataset = training_dataset.register(ws,
        name=os.environ.get('TRAIN_SET_NAME'), # Get from the environment
        description=f'The option Images to train, resized tot 64, 64',
        tags={'options': os.environ.get('ANIMALS'), 'AI-Model': 'CNN', 'Split size': str(1 - TRAIN_TEST_SPLIT_FACTOR), 'type': 'training', 'GIT-SHA': os.environ.get('GIT_SHA')},
        create_new_version=True)

    print(f"Training dataset registered: {training_dataset.id} -- {training_dataset.version}")

    testing_dataset = testing_dataset.register(ws,
        name=os.environ.get('TEST_SET_NAME'), # Get from the environment
        description=f'The option Images to test, resized tot 64, 64',
        tags={'options': os.environ.get('ANIMALS'), 'AI-Model': 'CNN', 'Split size': str(TRAIN_TEST_SPLIT_FACTOR), 'type': 'testing', 'GIT-SHA': os.environ.get('GIT_SHA')},
        create_new_version=True)

    print(f"Testing dataset registered: {testing_dataset.id} -- {testing_dataset.version}")

def main():
    ws = connectWithAzure()

    print('Processing the images')
    prepareDataset(ws)
    
    print('Splitting the images')
    trainTestSplitData(ws)

if __name__ == '__main__':
    main()