import shutil
import os
import time
import numpy as np
from PIL import Image
from io import BytesIO

from typing import Optional, List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

########## CONSTS ##########
SAVE_IMAGE = False
HEIGHT = 300
WIDTH = 300
DATASET_DIR = './datasets/'
############################
app = FastAPI()
if not os.path.isdir('./datasets'):
    os.mkdir('./datasets')

########################## UPLOAD ############################

# Helper functions

def get_dataset_id():
    return str(int(time.time()*1000))

def read_imagefile(data) -> Image.Image:
    image = Image.open(BytesIO(data))
    return image

def array_to_image(dataset_folder, image, file):
    transformed_image = Image.fromarray(image)
    transformed_image.save(DATASET_DIR + dataset_folder + '/' + file.filename)

def is_image_files(files):
    is_valid = True
    for file in files:
        # finding last token which comes after '.'
        file_ext = file.filename.split('.')[-1].lower()
        
        # checking for file type
        if not (file_ext in ['png', 'jpg', 'jpeg']):
           is_valid = False

    return is_valid

def transform(image):
    # resizing
    image = Image.fromarray(image)
    image = image.resize((WIDTH, HEIGHT))
    image = np.array(image)
    
    return image

# api route
@app.post("/upload/")
async def upload(files: List[UploadFile] = File(...)):
    if not is_image_files(files):
        return {
            'success' : False,
            'message' : 'Please upload only png, jpg, jpeg files.'
            }
    dataset_id = get_dataset_id()    
    os.mkdir(DATASET_DIR + dataset_id)
    data_array = []
    for file in files:
        # convering to numpy array
        image = np.array(read_imagefile(await file.read()))

        # transformation
        image = transform(image)

        # adding to numpy array
        data_array.append(image)

        # for saving image file to dataset
        if SAVE_IMAGE:
            array_to_image(dataset_id, image, file)

    # saving to npz file
    data_array = np.array(data_array)
    np.savez_compressed(DATASET_DIR + dataset_id + '/' + dataset_id, data_array)

    return {
            'success' : True,
            'message':'dataset created successully',
            'dataset_id': dataset_id
            }

##############################################################
########################## DOWNLOAD ##########################
@app.get('/download/{id}')
async def download(id):
    file_path = DATASET_DIR + id + '/' + id + '.npz'
    file_name = id + '.npz'
    
    # checking if file exist or not
    if os.path.isfile(file_path):
        return FileResponse(path=file_path, media_type='application/octet-stream', filename=file_name)
    else:
        return {
            'success' : False,
            'message': 'No dataset with that id exist'
            }

##############################################################
