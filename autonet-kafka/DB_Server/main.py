import shutil
import os
import time
import numpy as np
import pandas as pd
from io import StringIO
import csv
from PIL import Image
import codecs
from io import BytesIO

from typing import Optional, List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import requests
import json

########## CONSTS ##########
SAVE_IMAGE = False
HEIGHT = 64
WIDTH = 64
DATASET_DIR = "./datasets/"
############################
app = FastAPI()
if not os.path.isdir("./datasets"):
    os.mkdir("./datasets")

########################## UPLOAD ############################

# Helper functions


def get_dataset_id():
    return str(int(time.time() * 1000))


def read_imagefile(data) -> Image.Image:
    image = Image.open(BytesIO(data))
    return image


def array_to_image(dataset_folder, image, file):
    transformed_image = Image.fromarray(image)
    transformed_image.save(DATASET_DIR + dataset_folder + "/" + file.filename)


def is_image_files(files):
    is_valid = True
    for file in files:
        # finding last token which comes after '.'
        file_ext = file.filename.split(".")[-1].lower()

        # checking for file type
        if not (file_ext in ["png", "jpg", "jpeg"]):
            is_valid = False

    return is_valid


def is_csv_file(file):
    file_ext = file.filename.split(".")[-1].lower()
    return file_ext == "csv"


def transform(image):
    # resizing
    image = Image.fromarray(image)
    image = image.resize((WIDTH, HEIGHT))
    image = np.array(image)

    return image


# api route
@app.post("/upload/")
async def upload(images: List[UploadFile] = File(...), target: UploadFile = File(...)):
    if not images or not target:
        return {"success": False, "message": "Please upload both images and target"}
    if is_image_files(images) and is_csv_file(target):
        dataset_id = get_dataset_id()
        os.mkdir(DATASET_DIR + dataset_id)
        data_array = []
        for file in images:
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
        target_data = pd.read_csv(
            StringIO(str(target.file.read(), "utf-8")), encoding="utf-8"
        )
        np.savez_compressed(
            DATASET_DIR + dataset_id + "/" + dataset_id, data_array, target_data
        )

        return {
            "success": True,
            "message": "dataset created successully",
            "dataset_id": dataset_id,
        }
    else:
        return {"success": False, "message": "Please upload only png, jpg, jpeg files."}

##############################################################
########################## UPLOAD ############################

@app.post("/upload/{pid}")
async def upload(pid,images: List[UploadFile] = File(...), target: UploadFile = File(...)):

    url = "http://userservice:8002/project/"+str(pid)

    payload={}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)
    responsejson=response.json()

    if responsejson['code']!=200:
        return {"success": False, "message":"wrong project id"}
    # print(responsejson)
    if not images or not target:
        return {"success": False, "message": "Please upload both images and target"}
    if is_image_files(images) and is_csv_file(target):
        dataset_id = get_dataset_id()
        os.mkdir(DATASET_DIR + dataset_id)
        data_array = []
        for file in images:
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
        target_data = pd.read_csv(
            StringIO(str(target.file.read(), "utf-8")), encoding="utf-8"
        )
        np.savez_compressed(
            DATASET_DIR + dataset_id + "/" + dataset_id, data_array, target_data
        )

        url = "http://userservice:8002/project/"+str(pid)

        payload = json.dumps({"dataid": dataset_id})
        headers = {
        'Content-Type': 'application/json'
        }

        response = requests.request("PUT", url, headers=headers, data=payload)

        if response.json()["code"]!=200:
            return{
                "success":False
            }

        return {
            "success": True,
            "message": "dataset created successully",
            "dataset_id": dataset_id,
        }
    else:
        return {"success": False, "message": "Please upload only png, jpg, jpeg files."}


##############################################################
########################## DOWNLOAD ##########################
@app.get("/download/{id}")
async def download(id):
    file_path = DATASET_DIR + id + "/" + id + ".npz"
    file_name = id + ".npz"

    # checking if file exist or not
    if os.path.isfile(file_path):
        return FileResponse(
            path=file_path, media_type="application/octet-stream", filename=file_name
        )
    else:
        return {"success": False, "message": "No dataset with that id exist"}


##############################################################
