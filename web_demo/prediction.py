import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import urllib
import argparse
from flask import Flask, request, jsonify, make_response, render_template,\
 send_from_directory
from flask_restful import Api, Resource


parser = argparse.ArgumentParser(description="Provide paths")
parser.add_argument(
    'main_path',
     type=str,
     help="Path to your app")
parser.add_argument(
    'label_path',
    type=str,
    help="Path to where your index to label class is")
parser.add_argument(
    'static_path',
    type=str,
    help="Path to where your static folder which contains images is")

args = parser.parse_args()
main_path = args.main_path
label_path = args.label_path
static_path = args.static_path

model = torchvision.models.resnet152(pretrained=True)
model.eval()

app = Flask(__name__)
api = Api(app)

with open(label_path) as f:
    label_dictionary = eval(f.read())

# Function that takes in bytes image file, read it and transform it
def transform_image(img_bytes):
    transformation = transforms.Compose(
        [
            transforms.Resize(
                (256, 256)), transforms.ToTensor(), transforms.Normalize(
                mean=[
                    0.485, 0.456, 0.406], std=[
                        0.229, 0.224, 0.225])]) #Transformation to be applied

    img = Image.open(io.BytesIO(img_bytes))

    return transformation(img).unsqueeze(0)

#The front page of our webapp
class Index(Resource):
    def get(self):
        return make_response(render_template("upload.html"))

#Endpoint for viewing images only
class ViewImage(Resource):
    def post(self):
        target_path = static_path
        filenames_list = []
        num_images = 0

        if not os.path.isdir(target_path):
            os.mkdir(target_path)

        #This for loop allows us to process more than one given image
        for file in request.files.getlist("file"):  #The file here is an object
            file_name = file.filename
            filenames_list.append(file_name)
            num_images += 1

        return make_response(
            render_template(
                "view.html",
                img_list=filenames_list,
                num=num_images))

#Endpoint for image classification from link provided
class ImageFromLink(Resource):
    def post(self):
        if request.form:
            url = request.form.get('url')
            file_name = request.form.get('filename')
            target_path = static_path

            if not os.path.isdir(target_path):
                os.mkdir(target_path)

            img_path = target_path + '/' + file_name
            #This saves the image from the url to the img_path
            urllib.request.urlretrieve(url, img_path)

            with open(img_path, 'rb') as file:
                img_bytes = file.read()

            img = transform_image(img_bytes)
            model_output = model(img)
            label = torch.argmax(model_output, dim=1)
            label = str(label.item())
            class_ = str(label_dictionary.get(int(label)))
            message = "This picture contains a " + class_ + " with index " \
            + label

            return make_response(
                render_template(
                    "image_link.html",
                    img=file_name,
                    label=message))

#Endpoint for image classification with image shown
class GetPredictionWithImage(Resource):
    def post(self):
        if request.files:
            target_path = static_path
            img_list = []
            label_list = []
            num_images = 0

            for file in request.files.getlist("file"):
                file_name = file.filename
                file_name_path = "/".join([target_path, file_name])

                with open(file_name_path, 'rb') as file:
                    img_bytes = file.read()

                img = transform_image(img_bytes)
                print(np.asarray(img).shape)
                model_output = model(img)
                label = torch.argmax(model_output, dim=1)
                label = str(label.item())
                class_ = str(label_dictionary.get(int(label)))
                message = "This picture contains a " + class_ + " with index " \
                +label
                img_list.append(file_name)
                label_list.append(message)
                num_images += 1

            return make_response(
                render_template(
                    'complete.html',
                    img_list=img_list,
                    label_list=label_list,
                    num=num_images))

        else:
            return make_response(jsonify("Image not received"), 404)

#Endpoint for image classification without image shown
class GetPrediction(Resource):
    def post(self):
        #So that file_name can be passed in as an argument using ?file
        if request.args:
            #This checks if the right query parameter is used
            if "file" not in request.args.keys():
                return make_response(
                    jsonify("the query parameter is file"), 404)

            else:
                target_path = static_path
                file_name = request.args.get('file')
                file_path = "/".join([target_path, file_name])

                with open(file_path, 'rb') as file:
                    img_bytes = file.read()

                img = transform_image(img_bytes)
                model_output = model(img)
                label = torch.argmax(model_output, dim=1)
                label = str(label.item())
                class_ = str(label_dictionary.get(int(label)))
                message = "This picture contains a " + class_ + " with index " \
                +label

                return jsonify(message)

        #So that file can be passed in as a file
        elif request.files:
            file = request.files['file']
            file_name = file.filename
            target_path = static_path
            file_path = "/".join([target_path, file_name])

            with open(file_path, 'rb') as file:
                img_bytes = file.read()

            img = transform_image(img_bytes)
            model_output = model(img)
            label = torch.argmax(model_output, dim=1)
            label = str(label.item())
            class_ = str(label_dictionary.get(int(label)))
            message = "This picture contains a " + class_ + " with index " \
            + label

            return jsonify(message)

        else:
            return make_response(jsonify("Image not received"), 404)

api.add_resource(Index, "/")
api.add_resource(ImageFromLink, "/imagefromlink")
api.add_resource(ViewImage, "/viewimage")
api.add_resource(GetPredictionWithImage, "/getpredictionwithimage")
api.add_resource(GetPrediction, "/getprediction")

app.run(debug=True, port=5001, host="0.0.0.0")
