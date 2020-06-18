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



from flask import Flask,request,jsonify,make_response,render_template,send_from_directory
from flask_restful import Api,Resource

with open('/Users/admin/Documents/Abeja/projectAbeja2020/imagenet1000_clsidx_to_labels.txt') as f:
    label_dict=eval(f.read())

model=torchvision.models.resnet152(pretrained=True)
model.eval()

app=Flask(__name__)
api=Api(app)



def transform_image(img_bytes):
  transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
  img=Image.open(io.BytesIO(img_bytes))
  print(np.asarray(img).shape)
  return transform(img).unsqueeze(0)

class index(Resource):
    def get(self):
        return make_response(render_template("upload.html"))

class view_image(Resource):
    def post(self):
        target='/Users/admin/Documents/Abeja/projectAbeja2020/static'
        print(target)
        filenames=[]
        num_images=0
        if not os.path.isdir(target):
            os.mkdir(target)
        for file in request.files.getlist("file"): #file is an object
            filename=file.filename
            filenames.append(filename)
            dest= "/".join([target,filename])  #tell the server to upload the file to this destination
            num_images+=1
        # return send_from_directory("static",filename,as_attachment=True) this will save the images that are upload to our computer in static folder as file name
        return make_response(render_template("view.html",img_list=filenames,num=num_images))


class image_link(Resource):
    def post(self):
        if request.files:
            url=request.files['url']
            with open(url,'wb') as file:
                img_bytes=file.read()
            img=transform_image(img_bytes)
            print(np.asarray(img).shape)
            label=model(img)
            label=torch.argmax(label,dim=1)
            label=str(label.item())
            class_=str(label_dict.get(int(label)))
            message="This picture contains a "+class_ +" with index " + label

        #plt.imshow(img_raw)
            return make_response(render_template("image_link.html",img_url=url,label=message))



class get_prediction_image(Resource):
    def post(self):
        #if request.args:
            #filename=request.args.get('file')
        #if request.files:
        if request.files:
            dir='/Users/admin/Documents/Abeja/projectAbeja2020/static'
            imgs=[]
            labels=[]
            num_images=0
            for upload in request.files.getlist("file"):
                file=upload
                filename=file.filename
                filename_path="/".join([dir,filename])
                with open(filename_path,'rb') as file:
                    img_bytes=file.read()
            #img_raw=Image.open(io.BytesIO(img_bytes))  #get non transformed image for plotting
            #img_raw=np.transpose(np.asarray(img_raw),[1,2,0])  #convert to np array and ready image for plotting
            #img_raw=np.asarray(img_raw)
                img=transform_image(img_bytes)
                print(np.asarray(img).shape)
                label=model(img)
                label=torch.argmax(label,dim=1)
                label=str(label.item())
                class_=str(label_dict.get(int(label)))
                message="This picture contains a "+class_ +" with index " + label
                imgs.append(filename)
                labels.append(message)
                num_images+=1
            #plt.imshow(img_raw)
            return make_response(render_template('complete.html',img_list=imgs,label_list=labels,num=num_images))

        else:
            return make_response(jsonify("Image not received"),404)

class get_prediction(Resource):
    def post(self):
        if request.args:
            if "file" not in request.args.keys():
                return make_response(jsonify("query parameter is file"),404)
            else:
                dir='/Users/admin/Documents/Abeja/projectAbeja2020/static'
                filename=request.args.get('file')
                filepath="/".join([dir,filename])
                with open(filepath,'rb') as file:
                    img_bytes=file.read()
                img=transform_image(img_bytes)
                label=model(img)
                label=torch.argmax(label,dim=1)
                label=str(label.item())
                class_=str(label_dict.get(int(label)))
                message="This picture contains a " +class_ +" with index " + label
                return jsonify(message)

        elif request.files:
            file=request.files['file']
            filename=file.filename
            dir='/Users/admin/Documents/Abeja/projectAbeja2020/static'
            filepath="/".join([dir,filename])
            with open(filepath,'rb') as file:
                img_bytes=file.read()
            img=transform_image(img_bytes)
            label=model(img)
            label=torch.argmax(label,dim=1)
            label=str(label.item())
            class_=str(label_dict.get(int(label)))
            message="This picture contains a " +class_ +" with index " + label
            return jsonify(message)

        else:
            return make_response(jsonify("Image not received"),404)



api.add_resource(index,"/")
api.add_resource(image_link,"/image_link")
api.add_resource(view_image,"/view_image")
api.add_resource(get_prediction_image,"/get_prediction_image")
api.add_resource(get_prediction,"/get_prediction")


app.run(debug=True,port=5001,host="localhost")
