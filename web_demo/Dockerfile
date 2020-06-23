FROM python:3.7
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install torchvision
RUN pip install torch
RUN pip install numpy
RUN pip install Pillow
RUN pip install flask
RUN pip install flask_restful
RUN pip install matplotlib

COPY . /web_demo
WORKDIR /web_demo


EXPOSE 5001

ENV main_path /web_demo
ENV label_path /web_demo/imagenet1000_clsidx_to_labels.txt
ENV static_path /web_demo/static

ENTRYPOINT python prediction.py $main_path $label_path $static_path
