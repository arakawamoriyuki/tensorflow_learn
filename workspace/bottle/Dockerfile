# Linux cd4ac837b33e 4.4.14-moby #1 SMP Wed Jun 29 10:00:58 UTC 2016 x86_64 GNU/Linux
FROM continuumio/anaconda3

# user
RUN groupadd web
RUN useradd -d /home/bottle -m bottle

# bottle
ADD server.py /home/bottle/server.py
RUN pip install bottle

# TensorFlow v1.0.0 for Linux
RUN pip install tensorflow
ENV TF_BINARY_URL https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.0-cp36-cp36m-linux_x86_64.whl
RUN pip install --ignore-installed --upgrade $TF_BINARY_URL

#TODO: tensorboard

# server
EXPOSE 8080
ENTRYPOINT ["python", "/home/bottle/server.py", ">", "/dev/null", "2>&1", "&"]
USER bottle