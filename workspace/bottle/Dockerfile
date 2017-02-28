FROM continuumio/anaconda3

RUN groupadd web
RUN useradd -d /home/bottle -m bottle

ADD server.py /home/bottle/server.py
RUN pip install bottle

EXPOSE 8080
ENTRYPOINT ["python", "/home/bottle/server.py", ">", "/dev/null", "2>&1", "&"]
USER bottle
