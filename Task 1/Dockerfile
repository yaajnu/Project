FROM  python:3.7

WORKDIR /model

COPY requirements.txt ./requirements.txt


RUN pip3 install -r requirements.txt

EXPOSE 8501

COPY . /model

ENTRYPOINT ["streamlit","run"]

CMD ["model.py"]



