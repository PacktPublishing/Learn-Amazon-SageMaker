FROM python:3.7

RUN pip3 install --no-cache scikit-learn numpy pandas joblib
RUN pip3 install --no-cache flask

COPY sklearn-boston-housing-generic.py /usr/bin/train
COPY sklearn-boston-housing-serve.py /usr/bin/serve

RUN chmod 755 /usr/bin/train /usr/bin/serve

EXPOSE 8080
