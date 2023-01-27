FROM python:3.8

COPY . /ml-engineering-demo
WORKDIR /ml-engineering-demo

COPY requirements.txt .
RUN python -m pip install --upgrade pip -U
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "simple_linear_regr.py"]
