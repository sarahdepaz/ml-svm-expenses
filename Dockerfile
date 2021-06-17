# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster
WORKDIR /Users/sarahdepaz/Downloads/daily-expense-ml
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD ["python3", "./main_model.py"]