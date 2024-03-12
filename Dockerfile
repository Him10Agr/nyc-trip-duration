FROM python:3.10.12-slim

WORKDIR /app

COPY fast_api_app.py /app
COPY model.pkl /app
COPY src/ /app/src/
COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

CMD ["python3", "fast_api_app.py"]