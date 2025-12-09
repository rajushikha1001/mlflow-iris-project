FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/

CMD ["mlflow", "ui", "--backend-store-uri", "sqlite:///mlflow.db", "--host", "0.0.0.0"]