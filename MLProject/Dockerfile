FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow==2.19.0 pandas numpy scikit-learn

COPY model /app/model

ENV MODEL_PATH=/app/model

EXPOSE 5001

CMD ["mlflow", "models", "serve", "-m", "/app/model", "--host", "0.0.0.0", "--port", "5001", "--no-conda"]