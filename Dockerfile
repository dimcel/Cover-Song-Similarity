FROM python:3.9
WORKDIR /app

ARG MODEL_PATH
ARG SRC_PATH

COPY requirements.txt .
COPY app.py .
COPY config_api.yml .
COPY src/preprocessing.py ./src/preprocessing.py

COPY ${MODEL_PATH} /app/models/best_siamese_model.pth

COPY ${SRC_PATH}/ /app/src

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
