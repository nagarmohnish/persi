FROM python:3.10-slim

RUN apt-get update && apt-get install -y git-lfs && git lfs install

WORKDIR /app
COPY . .
RUN git lfs pull
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN ls -la /app/data/processed

EXPOSE 8000
CMD ["uvicorn", "main.py:app", "--host", "0.0.0.0", "--port", "8000"]