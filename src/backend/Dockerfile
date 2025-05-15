FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r src/backend/requirements.txt
RUN ls -la /app
RUN ls -la /app/data/processed
RUN ls -la /app/src/backend || echo "src/backend not found"
WORKDIR /app/src/backend
RUN chmod +r /app/src/backend/main.py || echo "main.py not found"
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]