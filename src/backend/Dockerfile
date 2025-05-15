FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN ls -la /app/data/processed
RUN ls -la /app/src/backend  # Debug main.py presence
WORKDIR /app/src/backend
RUN chmod +r /app/src/backend/main.py  # Ensure permissions
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]