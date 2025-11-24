# Dockerfile (CPU)
FROM python:3.11-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy only the package code (faster builds)
COPY recruitair ./recruitair

EXPOSE 8000

CMD ["uvicorn", "recruitair.api_evaluator.main:app", "--host", "0.0.0.0", "--port", "8000"]
