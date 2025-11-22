FROM python:3.11-slim

WORKDIR /recruitair/api_evaluator

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api_evaluator ./recruitair/api_evaluator

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--ws-max-size", "16777216"]