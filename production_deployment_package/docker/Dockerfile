# Production Docker build for hf-eco2ai-plugin
FROM python:3.11-slim

WORKDIR /app
COPY . /app/

RUN pip install -e .

EXPOSE 8000
CMD ["python", "-m", "hf_eco2ai.api"]
