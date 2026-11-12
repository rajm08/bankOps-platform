FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt
COPY backend /app/backend
COPY DataSets /app/DataSets
RUN mkdir -p /app/Tasks
COPY Tasks/*.pkl /app/Tasks/
COPY banking_knowledge_db /app/banking_knowledge_db
ENV PYTHONPATH=/app/backend
ENV PROJECT_ROOT=/app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
