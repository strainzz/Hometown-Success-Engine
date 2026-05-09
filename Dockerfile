# build 20260507-153028
# build 20260507-1354
# build 20260507-1353
FROM python:3.12-slim AS builder
WORKDIR /build
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY backend/ /app/
COPY pipeline/clustered/hubs.json /app/pipeline/clustered/hubs.json
COPY pipeline/clustered/athletes.json /app/pipeline/clustered/athletes.json
COPY pipeline/narratives/hubs.json /app/pipeline/narratives/hubs.json
COPY pipeline/geo/us-states.json /app/pipeline/geo/us-states.json
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
EXPOSE 8080
CMD exec gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT


