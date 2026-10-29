# Universal Agents: one image for the API, the worker and database upgrades.
#   docker build -t universal-agents .
#   docker run universal-agents serve --host 0.0.0.0     (or: worker, init-db, model-check, ...)
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AGENTS_DATA_DIR=/data

# tesseract reads scanned CVs and LinkedIn screenshots.
RUN printf 'Acquire::Retries "5";\nAcquire::http::Timeout "30";\n' > /etc/apt/apt.conf.d/80retries \
    && apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home agents \
    && mkdir /data && chown agents:agents /data

WORKDIR /app
COPY pyproject.toml README.md ./
COPY agents_service ./agents_service
RUN pip install --no-cache-dir '.[s3]'
# Made-up documents for `agents accuracy` after a model change.
COPY accuracy ./accuracy

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=4)" || exit 1
ENTRYPOINT ["agents"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
