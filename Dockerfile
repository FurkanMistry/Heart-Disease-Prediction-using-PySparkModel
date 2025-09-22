FROM python:3.11-slim-bookworm

# System deps (onnxruntime needs libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    DJANGO_WSGI_APP=heartrisk.wsgi:application

WORKDIR /app

# Install Python deps first for caching
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install gunicorn==21.2.0

# Copy project code
COPY . /app

# Ensure entrypoint has proper LF endings and is executable
RUN sed -i 's/\r$//' /app/entrypoint.sh || true && \
    chmod +x /app/entrypoint.sh || true

# Non-root
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
