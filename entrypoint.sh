#!/usr/bin/env bash
set -euo pipefail

# Set default settings module if not provided
export DJANGO_SETTINGS_MODULE=${DJANGO_SETTINGS_MODULE:-heartrisk.settings}

# Migrations & collectstatic
python manage.py migrate --noinput || true
python manage.py collectstatic --noinput || true

# Start Gunicorn
exec gunicorn "${DJANGO_WSGI_APP:-heartrisk.wsgi:application}" \
  --bind "0.0.0.0:${PORT:-8000}" \
  --workers "${WEB_CONCURRENCY:-2}" \
  --threads "${WEB_THREADS:-2}" \
  --timeout "${WEB_TIMEOUT:-60}" \
  --access-logfile '-' --error-logfile '-'
