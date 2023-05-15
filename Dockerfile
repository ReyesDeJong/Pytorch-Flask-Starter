FROM python:3.10-slim

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY /src ./src

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "60", "src.app:app"]
