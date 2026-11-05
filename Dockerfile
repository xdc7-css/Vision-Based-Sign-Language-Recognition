FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    scikit-learn>=1.3.0 \
    joblib>=1.3.0 \
    psutil>=5.9.0 \
    arabic-reshaper>=2.1.0 \
    python-bidi>=0.4.0

COPY . .

RUN mkdir -p data/models data/datasets data/cache data/logs config

ENV PYTHONPATH=/app/src
ENV QT_QPA_PLATFORM=offscreen

ENTRYPOINT ["python", "src/main.py"]
