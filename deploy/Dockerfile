# Reproducible CPU image for tests + experiments (multi-arch: works on x86-64 and aarch64 / Jetson-class hosts)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "run_all.py", "--quick"]
