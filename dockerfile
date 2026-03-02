@"
FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_clean.txt .

RUN pip install --no-cache-dir -r requirements_clean.txt

RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

COPY . .

CMD ["python", "train.py"]
"@ | Set-Content Dockerfile