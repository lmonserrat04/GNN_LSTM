# Usamos la imagen base de PyTorch compatible con su CUDA 12.1 (funciona perfecto con drivers 12.9)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Evitar bloqueos por zonas horarias o instalaciones interactivas
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Primero copiamos el requirements para instalar las dependencias
COPY requirements.txt .

# Instalamos tus librerías exactas
RUN pip install --no-cache-dir -r requirements.txt

# Instalamos PyTorch Geometric por separado para asegurar compatibilidad con CUDA 12
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Copiamos el resto del código
COPY . .

# Comando para arrancar el entrenamiento
CMD ["python", "train.py"]