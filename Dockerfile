FROM python:3.10-slim

WORKDIR /code

# Instalar dependencias del sistema (git a veces es necesario para algunas libs)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt

# Instalar dependencias Python (sin caché para reducir tamaño)
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

# Puerto
EXPOSE 8000

# Comando de inicio
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]