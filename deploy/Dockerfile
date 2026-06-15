FROM python:3.9-slim

# Instalar dependencias del sistema operativo necesarias para OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libxcb1
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copiar el archivo de requerimientos
COPY --chown=user requirements.txt .

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código del repositorio
COPY --chown=user . .

# Crear el directorio donde se guardarán temporalmente los videos subidos
RUN mkdir -p videos

# Hugging Face expone la app en el puerto 7860
EXPOSE 7860

# Iniciar la aplicación usando FastAPI en el puerto 7860
CMD ["uvicorn", "src.api.backend:app", "--host", "0.0.0.0", "--port", "7860"]
