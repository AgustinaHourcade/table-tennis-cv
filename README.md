# Proyecto Integrador: Detección, Segmentación y Estimación de Pose utilizando YOLO

Este repositorio contiene el desarrollo del **Proyecto Integrador** para la materia **Redes Neuronales**. El objetivo del proyecto es diseñar e implementar una solución integral de visión artificial que combine detección de objetos con clases personalizadas, segmentación de instancias y estimación de pose humana en tiempo real sobre videos de **Tenis de Mesa (Table Tennis)**.

---

## 🏓 Clases Personalizadas (Dataset en Roboflow)
El modelo ha sido entrenado para detectar tres clases específicas que no forman parte del dataset original de COCO:
1. **`TT Racket`**: Paleta de tenis de mesa (ping pong).
2. **`TT Table`**: Mesa de juego de tenis de mesa.
3. **`TT Net`**: Red de la mesa de tenis de mesa.

El etiquetado y curación del dataset se realiza utilizando la plataforma **Roboflow**, garantizando un mínimo de 50 imágenes por clase y exportando el dataset en formato compatible con YOLO.

---

## 🎯 Objetivos del Proyecto

El sistema procesa videos (de hasta 2 minutos de duración) integrando simultáneamente tres tareas de visión artificial con modelos de la familia YOLO:

1. **Detección de Clases Personalizadas**:
   - Identificación de `TT Racket`, `TT Table` y `TT Net`.
   - Dibujado de Bounding Boxes con el nombre de la clase y el nivel de confianza.
2. **Segmentación**:
   - Segmentación de objetos de la base de datos de COCO que superen un umbral de confianza definido.
   - **Exclusiones**: Se excluyen de la segmentación las personas y las clases personalizadas entrenadas (`TT Racket`, `TT Table`, `TT Net`).
3. **Estimación de Pose (Pose Estimation)**:
   - Detección de personas en el video.
   - Dibujado de Keypoints (puntos clave), esqueleto corporal y seguimiento visual de la pose de los jugadores.
4. **Reporte en Tiempo Real**:
   - Superposición de un panel informativo en el video que muestra dinámicamente la cantidad de objetos detectados de cada clase personalizada (por ejemplo: `TT Racket: 2`, `TT Table: 1`, `TT Net: 1`).

---

## 📁 Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

```text
TPI/
├── .gitignore                  # Exclusiones de archivos para Git
├── README.md                   # Documentación principal (este archivo)
├── requirements.txt            # Dependencias del proyecto
├── data.yaml                   # Configuración del dataset exportado de Roboflow
├── Proyecto Integrador...pdf   # Consigna oficial de la materia
│
├── notebooks/                  # Notebooks para entrenamiento y pruebas
│   └── entrenamiento.ipynb     # Notebook de Google Colab / Jupyter para Transfer Learning
│
├── src/                        # Código fuente del sistema de inferencia
│   ├── __init__.py
│   ├── main.py                 # Pipeline principal de procesamiento de video
│   └── utils.py                # Funciones de utilidad para dibujar overlays, conteo, etc.
│
├── weights/                    # Pesos de los modelos (Excluido de Git)
│   └── best.pt                 # Pesos resultantes del entrenamiento personalizado
│
└── output/                     # Videos procesados finales (Excluido de Git)
```

---

## 🛠️ Instalación y Requisitos

### Requisitos Previos
- Python 3.9 o superior.
- Git.
- Entorno de ejecución con soporte para GPU (opcional pero recomendado para el entrenamiento y una inferencia más fluida).

### Pasos para Configurar Localmente

1. **Clonar el repositorio**:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd TPI
   ```

2. **Crear y activar un entorno virtual**:
   - **En Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - **En Linux/macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Instalar dependencias**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

*(Nota: Si el archivo `requirements.txt` no está creado aún, las dependencias principales requeridas son `ultralytics`, `opencv-python`, `numpy` y `jupyter`).*

---

## 🚀 Entrenamiento y Uso

### 1. Entrenamiento del Modelo
El entrenamiento se realiza por medio de **Transfer Learning** a partir de un modelo preentrenado de YOLO.
- El dataset etiquetado en Roboflow debe descargarse en la carpeta raíz o configurarse en `data.yaml`.
- Abre el notebook [entrenamiento.ipynb](file:///c:/Users/Ignac/Universidad/Materias/Redes%20Neuronales/TPI/notebooks/entrenamiento.ipynb) dentro de la carpeta `notebooks/` y sigue los pasos para entrenar el modelo.
- Al finalizar, guarda el archivo de pesos obtenido (`best.pt`) dentro del directorio `weights/`.

### 2. Procesamiento de Video (Inferencia)
Para correr el pipeline que procesa el video original y genera el video resultante con la segmentación, pose y la detección con el reporte en tiempo real:

```bash
python src/main.py --input path/to/video.mp4 --output output/video_procesado.mp4
```

#### Argumentos de `src/main.py`:
- `--input`: Ruta al video original o link de video público (máximo 2 minutos).
- `--output`: Ruta donde se guardará el video resultante.
- `--conf`: Umbral de confianza mínimo para detección y segmentación (por defecto `0.5`).

---

## 📊 Documentación de Resultados y Métricas
*(Esta sección se completará una vez finalizado el entrenamiento del modelo)*

- **Arquitectura Utilizada**: YOLOv8 / YOLOv11 (según versión seleccionada).
- **Cantidad de Épocas**: `TBD`
- **Batch Size**: `TBD`
- **Tamaño de Imagen (Image Size)**: `TBD`
- **Métricas Obtenidas**:
  - Precision (P): `TBD`
  - Recall (R): `TBD`
  - mAP50: `TBD`
  - mAP50-95: `TBD`

---

## 👥 Integrantes del Grupo
- [Nombre Integrante 1] - [Email/Legajo]
- [Nombre Integrante 2] - [Email/Legajo]
- [Nombre Integrante 3] - [Email/Legajo]
