# Umbrales específicos de la clase OBB
CLASS_CONF_THRESHOLDS = {
    'TT TABLE': 0.18,
    'TT NET': 0.25,
    'TT RACKET': 0.12,
}

# Confianza mínima para segmentación y pose
CONF_SEG = 0.30
CONF_POSE = 0.25

# Tamaño de imagen para los modelos
MODEL_IMGSZ = 640
MODEL_SEG_IMGSZ = 480  # Se debe mantener en 480 ya que el modelo ONNX fue exportado de forma estática (dynamic=False)
