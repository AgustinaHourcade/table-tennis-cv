import os
from ultralytics import YOLO
import glob

def export_models():
    # Define absolute or relative paths from project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    models_to_export = [
        os.path.join(base_dir, "notebooks", "yolo26n-seg.pt"),
        os.path.join(base_dir, "notebooks", "yolo26n-pose.pt"),
    ]
    
    # Encontrar el mejor modelo OBB en notebooks/runs
    obb_models = glob.glob(os.path.join(base_dir, "notebooks", "runs", "**", "weights", "best.pt"), recursive=True)
    if obb_models:
        models_to_export.append(obb_models[0])
    else:
        print("Advertencia: No se encontró best.pt para OBB en notebooks/runs/obb/")
        # Fallback al notebook obb si existe
        fallback = os.path.join(base_dir, "notebooks", "yolo26m-obb.pt")
        if os.path.exists(fallback):
            models_to_export.append(fallback)
            
    for model_path in models_to_export:
        if os.path.exists(model_path):
            print(f"\n--- Exportando a ONNX: {model_path} ---")
            model = YOLO(model_path)
            # format='onnx' exporta el modelo, imgsz=640 optimiza la exportacion para ese tamaño fijo
            model.export(format="onnx", imgsz=640, dynamic=False)
        else:
            print(f"Error: No se encontró el archivo {model_path}")

if __name__ == "__main__":
    print("Iniciando exportación de modelos a ONNX...")
    export_models()
    print("\n¡Exportación completada!")
