import os
from ultralytics import YOLO
import glob

def export_models():
    # Define absolute or relative paths from project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    models_to_export = [
        os.path.join(base_dir, "models", "weights", "yolo26n-seg.pt"),
        os.path.join(base_dir, "models", "weights", "yolo26n-pose.pt"),
    ]
    
    # Encontrar el mejor modelo OBB en notebooks/runs
    obb_models = glob.glob(os.path.join(base_dir, "notebooks", "runs", "**", "weights", "best.pt"), recursive=True)
    if obb_models:
        # Move best.pt to models/weights/ for cleanliness
        import shutil
        best_pt_target = os.path.join(base_dir, "models", "weights", "best.pt")
        shutil.copy2(obb_models[0], best_pt_target)
        models_to_export.append(best_pt_target)
    else:
        print("Advertencia: No se encontró best.pt para OBB en notebooks/runs/obb/")
        # Fallback
        fallback = os.path.join(base_dir, "models", "weights", "yolo26m-obb.pt")
        if os.path.exists(fallback):
            models_to_export.append(fallback)
            
    for model_path in models_to_export:
        if os.path.exists(model_path):
            print(f"\n--- Exportando a ONNX: {model_path} ---")
            model = YOLO(model_path)
            
            # Configuramos exportación específica por tipo de modelo
            export_imgsz = 480 if "seg" in model_path else 640
            
            # format='onnx' exporta el modelo, imgsz define el tamaño estático
            # half=False para evitar problemas en CPUs de HF Spaces (poner en True solo si hay GPU)
            model.export(format="onnx", imgsz=export_imgsz, dynamic=False, half=False)
        else:
            print(f"Error: No se encontró el archivo {model_path}")

if __name__ == "__main__":
    print("Iniciando exportación de modelos a ONNX...")
    export_models()
    print("\n¡Exportación completada!")
