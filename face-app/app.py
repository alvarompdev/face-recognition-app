# Autor: Álvaro Muñoz Panadero - alvaromp.dev@gmail.com
# API FastAPI para el sistema de reconocimiento equilibrado

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import pickle
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import os

app = FastAPI(
    title="Balanced Face Recognition API",
    description="API para reconocimiento facial equilibrado",
    version="1.0.0"
)

# Variables globales para el modelo
recognizer_model = None
scaler = None
person_name = ""
face_cascade = None
cluster_centers = None
max_distance_threshold = 0.0
is_loaded = False

def load_recognition_model():
    """Carga el modelo de reconocimiento equilibrado"""
    global recognizer_model, scaler, person_name, face_cascade, cluster_centers, max_distance_threshold, is_loaded
    
    try:
        # Cargar modelo
        with open("face_model.pkl", 'rb') as f:
            model_data = pickle.load(f)
            
        recognizer_model = model_data['face_clusters']
        scaler = model_data['scaler']
        person_name = model_data['person_name']
        cluster_centers = model_data['cluster_centers']
        max_distance_threshold = model_data['max_distance_threshold']
        
        # Inicializar detector de caras
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        is_loaded = True
        print(f"✅ Modelo cargado para: {person_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        is_loaded = False
        return False

def extract_simple_features(face_gray):
    """Extrae características simples de una cara (igual que en face_recognizer)"""
    # Redimensionar a tamaño fijo
    face = cv2.resize(face_gray, (80, 80))
    face = cv2.equalizeHist(face)
    
    features = []
    
    # 1. Histograma global
    hist = cv2.calcHist([face], [0], None, [16], [0, 256])
    features.extend(hist.flatten())
    
    # 2. Dividir cara en 4 regiones principales
    h, w = face.shape
    regions = [
        face[0:h//2, 0:w//2],     # Superior izquierda
        face[0:h//2, w//2:w],     # Superior derecha
        face[h//2:h, 0:w//2],     # Inferior izquierda
        face[h//2:h, w//2:w]      # Inferior derecha
    ]
    
    for region in regions:
        if region.size > 0:
            hist = cv2.calcHist([region], [0], None, [8], [0, 256])
            features.extend(hist.flatten())
            features.append(np.mean(region))
    
    # 3. Características de bordes
    edges = cv2.Canny(face, 30, 100)
    features.append(np.sum(edges) / (h * w))
    
    # 4. Momentos básicos
    moments = cv2.moments(face)
    features.append(moments['m00'])
    features.append(moments['m10'])
    features.append(moments['m01'])
    
    return np.array(features, dtype=np.float64)

def recognize_face_from_features(face_gray):
    """Reconoce una cara usando el modelo equilibrado"""
    if not is_loaded:
        return "Modelo no cargado", 0.0
    
    try:
        # Extraer características
        features = extract_simple_features(face_gray)
        if len(features) == 0:
            return "Error extrayendo características", 0.0
        
        # Normalizar
        features_scaled = scaler.transform([features])
        
        # Encontrar cluster más cercano
        cluster_id = recognizer_model.predict(features_scaled)[0]
        nearest_center = cluster_centers[cluster_id]
        
        # Calcular distancia
        distance = np.linalg.norm(features_scaled[0] - nearest_center)
        
        # Decisión
        if distance <= max_distance_threshold:
            raw_confidence = 1.0 - (distance / max_distance_threshold)
            confidence = min(0.95, max(0.4, raw_confidence * 1.8))
            return person_name, confidence
        else:
            return "Desconocido", 0.05
            
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# Cargar modelo al iniciar
load_recognition_model()

@app.get("/")
async def root():
    """Endpoint de información"""
    return {
        "message": "Balanced Face Recognition API",
        "status": "loaded" if is_loaded else "model not found",
        "person": person_name if is_loaded else "none",
        "version": "1.0.0"
    }

@app.get("/status")
async def get_status():
    """Obtener estado del modelo"""
    return {
        "model_loaded": is_loaded,
        "person_name": person_name if is_loaded else None,
        "threshold": max_distance_threshold if is_loaded else None,
        "model_file_exists": os.path.exists("face_model.pkl")
    }

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """Reconocer caras en una imagen subida"""
    
    if not is_loaded:
        raise HTTPException(status_code=503, detail="Modelo no cargado. Ejecuta face_recognizer.py train primero")
    
    try:
        # Leer imagen
        content = await file.read()
        
        # Convertir a formato OpenCV
        img_pil = Image.open(BytesIO(content)).convert("RGB")
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detectar caras
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        
        if len(faces) == 0:
            return {"faces": [], "message": "No se detectaron caras en la imagen"}
        
        results = []
        for (x, y, w, h) in faces:
            # Extraer cara
            face = gray[y:y+h, x:x+w]
            
            # Reconocer
            name, confidence = recognize_face_from_features(face)
            
            results.append({
                "location": {
                    "left": int(x),
                    "top": int(y),
                    "right": int(x + w),
                    "bottom": int(y + h)
                },
                "name": name,
                "confidence": float(confidence),
                "is_known": name == person_name,
                "size": f"{w}x{h}"
            })
        
        return {
            "faces": results,
            "total_faces": len(results),
            "known_person": person_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

@app.post("/reload-model")
async def reload_model():
    """Recargar el modelo (útil después de re-entrenar)"""
    success = load_recognition_model()
    
    if success:
        return {
            "message": "Modelo recargado exitosamente",
            "person": person_name,
            "threshold": max_distance_threshold
        }
    else:
        raise HTTPException(status_code=503, detail="Error recargando el modelo")

@app.get("/model-info")
async def get_model_info():
    """Información detallada del modelo"""
    if not is_loaded:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    return {
        "person_name": person_name,
        "max_distance_threshold": max_distance_threshold,
        "clusters": len(cluster_centers) if cluster_centers is not None else 0,
        "model_type": "Balanced Face Recognition",
        "features": "Simple histogram + texture + edges + moments"
    }

if __name__ == "__main__":
    print("🚀 Iniciando Balanced Face Recognition API...")
    print(f"📊 Estado del modelo: {'✅ Cargado' if is_loaded else '❌ No encontrado'}")
    if is_loaded:
        print(f"👤 Persona registrada: {person_name}")
    else:
        print("⚠️  Ejecuta 'python face_recognizer.py train <nombre>' primero")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)