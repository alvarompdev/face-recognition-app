# Autor: Álvaro Muñoz Panadero - alvaromp.dev@gmail.com
# Sistema equilibrado: Reconoce fácil tu cara, detecta desconocidos
import cv2
import os
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys
import time

class BalancedFaceRecognizer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.scaler = StandardScaler()
        self.is_trained = False
        self.person_name = ""
        self.face_clusters = None
        self.cluster_centers = None
        self.max_distance_threshold = 0.0
        
    def extract_simple_features(self, face_gray):
        """Extrae características simples pero efectivas"""
        # Redimensionar a tamaño fijo
        face = cv2.resize(face_gray, (80, 80))
        face = cv2.equalizeHist(face)
        
        features = []
        
        # 1. Histograma global (reducido para ser más tolerante)
        hist = cv2.calcHist([face], [0], None, [16], [0, 256])
        features.extend(hist.flatten())
        
        # 2. Dividir cara en 4 regiones principales
        h, w = face.shape
        regions = [
            face[0:h//2, 0:w//2],     # Superior izquierda (ojo)
            face[0:h//2, w//2:w],     # Superior derecha (ojo)
            face[h//2:h, 0:w//2],     # Inferior izquierda
            face[h//2:h, w//2:w]      # Inferior derecha
        ]
        
        for region in regions:
            if region.size > 0:
                # Histograma más simple para cada región
                hist = cv2.calcHist([region], [0], None, [8], [0, 256])
                features.extend(hist.flatten())
                
                # Valor medio de la región
                features.append(np.mean(region))
        
        # 3. Características de bordes simplificadas
        edges = cv2.Canny(face, 30, 100)
        features.append(np.sum(edges) / (h * w))  # Densidad de bordes
        
        # 4. Momentos básicos
        moments = cv2.moments(face)
        features.append(moments['m00'])  # Área
        features.append(moments['m10'])  # Momento X
        features.append(moments['m01'])  # Momento Y
        
        return np.array(features, dtype=np.float64)
    
    def capture_person(self, person_name, n_images=10):
        """Captura fotos de manera más simple"""
        person_dir = os.path.join("dataset", person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False
        
        print(f"Capturando {n_images} fotos de {person_name}")
        print("Presiona SPACE para capturar, 'q' para salir")
        print("Varía ligeramente tu posición entre fotos")
        
        count = 0
        while count < n_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 4, minSize=(80, 80))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.putText(frame, f"Fotos: {count}/{n_images}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Caras detectadas: {len(faces)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if count < n_images:
                tips = ["Mira al frente", "Sonríe un poco", "Cara normal", 
                       "Gira un poco", "Levanta la cabeza", "Baja la cabeza"]
                tip = tips[count % len(tips)]
                cv2.putText(frame, tip, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow("Captura simple - SPACE: foto, Q: salir", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if len(faces) >= 1:
                    filename = os.path.join(person_dir, f"{person_name}_{int(time.time())}_{count}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"✓ Foto {count+1}/{n_images} guardada")
                    count += 1
                    time.sleep(0.5)
                else:
                    print("⚠️  No se detectó cara")
        
        cap.release()
        cv2.destroyAllWindows()
        return count >= 2  # Mínimo 2 fotos
    
    def train_balanced_model(self, person_name):
        """Entrena un modelo equilibrado"""
        person_dir = os.path.join("dataset", person_name)
        
        if not os.path.exists(person_dir):
            print(f"Error: No existe {person_dir}")
            return False
        
        features_list = []
        print(f"Entrenando modelo equilibrado para: {person_name}")
        
        # Procesar fotos
        for img_name in os.listdir(person_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(person_dir, img_name)
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
                
                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]
                    features = self.extract_simple_features(face)
                    
                    if len(features) > 0:
                        features_list.append(features)
                        
            except Exception as e:
                print(f"Error en {img_name}: {e}")
        
        if len(features_list) < 3:
            print("Error: Se necesitan al menos 3 fotos válidas")
            return False
        
        # Convertir y normalizar
        X = np.array(features_list)
        print(f"Procesadas {len(features_list)} fotos")
        
        # Normalizar características
        X_scaled = self.scaler.fit_transform(X)
        
        # Crear clusters de las características de la persona
        n_clusters = min(3, len(features_list))  # Máximo 3 clusters
        self.face_clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.face_clusters.fit(X_scaled)
        
        self.cluster_centers = self.face_clusters.cluster_centers_
        
        # Calcular umbral de distancia (distancia máxima dentro de los clusters)
        distances_to_centers = []
        for i, features in enumerate(X_scaled):
            cluster_id = self.face_clusters.predict([features])[0]
            center = self.cluster_centers[cluster_id]
            distance = np.linalg.norm(features - center)
            distances_to_centers.append(distance)
        
        # Umbral = distancia máxima + margen de tolerancia
        self.max_distance_threshold = np.max(distances_to_centers) * 1.5
        
        self.person_name = person_name
        self.is_trained = True
        
        print(f"✓ Modelo entrenado")
        print(f"  - Clusters creados: {n_clusters}")
        print(f"  - Umbral de distancia: {self.max_distance_threshold:.2f}")
        
        # Guardar
        self.save_model()
        return True
    
    def save_model(self):
        """Guarda el modelo"""
        model_data = {
            'face_clusters': self.face_clusters,
            'cluster_centers': self.cluster_centers,
            'scaler': self.scaler,
            'person_name': self.person_name,
            'max_distance_threshold': self.max_distance_threshold
        }
        
        with open("face_model.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        print("✓ Modelo guardado")
    
    def load_model(self):
        """Carga el modelo"""
        try:
            with open("face_model.pkl", 'rb') as f:
                model_data = pickle.load(f)
                
            self.face_clusters = model_data['face_clusters']
            self.cluster_centers = model_data['cluster_centers']
            self.scaler = model_data['scaler']
            self.person_name = model_data['person_name']
            self.max_distance_threshold = model_data['max_distance_threshold']
            self.is_trained = True
            
            print(f"✓ Modelo cargado para: {self.person_name}")
            print(f"  Umbral: {self.max_distance_threshold:.2f}")
            return True
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False
    
    def recognize_face(self, face_gray):
        """Reconocimiento equilibrado"""
        if not self.is_trained:
            return "No entrenado", 0.0
        
        try:
            # Extraer características
            features = self.extract_simple_features(face_gray)
            if len(features) == 0:
                return "Error", 0.0
            
            # Normalizar
            features_scaled = self.scaler.transform([features])
            
            # Encontrar cluster más cercano
            cluster_id = self.face_clusters.predict(features_scaled)[0]
            nearest_center = self.cluster_centers[cluster_id]
            
            # Calcular distancia al centro del cluster más cercano
            distance = np.linalg.norm(features_scaled[0] - nearest_center)
            
            # Decisión simple pero efectiva
            if distance <= self.max_distance_threshold:
                # Es la persona conocida
                confidence = max(0.1, 1.0 - (distance / self.max_distance_threshold))
                return self.person_name, confidence
            else:
                # Es desconocido
                return "Desconocido", 0.1
                
        except Exception as e:
            print(f"Error reconocimiento: {e}")
            return "Error", 0.0
    
    def run_recognition(self):
        """Reconocimiento en tiempo real"""
        if not self.is_trained:
            print("Error: Modelo no entrenado")
            return
        
        cap = cv2.VideoCapture(0)
        print(f"Reconocimiento para: {self.person_name}")
        print(f"Umbral de similitud: {self.max_distance_threshold:.2f}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Procesar cada 2 frames para mejor rendimiento
            if frame_count % 2 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
                
                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]
                    name, confidence = self.recognize_face(face)
                    
                    # Colores
                    if name == self.person_name and confidence > 0.4:
                        color = (0, 255, 0)  # Verde
                        label = f"{name} ({confidence:.2f})"
                    elif name == "Desconocido":
                        color = (0, 0, 255)  # Rojo
                        label = "Desconocido"
                    else:
                        color = (0, 165, 255)  # Naranja para baja confianza
                        label = f"{name}? ({confidence:.2f})"
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.putText(frame, f"Buscando: {self.person_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Reconocimiento Equilibrado - 'q' salir", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    recognizer = BalancedFaceRecognizer()
    
    if len(sys.argv) < 2:
        print("Sistema de Reconocimiento Equilibrado")
        print("Uso:")
        print("  python balanced_recognizer.py capture <nombre>")
        print("  python balanced_recognizer.py train <nombre>")
        print("  python balanced_recognizer.py recognize")
        print("  python balanced_recognizer.py test")
        return
    
    command = sys.argv[1].lower()
    
    if command == "capture":
        if len(sys.argv) < 3:
            name = input("Nombre de la persona: ")
        else:
            name = sys.argv[2]
        recognizer.capture_person(name, 8)
        
    elif command == "train":
        if len(sys.argv) < 3:
            name = input("Nombre para entrenar: ")
        else:
            name = sys.argv[2]
        recognizer.train_balanced_model(name)
        
    elif command == "recognize":
        if recognizer.load_model():
            recognizer.run_recognition()
        else:
            print("Error: No hay modelo. Ejecuta 'train' primero")
            
    elif command == "test":
        if recognizer.load_model():
            print(f"Modelo cargado para: {recognizer.person_name}")
            print(f"Umbral: {recognizer.max_distance_threshold:.2f}")
            print("Prueba el reconocimiento...")
            recognizer.run_recognition()
        else:
            print("No hay modelo entrenado")

if __name__ == "__main__":
    main()