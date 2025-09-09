# 🎯 Basic Face Recognition System

**Sistema de reconocimiento facial básico que reconoce fácilmente caras conocidas y detecta correctamente personas desconocidas.**

**Autor:** Álvaro Muñoz Panadero - [alvaromp.dev@gmail.com](mailto:alvaromp.dev@gmail.com)

---

## 🚀 Características principales

- ✅ **Reconocimiento preciso** de una persona específica
- ✅ **Detección efectiva** de personas desconocidas  
- ✅ **Sistema equilibrado** - No genera falsos positivos
- ✅ **Captura rápida** - Solo 2-3 fotos necesarias
- ✅ **API REST** incluida para integración web
- ✅ **Sin dependencias complejas** - Solo OpenCV y scikit-learn
- ✅ **Multiplataforma** - Windows, Mac, Linux

## 🎬 Demo rápido

```bash
# Capturar fotos (solo 3 fotos por defecto)
python face_recognizer.py capture alvaro

# Entrenar modelo
python face_recognizer.py train alvaro

# Reconocimiento en tiempo real
python face_recognizer.py recognize
```

---

## 📦 Instalación

### Requisitos previos
- Python 3.8 o superior
- Webcam (para captura y reconocimiento en tiempo real)

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/face-recognition-app.git
cd face-recognition-app
```

### 2. Crear entorno virtual

#### Windows:
```bash
python -m venv face-env
face-env\Scripts\activate
```

#### macOS/Linux:
```bash
python3 -m venv face-env
source face-env/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## 🎯 Uso del sistema

### Reconocimiento local (Webcam)

#### 1. Capturar fotos de entrenamiento
```bash
python face_recognizer.py capture <tu_nombre>

# Ejemplo:
python face_recognizer.py capture alvaro
```
- Se abrirá la webcam
- Presiona **SPACE** para capturar cada foto
- Varía ligeramente tu posición entre fotos
- Presiona **Q** para salir

#### 2. Entrenar el modelo
```bash
python face_recognizer.py train <tu_nombre>

# Ejemplo:
python face_recognizer.py train alvaro
```

El modelo solamente puede estar 'entrenado' para una persona simultáneamente, por lo que si lo último que has ejecutado es
```bash
python face_recognizer.py train alvaro
```
en ese momento estará 'entrenado' para reconocer únicamente la cara de "alvaro". Para cambiarlo para otra persona, basta con ejecutar el mismo comando pero cambiando el nombre por el de la nueva persona entrenada.

#### 3. Iniciar reconocimiento en tiempo real
```bash
python face_recognizer.py recognize
```
- **Verde** = Persona conocida
- **Rojo** = Desconocido

### API REST

#### 1. Iniciar servidor API
```bash
python app.py
```
El servidor se iniciará en: `http://localhost:8000`

#### 2. Probar la API

**Navegador (GET):**
- `http://localhost:8000/` - Información general
- `http://localhost:8000/status` - Estado del modelo
- `http://localhost:8000/docs` - Interfaz Swagger UI

**Postman (POST):**
- **URL**: `http://localhost:8000/recognize`
- **Method**: POST
- **Body**: form-data
- **Key**: `file` (tipo: File)
- **Value**: Seleccionar imagen

---

## 📁 Estructura del proyecto

```
face-recognition-app/
├── balanced_recognizer.py    # Sistema principal (todo-en-uno)
├── app.py                   # API REST con FastAPI
├── requirements.txt         # Dependencias
├── README.md               # Este archivo
├── dataset/                # Carpeta de fotos (se crea automáticamente)
│   └── alvaro/            # Fotos de entrenamiento de cada persona guardada
└── face_model.pkl # Modelo entrenado (se crea automáticamente)
```

---

## ⚙️ Comandos disponibles

### face_recognizer.py

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| `capture <nombre> [num_fotos]` | Capturar fotos de entrenamiento | `python face_recognizer.py capture juan 5` |
| `train <nombre>` | Entrenar modelo con las fotos | `python face_recognizer.py train juan` |
| `recognize` | Iniciar reconocimiento en tiempo real | `python face_recognizer.py recognize` |
| `test` | Probar modelo con información debug | `python face_recognizer.py test` |

### API (app.py)

| Endpoint | Método | Descripción |
|----------|---------|-------------|
| `/` | GET | Información general |
| `/status` | GET | Estado del modelo |
| `/model-info` | GET | Información detallada del modelo |
| `/recognize` | POST | Reconocer caras en imagen |
| `/reload-model` | POST | Recargar modelo sin reiniciar |

---

## 🔧 Configuración avanzada

### Ajustar sensibilidad
Edita en `face_recognizer.py`:
```python
# Línea ~150 - Más estricto (menos falsos positivos)
confidence_threshold = 0.85  # Por defecto: 0.4

# Línea ~140 - Cambiar umbral de distancia  
self.max_distance_threshold = np.max(distances_to_centers) * 1.2  # Por defecto: 1.5
```

### Cambiar número de fotos por defecto
```python
# Línea ~25
def capture_person(self, person_name, n_images=5):  # Por defecto: 3
```

---

## 🐛 Solución de problemas

### Error: "No se puede abrir la cámara"
```bash
# Verificar que no esté siendo usada por otra aplicación
# En Windows, cerrar Skype, Teams, etc.
```

### Error: "Modelo no cargado" 
```bash
# Asegurarse de entrenar primero
python face_recognizer.py train tu_nombre
```

### Error: "No se detectaron caras"
- Asegúrate de tener buena iluminación
- Posiciónate frente a la cámara
- La cara debe ocupar al menos 80x80 píxeles

### Falsos positivos (reconoce desconocidos como conocidos)
```bash
# Re-entrenar con más fotos variadas
python face_recognizer.py capture tu_nombre 8
python face_recognizer.py train tu_nombre
```

---

## 📊 Cómo funciona

### Tecnología utilizada
- **OpenCV**: Detección de caras y procesamiento de imágenes
- **scikit-learn**: Clustering (K-Means) y clasificación
- **FastAPI**: API REST moderna y rápida

### Algoritmo
1. **Extracción de características**: Histogramas, texturas, bordes y momentos geométricos
2. **Clustering**: Agrupa las variaciones naturales de tu cara
3. **Clasificación**: Calcula distancia a los clusters conocidos
4. **Decisión**: Si la distancia es menor al umbral → Conocido, sino → Desconocido

### Ventajas vs otros sistemas
- ❌ **face_recognition**: Problemas con dlib en Windows
- ❌ **Sistemas complejos**: Requieren cientos de fotos
- ✅ **Este sistema**: Solo 3 fotos, sin dependencias problemáticas

---

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

---

## 📞 Contacto

**Álvaro Muñoz Panadero**
- Email: [alvaromp.dev@gmail.com](mailto:alvaromp.dev@gmail.com)
- GitHub: [@tu-usuario](https://github.com/tu-usuario)

---

## 🎉 Agradecimientos

- OpenCV team por la excelente librería de visión por computadora
- scikit-learn por los algoritmos de machine learning
- FastAPI por el framework web moderno

---

⭐ **¡Si te ha sido útil este proyecto, dale una estrella en GitHub!**