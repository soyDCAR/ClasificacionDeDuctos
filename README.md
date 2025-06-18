# Clasificador de Imágenes de Ductos con PyTorch

Este proyecto utiliza una red neuronal convolucional basada en ResNet18 para clasificar imágenes de ductos. Está implementado con PyTorch y realiza entrenamiento, validación, evaluación y predicción a partir de imágenes etiquetadas según su nombre de archivo.

---

##📁 Estructura esperada del dataset

Coloca tus imágenes en una carpeta llamada img/.
El nombre de cada imagen debe seguir un patrón que incluya _dX_, donde X representa la clase del ducto.

Ejemplos de nombres válidos:
img490_d2_o0_v2
img102_d1_o1_v3
Desglose del formato:

imgXXX: identificador único de la imagen
_dX_: clase del ducto (por ejemplo, d2 indica clase 2)
_oY_: número de ductos ocupados (Y)
_vZ: número de ductos vacíos (Z)


---

## ⚙️ Requisitos

- Python 3.8+
- PyTorch
- torchvision
- pandas
- matplotlib
- scikit-learn
- PIL

Puedes instalar las dependencias con:

```bash
pip install torch torchvision pandas matplotlib scikit-learn pillow
```

## 🚀 Entrenamiento del modelo

El código realiza:

Carga y partición de las imágenes (80% entrenamiento, 20% validación)
Aumento de datos para el entrenamiento
Carga del modelo ResNet18 preentrenado
Entrenamiento durante 10 épocas
Evaluación del modelo con matriz de confusión y clasificación
Para ejecutar el entrenamiento, simplemente corre:
```bash
python ClasificaciónDuctos.pynb
```

## 📈 Evaluación

### Al final del entrenamiento, el script imprime:

Un reporte de clasificación (precision, recall, f1-score)
Una matriz de confusión visualizada con matplotlib
💾 Guardado del modelo

El modelo entrenado se guarda automáticamente como:

modelo_ductos.pth

## 🔍 Predicción de nuevas imágenes

El script incluye una función para predecir imágenes nuevas:

predecir_imagen("ruta/a/tu/imagen.jpg")
Ejemplo:

print(predecir_imagen("img_predic.jpg"))
🧠 Arquitectura del modelo

Modelo base: ResNet18 preentrenado en ImageNet
Capa final adaptada al número de clases detectadas dinámicamente
📌 Notas

El dataset es leído dinámicamente desde los nombres de archivo (no se requiere archivo CSV de etiquetas).
Asegúrate de tener una GPU disponible para acelerar el entrenamiento (opcional).
