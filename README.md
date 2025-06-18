# Clasificador de Imágenes de Ductos con PyTorch

Este proyecto utiliza una red neuronal convolucional basada en **EfficientNet-B0** para clasificar imágenes de ductos. Está implementado con PyTorch y realiza entrenamiento, validación, evaluación y predicción a partir de imágenes etiquetadas según su nombre de archivo.

---

## 📁 Estructura esperada del dataset

Coloca tus imágenes en una carpeta llamada `img/`. El nombre de cada imagen debe seguir un patrón que incluya `_dX_` (ductos totales), `_oY_` (ductos ocupados) y `_vZ` (ductos vacíos), donde X, Y y Z representan los valores numéricos correspondientes.

### Ejemplos de nombres válidos:
- `img490_d2_o0_v2.png`
- `img102_d1_o1_v3.jpg`

### Desglose del formato:

- `imgXXX`: identificador único de la imagen.
- `_dX_`: clase del ducto total (por ejemplo, `d2` indica 2 ductos).
- `_oY_`: número de ductos ocupados (Y).
- `_vZ`: número de ductos vacíos (Z).

---

## ⚙️ Requisitos

- Python 3.9+
- PyTorch
- `torchvision`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `Pillow` (PIL)
- `transformers`
- `tqdm`
- `pycocotools`
- `gradio`
- `GroundingDINO` (instalación desde GitHub)
- `numpy` (versión específica)

Puedes instalar las dependencias con los siguientes comandos:

```bash
# Instalación de librerías generales y de PyTorch
pip install opencv-python pillow numpy transformers tqdm pycocotools torch torchvision torchaudio scikit-learn pandas gradio

# Instalación de GroundingDINO desde su repositorio de GitHub
pip install git+https://github.com/IDEA-Research/GroundingDINO.git

# Para asegurar la compatibilidad con GroundingDINO, se recomienda una versión específica de numpy
pip install "numpy<2.0"
```

**Nota:** Podrías ver advertencias durante la instalación relacionadas con `pip` y `sysconfig` ("WARNING: Value for prefixed-purelib does not match"). Estas generalmente no afectan el funcionamiento del proyecto. Si encuentras problemas, asegúrate de que tu entorno virtual esté activado y que tengas las herramientas de compilación necesarias en tu sistema.

## 🧹 Preparación y Limpieza del Dataset

El notebook `LimpiezaDatasheet.ipynb` contiene scripts esenciales para preparar y verificar el dataset:

1.  **Renombrar archivos de `imagen` a `img`:** Este script busca archivos `*.png` que comiencen con "imagen" y los renombra a "img".
    ```python
    import os
    carpeta = 'img'
    # ... (resto del código para renombrar)
    ```
    **Output Ejemplo:**
    ```
    Archivos encontrados en 'img': ['.DS_Store', 'img000_d6_o4_v2.png', ...]
    No se encontró ningún archivo .png que empiece con 'imagen'.
    ```

2.  **Conteo y Verificación de Archivos:** Este script verifica la cantidad de imágenes, busca números faltantes y duplicados en la secuencia de nombres de archivo.
    ```python
    import os
    import re
    # ... (resto del código para conteo)
    ```
    **Output Ejemplo:**
    ```
    🔢 Total únicos encontrados: 836
    📛 Números faltantes en la secuencia (0–489): []
    ⚠️ Números duplicados: [378]
    ```

3.  **Cambiar extensiones a minúsculas:** Este script convierte las extensiones de archivo (`.PNG`, `.JPG`, `.JPEG`) a minúsculas.
    ```python
    import os
    # ... (resto del código para cambiar a minúsculas)
    ```
    **Output Ejemplo:**
    ```
    ✅ Renombrado: img000_d6_o4_v2.png -> img000_d6_o4_v2.png
    ```

Puedes ejecutar las celdas de este notebook para asegurar que tu dataset esté en el formato correcto antes del entrenamiento.

## 📊 Análisis de Distribución de Clases

El notebook `conteo_por_clases.ipynb` genera un gráfico de barras que muestra la distribución de las clases de ductos (`dX`) en tu dataset (filtrando las clases hasta `d6`).

```python
import os
import re
from collections import Counter
import matplotlib.pyplot as plt
# ... (código completo del notebook)
```

**Output Ejemplo:**
```
Counter({1: 225, 2: 180, 4: 125, 3: 105, 6: 92, 5: 53, 0: 41})
```
*(Se generará un gráfico de barras visualizando esta distribución).*

## 🚀 Entrenamiento del Modelo (`Entrenamiento_modelo.ipynb`)

El notebook `Entrenamiento_modelo.ipynb` se encarga de todo el proceso de entrenamiento del modelo de clasificación multitarea (para `dX` y `oX`).

El código realiza:
* Carga y partición de las imágenes (80% entrenamiento, 20% validación).
* Aumento de datos (`RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`, `RandomAffine`) para el entrenamiento.
* Carga del modelo **EfficientNet-B0** preentrenado con `ImageNet` weights.
* Implementación de `FocalLoss` como función de pérdida para manejar desequilibrio de clases.
* Entrenamiento durante 10 épocas.
* Evaluación del modelo con reporte de clasificación y matriz de confusión.

Para ejecutar el entrenamiento, abre el notebook `Entrenamiento_modelo.ipynb` y corre todas sus celdas.

### 📈 Evaluación

Al final del entrenamiento, el script imprime:
* **Reporte de Clasificación:** (`precision`, `recall`, `f1-score`) para cada clase de `dX` y `oX`.
* **Accuracy Global:** Porcentajes de precisión general para `dX` y `oX`.
* **Matrices de Confusión:** Visualizadas con `matplotlib` (una para `dX` y otra para `oX`).

**Output Ejemplo (Consola):**
```
Clases dX: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
Clases oX: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
Época 1/10 - Loss Entrenamiento: 2.1734, Validación: 1.7454
...
Época 10/10 - Loss Entrenamiento: 0.0026, Validación: 1.5679
✅ Modelo guardado como 'modelo_ductos_multitarea_efnet.pth'

--- Reporte para dX (ductos totales) ---
              precision    recall  f1-score   support

          d0       0.62      0.56      0.59         9
          d1       0.85      0.65      0.74        43
          d2       0.53      0.56      0.54        34
          d3       0.33      0.43      0.38        21
          d4       0.41      0.52      0.46        27
          d5       0.50      0.62      0.55        13
          d6       0.71      0.53      0.61        19
         d7+       0.00      0.00      0.00         2

    accuracy                           0.55       168
   macro avg       0.50      0.48      0.48       168
weighted avg       0.58      0.55      0.56       168


--- Reporte para oX (ductos ocupados) ---
              precision    recall  f1-score   support

          o0       0.64      0.53      0.58        17
          o1       0.69      0.67      0.68        52
          o2       0.50      0.65      0.57        46
          o3       0.28      0.35      0.31        20
          o4       0.22      0.13      0.17        15
          o5       0.56      0.45      0.50        11
          o6       0.00      0.00      0.00         6
         o7+       0.00      0.00      0.00         1

    accuracy                           0.52       168
   macro avg       0.36      0.35      0.35       168
weighted avg       0.50      0.52      0.51       168


✅ Accuracy global dX: 55.36%
✅ Accuracy global oX: 52.38%
```

### 💾 Guardado del modelo

El modelo entrenado se guarda automáticamente como: `modelo_ductos_multitarea_efnet.pth`.

## 🔍 Predicción de nuevas imágenes (`Predecir_imagen.ipynb`)

El notebook `Predecir_imagen.ipynb` permite cargar una imagen individual y utilizar el modelo entrenado para predecir el número total de ductos (`dX`) y el número de ductos ocupados (`oX`).

Asegúrate de que el archivo `modelo_ductos_multitarea_efnet.pth` esté en el mismo directorio que este notebook, y actualiza la variable `ruta_img` con la ruta a tu imagen.

```python
import torch
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torchvision.transforms as transforms

# ... (código completo del notebook)

# Ruta manual de la imagen a predecir
ruta_img = "img_predic.jpg" # <--- ¡CAMBIA ESTA RUTA A TU IMAGEN!

# ... (resto del código para la predicción)
```

**Output Ejemplo:**
```
📸 Imagen: img_predic.jpg
🔢 Ductos totales (dX): 4
✅ Ductos ocupados (oX): 3
⬜ Ductos vacíos     : 1
```

## 🧠 Arquitectura del Modelo

* **Modelo Base:** EfficientNet-B0 preentrenado en ImageNet.
* **Capas de Clasificación:** Se reemplazan las capas finales de clasificación por dos capas lineales separadas, una para predecir el número total de ductos (`dX`) y otra para el número de ductos ocupados (`oX`), adaptadas al número de clases detectadas dinámicamente en el dataset.
* **Función de Pérdida:** Se utiliza `FocalLoss` para ambas tareas de clasificación, lo que ayuda a manejar el desequilibrio en el número de muestras por clase.

## 📌 Notas

* El dataset es leído dinámicamente desde los nombres de archivo, extrayendo las etiquetas de `dX` y `oX` (no se requiere un archivo CSV de etiquetas separado).
* Las clases se agrupan: cualquier número de ductos mayor a 6 se agrupa en una clase "7+" para ambas tareas (`dX` y `oX`).
* Asegúrate de tener una GPU disponible y configurada para PyTorch (`torch.cuda.is_available()`) para acelerar significativamente el entrenamiento (opcional, el código por defecto usará CPU si no hay GPU).
