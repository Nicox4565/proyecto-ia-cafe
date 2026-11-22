# Clasificación de enfermedades foliares del cafeto con redes neuronales convolucionales

Proyecto final del curso de Inteligencia Artificial.
**Autora:** Bazan Hurtado Susan Carol · **Docente:** Dr. Ciro Rodríguez Rodríguez

Se compara una CNN entrenada desde cero y una ResNet50 con transferencia de aprendizaje para clasificar cuatro enfermedades foliares del cafeto (cercospora, minador, phoma y roya) a partir de recortes de lesiones de **BRACOL**, con validación externa en imágenes de **JMuBEN** (Kenia). El informe completo está en `docs/INFORME_IA_CAFE.docx`.

## 1. ¿Cuál es el problema?
La roya amarilla y otras enfermedades foliares causan pérdidas considerables en la caficultura peruana (por ejemplo, SENASA reportó una incidencia del 44 % en 12 regiones en 2023, según Agraria.pe; ver informe, sección 5). Su manejo oportuno exige reconocer qué enfermedad afecta a la hoja, y no hay un modelo cuya generalización a imágenes de otra procedencia se haya medido con rigor.

## 2. ¿Cuál es el objetivo?
Desarrollar y evaluar un sistema de clasificación de enfermedades foliares del cafeto basado en CNN, con imágenes reales de BRACOL y validación externa con JMuBEN (objetivos OE1 a OE6 en el informe, sección 8).

## 3. ¿Qué dataset se utilizó?
| Dataset | Uso | Fuente |
|---|---|---|
| BRACOL (versión con anotaciones YOLO) | Entrenamiento, validación y prueba (8 091 recortes de lesión) | Krohling, Esgario y Ventura (2019), DOI 10.17632/yy2k5y8mxg.1 |
| JMuBEN / JMuBEN2 | Validación externa (2 000 imágenes de 4 clases; 500 por clase) | Jepkoech et al. (2021), *Data in Brief* 36:107142, DOI 10.1016/j.dib.2021.107142 |

Detalles, licencias y procedencia en `data/README.md`.

## 4. ¿Cómo obtener los datos?
Los datos **no se incluyen en el repositorio** (tamaño y licencias). Ver `data/README.md`: se descargan de Mendeley Data y se colocan como `data/raw/bracol_yolo.zip`, `data/raw/jmuben.zip` y `data/raw/jmuben2.zip`. El notebook 01 los extrae, genera los recortes y arma el manifest.

## 5. ¿Cómo instalar el proyecto?
Se ejecutó en **Google Colab (GPU T4)**. Para un entorno propio:
```bash
pip install -r requirements.txt
```
Las versiones exactas provienen de las ejecuciones reales (TensorFlow 2.21, Keras 3.13).

## 6. ¿Cómo ejecutar el código?
1. Crea en Google Drive la carpeta `proyecto_cafe_cnn/` con las subcarpetas `src/` (copia aquí `src/common.py` y `src/extras.py`) y `data/raw/` (con los tres zip).
2. Abre en Colab, **en este orden**, los notebooks de `notebooks/`:

| Notebook | Contenido |
|---|---|
| `01_data_understanding.ipynb` | Diagnóstico de los datos, recortes, calidad y data card |
| `02_eda.ipynb` | Análisis exploratorio y prueba de atajos |
| `03_preprocessing.ipynb` | Verificación de fuga, partición congelada, aumento de datos |
| `04_modeling.ipynb` | Baselines, CNN desde cero, ResNet50 y variante A/B |
| `05_evaluation.ipynb` | Métricas, significancia, validación externa, Grad-CAM y prototipo |

Cada notebook indica sus entradas y salidas; los resultados se guardan en Drive y las celdas de entrenamiento se saltan lo que ya está hecho.

## 7. ¿Cómo entrenar el modelo?
Notebook `04_modeling.ipynb`: ejecuta primero la prueba rápida (sección 2) y luego las secciones 3 y 4. El entrenamiento completo (3 modelos × 3 semillas) tarda unas 3 a 4 horas en una GPU T4.

## 8. ¿Cómo reproducir los experimentos?
- Semillas: **42, 43 y 44**.
- Partición congelada en `data/processed/split.csv` (la genera el notebook 03) y clases en `classes.json`.
- Versiones del entorno en `requirements.txt` y en `requirements_freeze.txt` (generado por el notebook 01).
- La selección de modelo y semilla se hace **solo con validación**; test y conjunto externo solo se reportan.

## 9. ¿Cómo ejecutar el prototipo?
- En Colab: sección 9 de `05_evaluation.ipynb` (genera un enlace público **temporal**, válido solo mientras la sesión esté activa).
- En local o Colab, desde la terminal:
```bash
python prototipo/app.py --weights models/resnet50_seed44.weights.h5
```
Los pesos se describen en `models/README.md`. **Entrada esperada:** un recorte de lesión; con una foto de hoja completa el resultado no está validado.

## 10. ¿Cuáles fueron los resultados?
Macro-F1 (media ± desviación entre 3 semillas):

| Modelo | Validación | Prueba | Externa (JMuBEN) |
|---|---|---|---|
| Dummy (frecuencia de clase) | 0.213 | 0.218 | 0.100 |
| Regresión logística + color | 0.412 | 0.324 | 0.288 |
| CNN desde cero | 0.622 ± 0.005 | 0.667 ± 0.031 | 0.294 ± 0.046 |
| **ResNet50** | **0.723 ± 0.006** | **0.780 ± 0.006** | 0.414 ± 0.035 |
| ResNet50 + degradación de resolución | 0.721 ± 0.001 | 0.784 ± 0.017 | 0.442 ± 0.015 |

- ResNet50 supera a la CNN desde cero por 0.096 de macro-F1 (IC 95 % por grupos [0.043, 0.145]) y a la regresión logística por 0.460.
- La degradación de resolución **no se adoptó**: no mejoró la validación (−0.002).
- Cae fuera de BRACOL: en JMuBEN el macro-F1 es 0.414 y el modelo seleccionado no reconoce ninguna imagen de cercospora.
- Limitaciones y amenazas a la validez: informe, sección 23.

## Estructura
```
PROYECTO-IA-CAFE/
├── README.md
├── LICENSE · requirements.txt · .gitignore
├── data/            README.md (cómo obtener los datos)
├── notebooks/       01 a 05 (.ipynb)
├── src/             common.py, extras.py
├── prototipo/       app.py
├── models/          README.md (pesos)
├── results/         figures/ (y tablas CSV generadas por los notebooks)
└── docs/            INFORME_IA_CAFE.docx
```

## Licencia
Código bajo licencia MIT (ver `LICENSE`). Los datos pertenecen a sus autores y se usan bajo sus licencias (ver `data/README.md`).
