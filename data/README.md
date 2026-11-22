# Datos

Los datos no se versionan en este repositorio. Se descargan de las fuentes originales.

## BRACOL (conjunto principal)
- **Fuente:** Mendeley Data, DOI **10.17632/yy2k5y8mxg.1** (Krohling, Esgario y Ventura, 2019).
- **Captura:** teléfonos inteligentes, envés de la hoja sobre fondo blanco; etiquetado asistido por un experto.
- **Versión usada:** distribución con anotaciones YOLO (`bracol_yolo.zip`), carpeta `BRACOL_REVIEWED_ANNOTATIONS/BRACOL_REVIEWED` (1 744 imágenes de 2 048 × 1 024 px; train/valid/test = 1 213 / 351 / 180). Clases: Cercospora, Miner, Phoma, Rust.
- **Licencia:** CC BY 4.0 según los archivos de configuración de la versión de Roboflow. *Verificar también en la página de Mendeley.*

## JMuBEN y JMuBEN2 (validación externa)
- **Artículo:** Jepkoech et al. (2021), *Data in Brief* 36:107142, DOI 10.1016/j.dib.2021.107142.
- **Depósitos Mendeley:** 10.17632/t2r6rszp5c.1 y 10.17632/tgv3zb82nd.1 (el artículo y la página de Mendeley no coinciden en cuál es JMuBEN y cuál JMuBEN2; comprobar con el contenido de los zip).
- **Captura:** cámara Fujifilm X-T4, Mutira (Kirinyaga, Kenia). Las imágenes fueron recortadas, redimensionadas y con aumento de datos previo.
- **Contenido:** 5 clases (cercospora, healthy, leaf rust, miner, phoma); ~58 550 imágenes contadas en los archivos (58 555 según el artículo).
- **Licencia:** *verificar en las páginas de Mendeley.*

## Organización esperada
```
data/raw/
├── bracol_yolo.zip
├── jmuben.zip       (Cerscospora, Leaf rust, Phoma)
└── jmuben2.zip      (Healthy, Miner)
data/processed/      (lo genera el notebook 01 y 03)
├── bracol_crops.zip
├── jmuben_sample/   (500 imágenes por clase)
├── manifest_final.csv
├── split.csv
└── classes.json
```

## Cómo se obtienen los datos procesados
Ejecutar los notebooks 01 y 03. El notebook 01 extrae los zip, genera 8 226 recortes de BRACOL (8 091 tras eliminar 135 casi duplicados entre particiones), toma una muestra estratificada de JMuBEN (500 por clase) y calcula la calidad de datos. El notebook 03 verifica que no haya fuga y congela la partición.
