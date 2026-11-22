# Modelos

Los pesos entrenados no se incluyen en el repositorio por su tamaño (ResNet50 ≈ 100 MB por semilla).

- **Modelo seleccionado:** `resnet50_seed44.weights.h5` (la semilla con mayor macro-F1 en validación).
- Los notebooks los guardan en `MyDrive/proyecto_cafe_cnn/models/` (CNN desde cero y ResNet50; la variante con degradación en `variants/deg/models/`).
- **Para obtenerlos:** [https://drive.google.com/file/d/1OWd_6sgqTFmhjWQuf-fBLLeJeRDbrs6O/view?usp=sharing].
- **Para regenerarlos:** ejecutar el notebook 04 (semillas 42, 43 y 44).
