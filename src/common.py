"""common.py - utilidades compartidas del proyecto (clasificación de enfermedades foliares del cafeto).

Se guarda en <PROJECT>/src/ y los notebooks 01-05 lo importan. Las funciones que usan
TensorFlow lo importan dentro de la función, para poder probar el resto sin TF.
"""
import os
import json
import time
import random
import hashlib
import numpy as np
import pandas as pd
from PIL import Image

PROJECT = os.environ.get("PROJECT_DIR", "/content/drive/MyDrive/proyecto_cafe_cnn")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
IMG_SIZE = 224


# ----------------------------------------------------------------------------
# Entorno y reproducibilidad
# ----------------------------------------------------------------------------
def paths():
    p = {k: os.path.join(PROJECT, k) for k in
         ["data/raw", "data/processed", "models", "results", "figures"]}
    for d in p.values():
        os.makedirs(d, exist_ok=True)
    return p


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)  # fija random, numpy y TF
    except ImportError:
        pass


# ----------------------------------------------------------------------------
# Inventario de imágenes (manifest), calidad de datos y duplicados
# ----------------------------------------------------------------------------
def list_images(root):
    out = []
    for dp, _, fn in os.walk(root):
        for f in fn:
            if f.lower().endswith(IMG_EXT):
                out.append(os.path.join(dp, f))
    return sorted(out)


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def dhash_bits(img, hash_size=8):
    """Hash perceptual de diferencias (64 bits) para detectar casi-duplicados."""
    g = img.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def build_manifest(root, source, label_fn=None, max_per_label=None, seed=42):
    """Recorre `root` y devuelve un DataFrame con una fila por imagen.

    label_fn(path) -> etiqueta cruda. Por defecto, el nombre de la carpeta padre.
    max_per_label: si se indica, toma una muestra aleatoria (con semilla) por etiqueta
    ANTES de procesar (útil para JMuBEN, que tiene ~58 mil imágenes).
    """
    label_fn = label_fn or (lambda p: os.path.basename(os.path.dirname(p)))
    items = [(p, str(label_fn(p)).strip().lower()) for p in list_images(root)]
    if max_per_label:
        rng = random.Random(seed)
        by = {}
        for p, l in items:
            by.setdefault(l, []).append(p)
        items = []
        for l, ps in by.items():
            if len(ps) > max_per_label:
                ps = rng.sample(ps, max_per_label)
            items += [(p, l) for p in ps]
        items.sort()
    rows = []
    for p, l in items:
        row = dict(path=p, source=source, label_raw=l, corrupt=False)
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im = im.convert("RGB")
                row["width"], row["height"] = im.width, im.height
                thumb = np.asarray(im.resize((32, 32)), dtype=np.float32)
                row["mean_r"], row["mean_g"], row["mean_b"] = thumb.reshape(-1, 3).mean(0)
                row["dhash"] = "".join("1" if b else "0" for b in dhash_bits(im))
            row["md5"] = file_md5(p)
        except Exception as e:  # imagen corrupta o ilegible
            row["corrupt"] = True
            row["error"] = str(e)
        rows.append(row)
    return pd.DataFrame(rows)


def near_dup_groups(dhash_strs, max_dist=4):
    """Agrupa imágenes con distancia de Hamming <= max_dist entre sus dhash.
    Devuelve un id de grupo por imagen. Costo O(n^2): usar solo con el dataset de entrenamiento."""
    n = len(dhash_strs)
    if n > 8000:
        print(f"Aviso: n={n}; la matriz de distancias será grande.")
    A = np.array([[int(c) for c in s] for s in dhash_strs], dtype=np.float32)
    dist = A.shape[1] - (A @ A.T + (1 - A) @ (1 - A).T)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ii, jj = np.where(np.triu(dist <= max_dist, k=1))
    for i, j in zip(ii, jj):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    return np.array([find(i) for i in range(n)])


# ----------------------------------------------------------------------------
# Particiones sin fuga de información
# ----------------------------------------------------------------------------
def make_splits(df, label_col="label", group_col="group", seed=42):
    """Partición estratificada y AGRUPADA (aprox. 2/3 train, 1/6 val, 1/6 test).
    Imágenes del mismo grupo (casi-duplicados / misma hoja) quedan siempre en el mismo conjunto."""
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    folds = [te for _, te in sgkf.split(df, df[label_col], groups=df[group_col])]
    split = np.array(["train"] * len(df), dtype=object)
    split[folds[0]] = "test"
    split[folds[1]] = "val"
    out = df.copy()
    out["split"] = split
    return out


def check_leakage(df, cols=("md5", "group")):
    """Verifica que no haya intersección entre train/val/test en cada columna."""
    d = df[df.split.isin(["train", "val", "test"])]
    ok = True
    for col in cols:
        sets = {s: set(d.loc[d.split == s, col]) for s in ["train", "val", "test"]}
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            inter = sets[a] & sets[b]
            print(f"[{col}] {a} ∩ {b}: {len(inter)}")
            ok &= len(inter) == 0
    return ok


# ----------------------------------------------------------------------------
# Datos para TensorFlow
# ----------------------------------------------------------------------------
def make_dataset(file_paths, int_labels, batch_size=32, training=False, seed=42, img_size=IMG_SIZE):
    """tf.data: devuelve imágenes float32 en [0,255]; el preprocesamiento va DENTRO del modelo.
    El aumento de datos se aplica solo si training=True (nunca en val/test)."""
    import tensorflow as tf
    AUTOTUNE = tf.data.AUTOTUNE

    def _load(path, y):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size), method="bilinear")
        return tf.cast(img, tf.float32), y

    ds = tf.data.Dataset.from_tensor_slices(
        (np.array(list(file_paths)), np.array(list(int_labels), dtype=np.int32)))
    if training:
        ds = ds.shuffle(len(file_paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE).batch(batch_size)
    if training:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.06),   # ±~21.6°
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


# ----------------------------------------------------------------------------
# Modelos
# ----------------------------------------------------------------------------
def build_cnn_scratch(num_classes, img_size=IMG_SIZE):
    import tensorflow as tf
    L = tf.keras.layers
    inp = tf.keras.Input((img_size, img_size, 3), name="image")
    x = L.Rescaling(1.0 / 255)(inp)
    for f in (32, 64, 128, 256):
        x = L.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = L.BatchNormalization()(x)
        x = L.MaxPooling2D()(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.5)(x)
    out = L.Dense(num_classes, activation="softmax", name="out")(x)
    return tf.keras.Model(inp, out, name="cnn_scratch")


def build_resnet(num_classes, img_size=IMG_SIZE, dropout=0.5, dense=256):
    """ResNet50 (ImageNet) + cabeza de clasificación. Devuelve (modelo, base)."""
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    L = tf.keras.layers
    inp = tf.keras.Input((img_size, img_size, 3), name="image")
    x = L.Lambda(preprocess_input, name="preprocess")(inp)
    base = ResNet50(weights="imagenet", include_top=False,
                    input_shape=(img_size, img_size, 3), name="resnet50")
    base.trainable = False
    x = base(x, training=False)
    x = L.GlobalAveragePooling2D(name="gap")(x)
    x = L.Dense(dense, activation="relu", name="dense_256")(x)
    x = L.Dropout(dropout, name="drop")(x)
    out = L.Dense(num_classes, activation="softmax", name="out")(x)
    return tf.keras.Model(inp, out, name="resnet50_cafe"), base


def unfreeze_top(base, from_layer="conv5_block1_1_conv"):
    """Fase 2 (fine-tuning): descongela el último bloque, manteniendo BatchNorm congelada."""
    import tensorflow as tf
    base.trainable = True
    reached = False
    for l in base.layers:
        if l.name == from_layer:
            reached = True
        l.trainable = reached and not isinstance(l, tf.keras.layers.BatchNormalization)


def compile_model(model, lr):
    import tensorflow as tf
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# ----------------------------------------------------------------------------
# Entrenamiento y guardado de artefactos
# ----------------------------------------------------------------------------
def save_probs(kind, seed, split, y, probs):
    d = os.path.join(paths()["results"], f"{kind}_seed{seed}")
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, f"y_{split}.npy"), np.asarray(y))
    np.save(os.path.join(d, f"probs_{split}.npy"), np.asarray(probs))


def run_experiment(kind, seed, df, classes, epochs_head=15, epochs_ft=20, batch_size=32):
    """Entrena `kind` in {'cnn_scratch','resnet50'} y guarda historial, pesos y probabilidades
    de val/test/external. El conjunto test se predice UNA vez, al final, sin usarse para decidir nada."""
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight
    P = paths()
    set_seed(seed)
    c2i = {c: i for i, c in enumerate(classes)}
    tr, va, te = (df[df.split == s] for s in ["train", "val", "test"])
    ex = df[(df.split == "external") & (df.label.isin(classes))]
    ds_tr = make_dataset(tr.path, tr.label.map(c2i), batch_size, True, seed)
    ds_va = make_dataset(va.path, va.label.map(c2i), batch_size, False)

    y_tr = tr.label.map(c2i).values
    w = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_tr)
    class_weight = {i: float(v) for i, v in enumerate(w)}

    def cbs():
        return [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)]

    t0 = time.time()
    history = []
    if kind == "cnn_scratch":
        model = build_cnn_scratch(len(classes))
        compile_model(model, 1e-3)
        h = model.fit(ds_tr, validation_data=ds_va, epochs=epochs_head + epochs_ft,
                      class_weight=class_weight, callbacks=cbs(), verbose=2)
        history.append({"phase": "scratch", **h.history})
    elif kind == "resnet50":
        model, base = build_resnet(len(classes))
        compile_model(model, 1e-3)
        h1 = model.fit(ds_tr, validation_data=ds_va, epochs=epochs_head,
                       class_weight=class_weight, callbacks=cbs(), verbose=2)
        history.append({"phase": "warmup", **h1.history})
        unfreeze_top(base)
        compile_model(model, 1e-5)   # recompilar tras cambiar trainable
        h2 = model.fit(ds_tr, validation_data=ds_va, epochs=epochs_ft,
                       class_weight=class_weight, callbacks=cbs(), verbose=2)
        history.append({"phase": "finetune", **h2.history})
    else:
        raise ValueError(kind)
    train_seconds = time.time() - t0

    out_dir = os.path.join(P["results"], f"{kind}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump({"history": history, "train_seconds": train_seconds,
                   "params": int(model.count_params()), "classes": list(classes),
                   "class_weight": class_weight}, f, default=float)
    model.save_weights(os.path.join(P["models"], f"{kind}_seed{seed}.weights.h5"))

    for name, d in [("val", va), ("test", te), ("external", ex)]:
        if len(d) == 0:
            continue
        ds = make_dataset(d.path, d.label.map(c2i), 64, False)
        save_probs(kind, seed, name, d.label.map(c2i).values, model.predict(ds, verbose=0))
    return model


# ----------------------------------------------------------------------------
# Baselines clásicos (sin deep learning)
# ----------------------------------------------------------------------------
def color_hist_features(path, bins=16):
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB").resize((128, 128)))
    feats = [np.histogram(a[..., c], bins=bins, range=(0, 255))[0] for c in range(3)]
    f = np.concatenate(feats).astype(np.float32)
    return f / f.sum()


def run_baselines(df, classes, seed=42):
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    c2i = {c: i for i, c in enumerate(classes)}
    feats = {}
    for s in ["train", "val", "test", "external"]:
        d = df[df.split == s]
        if s == "external":
            d = d[d.label.isin(classes)]
        if len(d):
            feats[s] = (np.stack([color_hist_features(p) for p in d.path]), d.label.map(c2i).values)
    Xtr, ytr = feats["train"]
    models = {"dummy": DummyClassifier(strategy="prior"),
              "logreg_color": make_pipeline(StandardScaler(),
                                            LogisticRegression(max_iter=3000, class_weight="balanced"))}
    for kind, m in models.items():
        m.fit(Xtr, ytr)
        for s in ["val", "test", "external"]:
            if s in feats:
                X, y = feats[s]
                probs = np.zeros((len(X), len(classes)), dtype=np.float32)
                probs[:, m.classes_] = m.predict_proba(X)
                save_probs(kind, seed, s, y, probs)


# ----------------------------------------------------------------------------
# Métricas, incertidumbre e interpretabilidad
# ----------------------------------------------------------------------------
def compute_metrics(y_true, probs):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    y_pred = probs.argmax(1)
    labels = list(range(probs.shape[1]))
    m = dict(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        macro_precision=precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        macro_recall=recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        macro_f1=f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    )
    try:
        m["roc_auc_ovr"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=labels)
    except ValueError:
        m["roc_auc_ovr"] = np.nan   # p. ej. clases ausentes en el conjunto externo
    return m


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=42, alpha=0.05):
    """IC del macro-F1 por bootstrap sobre las imágenes del conjunto evaluado."""
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        vals.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def gradcam(model, img, class_index=None, layer="conv5_block3_out"):
    """Grad-CAM para el modelo ResNet50 de build_resnet(). img: array HxWx3 en [0,255] (224x224)."""
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import preprocess_input
    base = model.get_layer("resnet50")
    conv_model = tf.keras.Model(base.input, base.get_layer(layer).output)
    head = [model.get_layer(n) for n in ["gap", "dense_256", "drop", "out"]]
    x = preprocess_input(np.array(img, dtype=np.float32)[None].copy())
    with tf.GradientTape() as tape:
        conv_out = conv_model(x)
        tape.watch(conv_out)
        y = conv_out
        for l in head:
            y = l(y, training=False)
        c = int(tf.argmax(y[0])) if class_index is None else class_index
        score = y[:, c]
    grads = tape.gradient(score, conv_out)
    alpha = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * alpha, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], img.shape[:2], method="bilinear")[..., 0].numpy()
    return cam, c, y[0].numpy()
