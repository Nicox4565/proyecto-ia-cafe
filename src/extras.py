# extras.py - utilidades adicionales (no modifica common.py)
import os
import shutil
import zipfile
import numpy as np
import pandas as pd
import common as C

LOCAL_CROPS = '/content/bracol_crops'
LOCAL_JM = '/content/jmuben_sample'


def prepare_local():
    # Copia los datos a disco local (mucho más rápido que Drive) y devuelve el DataFrame de split.csv
    # con las rutas ya apuntando a /content.
    P = C.paths()
    if not os.path.exists(LOCAL_CROPS):
        with zipfile.ZipFile(f"{P['data/processed']}/bracol_crops.zip") as z:
            z.extractall(LOCAL_CROPS)
    src = f"{P['data/processed']}/jmuben_sample"
    if not os.path.exists(LOCAL_JM):
        shutil.copytree(src, LOCAL_JM)
    df = pd.read_csv(f"{P['data/processed']}/split.csv")
    df['path'] = df.path.str.replace(src, LOCAL_JM, regex=False)
    faltan = int((~df.path.map(os.path.exists)).sum())
    assert faltan == 0, f'{faltan} rutas no existen'
    return df


def degrade_batch(x, p=0.5, lo=48, hi=128):
    # Con probabilidad p, reduce cada imagen a un lado aleatorio en [lo, hi) y la vuelve a ampliar.
    import tensorflow as tf
    x = tf.convert_to_tensor(x, tf.float32)
    H, W = int(x.shape[1]), int(x.shape[2])

    def one(img):
        def do():
            s = tf.random.uniform([], lo, hi, dtype=tf.int32)
            small = tf.image.resize(img, tf.stack([s, s]), method='bilinear', antialias=True)
            return tf.image.resize(small, (H, W), method='bilinear')
        out = tf.cond(tf.random.uniform([]) < p, do, lambda: img)
        out.set_shape((H, W, 3))
        return out
    return tf.map_fn(one, x)


_orig_make_dataset = None


def enable_degrade(p=0.5):
    # Reemplaza C.make_dataset por una versión que degrada resolución solo con training=True.
    # run_experiment busca make_dataset en el módulo common, así que no hay que tocar nada más.
    global _orig_make_dataset
    if _orig_make_dataset is None:
        _orig_make_dataset = C.make_dataset

    def wrapped(file_paths, int_labels, batch_size=32, training=False, seed=42, img_size=C.IMG_SIZE):
        import tensorflow as tf
        ds = _orig_make_dataset(file_paths, int_labels, batch_size, training, seed, img_size)
        if training and p > 0:
            ds = ds.map(lambda x, y: (degrade_batch(x, p), y), num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
    C.make_dataset = wrapped
    print(f'degradación de resolución ACTIVADA (p={p})')


def disable_degrade():
    global _orig_make_dataset
    if _orig_make_dataset is not None:
        C.make_dataset = _orig_make_dataset
        _orig_make_dataset = None
    print('degradación de resolución DESACTIVADA')
