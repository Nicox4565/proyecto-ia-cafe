"""Prototipo: clasificador de enfermedades foliares del cafeto (ResNet50 + Grad-CAM).

Uso:
    python prototipo/app.py --weights models/resnet50_seed44.weights.h5
    python prototipo/app.py --weights models/resnet50_seed44.weights.h5 --share   # enlace público temporal

Entrada esperada: un RECORTE de lesión (como los usados en el entrenamiento).
Con una foto de hoja completa el resultado no está validado (ver informe, secciones 21.5 y 23).
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import common as C  # noqa: E402

CLASSES = ["cercospora", "miner", "phoma", "rust"]   # mismo orden que data/processed/classes.json


def load_model(weights):
    model, _ = C.build_resnet(len(CLASSES))
    model.load_weights(weights)
    return model


def make_predict(model):
    def predict(img):
        if img is None:
            return None, None
        arr = np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32)
        cam, _, probs = C.gradcam(model, arr)
        heat = plt.get_cmap("jet")(cam)[..., :3]
        overlay = (0.6 * arr / 255 + 0.4 * heat).clip(0, 1)
        return {c: float(p) for c, p in zip(CLASSES, probs)}, (overlay * 255).astype("uint8")
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(HERE, "..", "models", "resnet50_seed44.weights.h5"))
    ap.add_argument("--share", action="store_true", help="crea un enlace público temporal (Gradio)")
    args = ap.parse_args()
    if not os.path.exists(args.weights):
        sys.exit(f"No se encuentran los pesos: {args.weights}\nVer models/README.md")

    import gradio as gr
    demo = gr.Interface(
        fn=make_predict(load_model(args.weights)),
        inputs=gr.Image(type="pil", label="Recorte de lesión en hoja de cafeto"),
        outputs=[gr.Label(num_top_classes=len(CLASSES), label="Diagnóstico estimado"), gr.Image(label="Grad-CAM")],
        title="Clasificador de enfermedades foliares del cafeto",
        description="Prototipo de investigación. No sustituye el diagnóstico de un especialista.",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
