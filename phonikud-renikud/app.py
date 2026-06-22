"""
uv sync
wget https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx
wget https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx -O renikud.onnx
uv run app.py
"""

import argparse

from flask import Flask, request, jsonify
from phonikud_onnx import Phonikud
from phonikud import phonemize
from renikud_onnx import G2P

app = Flask(__name__)
phonikud = Phonikud("phonikud-1.0.int8.onnx")

renikud = G2P("renikud.onnx")


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/generate", methods=["POST"])
def generate():
    mode = request.form["mode"]
    text = request.form.get("text", "")
    phonemes = request.form.get("phonemes", "")
    backend = request.form.get("backend", "renikud")
    use_renikud = backend != "phonikud"

    if use_renikud:
        if mode == "text":
            with_diacritics = phonikud.add_diacritics(text)
            phonemes = renikud.phonemize(text)
        elif mode == "diacritics":
            with_diacritics = text
            phonemes = renikud.phonemize(text)
        else:
            with_diacritics = None
    else:
        if mode == "text":
            with_diacritics = phonikud.add_diacritics(text)
            phonemes = phonemize(with_diacritics)
        elif mode == "diacritics":
            with_diacritics = text
            phonemes = phonemize(with_diacritics)
        else:
            with_diacritics = None

    return jsonify(
        {
            "backend": "renikud" if use_renikud else "phonikud",
            "diacritics": with_diacritics,
            "phonemes": phonemes,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    app.run(debug=True, host=args.host, port=args.port)
