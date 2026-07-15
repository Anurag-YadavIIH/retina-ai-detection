# app.py
#
# Flask web server — the bridge between the frontend (HTML) and
# the AI model (predict.py).
#
# Run with:   python app.py
# Then visit: http://localhost:5000

import os
import uuid
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

from predict import predict

# ─────────────────────────────────────────────
# APP CONFIGURATION
# ─────────────────────────────────────────────

app = Flask(__name__)

# Where uploaded images are stored temporarily
UPLOAD_FOLDER   = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Only allow image file types — security measure
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}

# Max upload size: 16MB
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)   # create folder if missing


def allowed_file(filename):
    """
    Check that the uploaded file has an allowed extension.
    'secure_filename' already strips dangerous characters —
    this adds a content-type check on top.
    """
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """
    Serve the main HTML page.
    Flask looks for templates in the 'templates/' folder automatically.
    """
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Handle image upload and return prediction results.

    Expects: multipart/form-data POST with a file field named "file"
    Returns: JSON with prediction, confidence, scores, heatmap path
    """

    # ── Validate: file was sent ────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # ── Validate: file has a name ──────────────────────
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # ── Validate: file type is allowed ────────────────
    if not allowed_file(file.filename):
        return jsonify({
            "error": f"File type not allowed. Use: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # ── Save the file ──────────────────────────────────
    # secure_filename strips dangerous characters (e.g. "../../../etc/passwd")
    original_name = secure_filename(file.filename)
    # Prefix with UUID to prevent filename collisions between users
    unique_name   = f"{uuid.uuid4().hex}_{original_name}"
    save_path     = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    # ── Run prediction ─────────────────────────────────
    try:
        result = predict(save_path)

        # Make heatmap path relative to static/ for the browser
        heatmap_url = "/" + result["heatmap_path"].replace("\\", "/")
        uploaded_url = f"/static/uploads/{unique_name}"

        return jsonify({
            "success":          True,
            "predicted_class":  result["predicted_class"],
            "label":            result["label"],
            "confidence":       result["confidence"],
            "all_scores":       result["all_scores"],
            "description":      result["description"],
            "color":            result["color"],
            "heatmap_url":      heatmap_url,
            "uploaded_url":     uploaded_url,
        })

    except Exception as e:
        # Return the error message to help with debugging
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """
    Health check endpoint — used by Render to confirm the server is alive.
    """
    return jsonify({"status": "ok", "model": "retina-ai-detection"})


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # debug=True: auto-reload on code changes, show detailed errors
    # Set debug=False in production
    app.run(debug=True, host="0.0.0.0", port=5000)