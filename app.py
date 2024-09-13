import os
import sys
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image

from fundus_v2  import fundus_predictor, fundus_cnn_el_classifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
subdirectory_path = os.path.join(current_dir, 'fundus_v2')
sys.path.append(subdirectory_path)

app = Flask(__name__, template_folder='templates')
cnn_ensemble_classifier = fundus_cnn_el_classifier.CNNEnsembleClassifier()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        # Read the uploaded image as a PIL Image
        pil_image = Image.open(file)
        processed_tensor_image = cnn_ensemble_classifier.preprocess_image(pil_image=pil_image)
        predicted_class, prediction_matrix = cnn_ensemble_classifier.predict(processed_tensor_image)
        print(f"Predicted class: {predicted_class}")
        print(f"Prediction matrix: {prediction_matrix}")
        
        # Get the selected model type from the dropdown
        selected_model = request.form.get("model")

        return render_template("result.html", predicted_class=predicted_class, prediction_matrix=prediction_matrix, selected_model=selected_model)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
