import flask
from PIL import Image
import requests

from animal10 import transform_image
from predict import predict_label

app = flask.Flask(__name__)
app.config["JSON_AS_ASCII"] = False

@app.route("/predict", methods=["POST"])
def predict():
    response = {
        "success": False,
        "Content-Type": "application/json"
    }

    if flask.request.method == "POST":
        if flask.request.get_json().get("image_url"):
            image = Image.open(requests.get(flask.request.get_json().get("image_url"), stream=True).raw).convert("RGB")
            label = predict_label(image)

            response["label"] = label
            response["success"] = True

    return flask.jsonify(response)


if __name__ == "__main__":
    app.run()
