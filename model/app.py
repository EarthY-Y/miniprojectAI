from flask import Flask, render_template, request, jsonify

from chat import get_response

app = Flask(__name__)


@app.get("/")
# @app.route("/", method=["GET"])
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("massage")
    #TODO: check if text is valid
    response = get_response(text)
    massage = {"answer": response}
    return jsonify(massage)

if __name__ =="__main__":
    app.run(debug=True)