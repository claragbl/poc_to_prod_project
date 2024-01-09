from flask import Flask, render_template, jsonify, request
from predict.predict import run as run_predict

app = Flask(__name__, static_url_path='/static')


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict', methods=['GET'])
def predict():
    artefacts_path = '/Users/claragaubil/Documents/EPF/EPF_5A/Theory & Practice/From PoC to Prod/TP2/poc-to-prod-capstone/train/data/artefacts/2024-01-09-12-11-21'
    model = run_predict.TextPredictionModel.from_artefacts(artefacts_path)

    # Get user_text from the query parameters
    user_text = request.args.get('user_text', '')

    # prediction
    predictions = model.predict([user_text])
    results_label = [model.labels_to_index[str(idx)] for idx in predictions[0]]

    # prediction result
    return jsonify({'input_text': user_text, 'predictions': results_label})


if __name__ == '__main__':
    app.run(debug=True, port=5002)
