from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load sentiment model
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'review' not in data:
        return jsonify({'error': 'No review provided'}), 400

    review = data['review']
    result = sentiment_pipeline(review)[0]
    return jsonify({
        'label': result['label'],
        'confidence': round(result['score'], 4)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
