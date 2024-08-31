from flask import Flask, request, jsonify, render_template
from model import generate_test_cases_with_llm
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_test_cases', methods=['POST'])
def generate_test_cases():
    data = request.json
    context = data.get('context', '')
    images = data.get('images', [])

    try:
        test_cases = generate_test_cases_with_llm(context, images)
        return jsonify({"test_cases": test_cases})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)