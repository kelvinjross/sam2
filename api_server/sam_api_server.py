from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/sam2', methods=['POST'])
def process_data():
    try:
        # Get JSON data from the POST request
        data = request.get_json()
        
        # Check if data is valid
        if not data:
            return jsonify({"error": "Invalid or missing JSON data"}), 400

        # Process the data (example: add two numbers)
        num1 = data.get('num1')
        num2 = data.get('num2')
        if num1 is None or num2 is None:
            return jsonify({"error": "num1 and num2 are required"}), 400

        # Compute the result
        result = num1 + num2
        
        # Return the result as JSON
        return jsonify({"num1": num1, "num2": num2, "result": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3030)

