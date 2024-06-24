# test_model1.py
import sys
import os

# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import models.testFastApiModel as model1

def test_model():
    # Example input data
    input_data = [[1.0, 2.0]]  # Modify according to your model's expected input format
    try:
        prediction = model1.predict(input_data)
        print(f"Prediction: {prediction}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_model()
