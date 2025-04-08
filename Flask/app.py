from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load trained model and scalers
X_SCALER_PATH = os.path.join(os.path.dirname(__file__), '../scalar/X_scaler.npy')
Y_SCALER_PATH = os.path.join(os.path.dirname(__file__), '../scalar/y_scaler.npy')

model_path = "C:/Traffic_pred_Project/Model/lstm_model_optimized.h5"
model = tf.keras.models.load_model(model_path, compile=False)
X_scaler = np.load(X_SCALER_PATH, allow_pickle=True).item()
y_scaler = np.load(Y_SCALER_PATH, allow_pickle=True).item()

SEQUENCE_LENGTH = 4
TARGET_COLUMNS = ["Traffic_Volume", "Congestion_Level", "Average_Speed"]

@app.route('/')
def home():
    return render_template('index.html')

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid GUI warning
import matplotlib.pyplot as plt
import io
import base64
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read CSV file
        df = pd.read_csv(file)

        # Ensure target columns exist in the file
        missing_targets = [col for col in TARGET_COLUMNS if col not in df.columns]
        if missing_targets:
            return jsonify({'error': f'Missing target columns: {missing_targets}'}), 400

        # Extract actual values
        actual_values = df[TARGET_COLUMNS].values[-1]
        df = df.drop(columns=TARGET_COLUMNS, errors='ignore')  # Remove target columns

        # Ensure correct number of features
        if df.shape[1] != X_scaler.n_features_in_:
            return jsonify({'error': f'Expected {X_scaler.n_features_in_} features, got {df.shape[1]}'}), 400

        # Convert to numpy array and reshape
        data = df.to_numpy()
        if data.shape[0] < SEQUENCE_LENGTH:
            return jsonify({'error': f'CSV must have at least {SEQUENCE_LENGTH} rows'}), 400

        data = data[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, -1)

        # Scale input
        data_scaled = X_scaler.transform(data.reshape(-1, data.shape[2]))

        # Get prediction
        prediction = model.predict(data_scaled.reshape(1, SEQUENCE_LENGTH, -1))

        # Scale back to original values
        prediction = y_scaler.inverse_transform(prediction)
        actual_values = actual_values.reshape(1, -1)  
        actual_values = y_scaler.inverse_transform(actual_values)

        actual = actual_values.tolist()[0]
        prediction = prediction.tolist()[0]

        # Plot: Actual vs Predicted for Traffic, Congestion, Speed
        import io
        import base64
        import matplotlib.pyplot as plt

        labels = ['Traffic Volume', 'Congestion Level', 'Average Speed']

        # Split the actual and predicted data
        traffic_index = 0
        congestion_index = 1
        speed_index = 2

        # Plot 1: Traffic Volume
        plt.figure(figsize=(6, 4))
        x1 = [0]
        plt.bar(x1, [actual[traffic_index]], width=0.3, label='Actual')
        plt.bar([i + 0.3 for i in x1], [prediction[traffic_index]], width=0.3, label='Predicted')
        plt.xticks([i + 0.15 for i in x1], ['Traffic Volume'])
        plt.ylabel('Value')
        plt.title('Traffic Volume: Actual vs Predicted')
        plt.legend()
        plt.tight_layout()

        # Save to buffer
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png')
        buf1.seek(0)
        plot_traffic = base64.b64encode(buf1.read()).decode('utf-8')
        buf1.close()
        plt.close()

        # Plot 2: Congestion & Speed
        plt.figure(figsize=(6, 4))
        x2 = [0, 1]
        actual_sub = [actual[congestion_index], actual[speed_index]]
        prediction_sub = [prediction[congestion_index], prediction[speed_index]]
        plt.bar(x2, actual_sub, width=0.3, label='Actual')
        plt.bar([i + 0.3 for i in x2], prediction_sub, width=0.3, label='Predicted')
        plt.xticks([i + 0.15 for i in x2], ['Congestion Level', 'Average Speed'], rotation=15)
        plt.ylabel('Values')
        plt.title('Congestion & Speed: Actual vs Predicted')
        plt.legend()
        plt.tight_layout()

        # Save to buffer
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png')
        buf2.seek(0)
        plot_cong_speed = base64.b64encode(buf2.read()).decode('utf-8')
        buf2.close()
        plt.close()


        return jsonify({
        'actual': actual,
        'prediction': prediction,
        'plot_traffic': plot_traffic,
        'plot_cong_speed': plot_cong_speed
        })


    except Exception as e:
        print("Error:", str(e))  # Debugging
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)