# 🚦 Short-Term LSTM-Based Traffic Prediction with Flask

This project predicts short-term traffic parameters using an LSTM (Long Short-Term Memory) deep learning model. It features a user-friendly Flask web interface where users can upload their CSV data and get traffic predictions instantly.

---

## 📌 Features

- Upload CSV traffic data via drag-and-drop or file input
- Predicts:
  - **Traffic Volume**
  - **Congestion Level**
  - **Average Speed**
- LSTM model trained on real-world traffic data
- Flask-based web UI with real-time results
- Uses `StandardScaler` for input/output scaling

---

## 🗂️ Folder Structure
project-root/<br>
├── app.py                        # Flask app<br>
├── requirements.txt              # Python dependencies<br>
├── README.md                     # This file<br>

├── model/<br>
│   └── lstm_model_optimized.h5   # Trained LSTM model<br>

├── scalar/<br>
│   ├── X_scaler.npy              # Feature scaler<br>
│   └── y_scaler.npy              # Target scaler<br>

├── static/<br>
│   ├── style.css                 # CSS styles<br>
│   └── script.js                 # JavaScript logic<br>

├── templates/<br>
│   └── index.html                # Frontend HTML<br>

├── dataset/                      # (Optional) Sample datasets<br>

├── src/                          # Source code for model handling<br>
│   ├── preprocess.py             # Data preprocessing functions<br>
│   ├── train_model.py            # LSTM model training script<br>
│   └── evaluate_model.py         # Model evaluation and metrics<br>

---

## 🚀 Getting Started<br>

### 1. Clone the Repository<br>

git clone https://github.com/your-username/traffic-prediction-lstm.git<br>
cd traffic-prediction-lstm<br>


### 2. Create a Virtual Environment (Optional but Recommended)<br>

python -m venv venv<br>
venv\Scripts\activate       # On Windows<br>
# or<br>
source venv/bin/activate    # On macOS/Linux<br>

### 3. Install Dependencies<br>

pip install -r requirements.txt<br>

### 4. Run the Flask App<br>

python app.py
<br>

🧠 How to Use<br>

1.Upload a .csv file with your traffic data (at least 10 rows).<br>

2.The file must have the same columns and order as used during model training.<br>

3.The app will return the predicted traffic volume, congestion level, and average speed.<br>

⚠️ Important: Make sure the CSV column names and order match the trained dataset exactly. Preprocessing must follow the format used during model training.<br>

📊 Dataset Credit
<br>
The dataset used in this project was adapted from:<br>

📎 Source: [Bangalore City Traffic Dataset – Kaggle](https://www.kaggle.com/datasets/preethamgouda/banglore-city-traffic-dataset/data)<br>
📌 Author: Preetham Gouda<br>

Credits to the original authors for providing the dataset.<br>

✨ Acknowledgments<br>
TensorFlow for deep learning<br>

Flask for the lightweight web server<br>

📬 Contact<br>
Have feedback or questions?<br>
Feel free to reach out at svishwasundar6@gmail.com<br>

⭐ Give a Star<br>
If you found this helpful, consider starring this repo to support the project!<br>
