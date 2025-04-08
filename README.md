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

project-root/ ├── model/ │ └── lstm_model_optimized.h5 # Trained LSTM model ├── scalar/ │ ├── X_scaler.npy # Feature scaler │ └── y_scaler.npy # Target scaler ├── static/ │ ├── style.css # CSS styles │ └── script.js # JavaScript logic ├── templates/ │ └── index.html # Frontend HTML ├── dataset/ # (Optional) Folder to include sample datasets ├── app.py # Flask app ├── requirements.txt # Python dependencies └── README.md # This file



---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/traffic-prediction-lstm.git
cd traffic-prediction-lstm


2. Create a Virtual Environment (Optional but Recommended)

python -m venv venv
venv\Scripts\activate       # On Windows
# or
source venv/bin/activate    # On macOS/Linux

3. Install Dependencies

pip install -r requirements.txt

4. Run the Flask App

python app.py

🧠 How to Use

1.Upload a .csv file with your traffic data (at least 10 rows).

2.The file must have the same columns and order as used during model training.

3.The app will return the predicted traffic volume, congestion level, and average speed.

⚠️ Important: Make sure the CSV column names and order match the trained dataset exactly. Preprocessing must follow the format used during model training.

📊 Dataset Credit

The dataset used in this project was adapted from:

📎 Source: [Bangalore City Traffic Dataset – Kaggle](https://www.kaggle.com/datasets/preethamgouda/banglore-city-traffic-dataset/data)
📌 Author: Preetham Gouda

Credits to the original authors for providing the dataset.

✨ Acknowledgments
TensorFlow for deep learning

Flask for the lightweight web server

📬 Contact
Have feedback or questions?
Feel free to reach out at svishwasundar6@gmail.com

⭐ Give a Star
If you found this helpful, consider starring this repo to support the project!