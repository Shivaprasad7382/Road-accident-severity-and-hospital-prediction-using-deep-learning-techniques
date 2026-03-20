🚗 Road Accident Severity and Hospital Recommendation System
📌 Overview

This project predicts the severity of road accidents using a Deep Learning (CNN) model and recommends nearby hospitals. It also supports emergency alert mechanisms for real-time response.

⚙️ Features

🔍 Accident severity prediction using CNN

🌐 Flask-based web application

🏥 Hospital recommendation system

📡 API-based architecture

📞 Emergency alert system (Twilio-ready)

🗺️ Google Maps integration (extendable)

🧠 Technologies Used

Python

Flask

OpenCV

TensorFlow / Keras

NumPy

MySQL (optional for DB)

Twilio API (for alerts)

Google Maps API (for location services)

🏗️ Project Structure
app.py
model/
    train_model.py
templates/
    index.html
utils/
    maps.py
    alerts.py
requirements.txt
README.md
▶️ How to Run the Project
1. Clone / Extract the Project
unzip advanced_road_accident_system.zip
cd advanced_road_accident_system
2. Install Dependencies
pip install -r requirements.txt
3. Run the Application
python app.py
4. Open in Browser
http://127.0.0.1:5000/
🧪 Model Training (Optional)

To train the CNN model:

cd model
python train_model.py

This will generate:

accident_model.h5
🔄 Workflow

User uploads accident image

Image processed using OpenCV

CNN model predicts severity

System returns severity level

Nearby hospitals recommended

Alert system can notify ambulance & relatives

🚀 Future Enhancements

🎥 Real-time video accident detection

📍 GPS-based live location tracking

☁️ Cloud deployment (AWS / GCP)

📊 Model accuracy visualization

📞 Automatic voice alerts

💡 Key Learning Outcomes

Backend development with Flask

Deep Learning model integration

API design and real-time processing

System architecture similar to real-world applications

👨‍💻 Author

Shivaprasad Chinthoju (Shiva)
