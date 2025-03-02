import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from geopy.geocoders import Nominatim
from geopy.distance import geodesic


# Load dataset with diverse accident scenarios
data = {
    "Weather": np.random.choice(["Clear", "Rain", "Fog", "Snow", "Storm"], 500),
    "Light_Conditions": np.random.choice(["Daylight", "Night", "Dusk", "Dawn"], 500),
    "Vehicle_Type": np.random.choice(["Car", "Bike", "Truck", "Bus", "Auto"], 500),
    "Speed": np.random.randint(20, 120, 500),
    "Alcohol_Involved": np.random.choice([0, 1], 500),
    "Distraction": np.random.choice([0, 1], 500),
    "Road_Surface": np.random.choice(["Dry", "Wet", "Icy", "Snow-covered"], 500),
    "Latitude": np.random.uniform(17.30, 17.50, 500),
    "Longitude": np.random.uniform(78.40, 78.60, 500),
    "Severity": np.random.choice(["Major", "Minor"], 500)
}
df = pd.DataFrame(data)

# Encode categorical columns
le = LabelEncoder()
categorical_columns = ['Weather', 'Light_Conditions', 'Vehicle_Type', 'Road_Surface']
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

df['Severity'] = df['Severity'].apply(lambda x: 1 if x.lower() == 'major' else 0)  # Binary classification: Major (1) or Minor (0)
X = df.drop(columns=['Severity', 'Latitude', 'Longitude'])  # Features
y = df['Severity']  # Target

# Data Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build Deep Learning Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Function to Predict Accident Severity
def predict_severity():
    user_input = []
    for col in X.columns:
        value = float(input(f"Enter {col}: "))
        user_input.append(value)
    data_scaled = scaler.transform([user_input])
    prediction = model.predict(data_scaled)
    severity_label = 'Major' if prediction[0][0] > 0.5 else 'Minor'
    return severity_label

# Function to Recommend Nearest Hospital
def recommend_hospital():
    latitude = float(input("Enter accident latitude: "))
    longitude = float(input("Enter accident longitude: "))
    
    hospitals = [
        {"name": "Aadhaar Hospitals", "lat": 17.3850, "lon": 78.4867},
        {"name": "New Shadow Hospitals", "lat": 17.3950, "lon": 78.4967},
        {"name": "Manogna Hospital", "lat": 17.3750, "lon": 78.4767},
        {"name": "Afsar Poly Clinic", "lat": 17.3650, "lon": 78.4667},
        {"name": "Abdullapurmet Hospital", "lat": 17.3500, "lon": 78.4500},
        {"name": "Sunrise Multispeciality", "lat": 17.3600, "lon": 78.4800},
        {"name": "Medicare Hospital", "lat": 17.3700, "lon": 78.4900},
        {"name": "Apollo Hospitals", "lat": 17.3800, "lon": 78.5000},
        {"name": "Global Hospital", "lat": 17.3900, "lon": 78.5100},
        {"name": "KIMS Hospital", "lat": 17.4000, "lon": 78.5200}
    ]
    
    for i in range(30):
        hospitals.append({"name": f"Hospital {i+11}", "lat": 17.30 + (i * 0.005), "lon": 78.40 + (i * 0.005)})
    
    min_distance = float("inf")
    nearest_hospital = None
    for hospital in hospitals:
        distance = geodesic((latitude, longitude), (hospital['lat'], hospital['lon'])).km
        if distance < min_distance:
            min_distance = distance
            nearest_hospital = hospital
    
    return f"Recommended Hospital: {nearest_hospital['name']} ({min_distance:.2f} km away)"

# Running the system
if __name__ == "__main__":
    severity = predict_severity()
    print(f"Predicted Severity: {severity}")
    hospital = recommend_hospital()
    print(hospital)
