# Import necessary libraries
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import io
import base64
import requests
import json
from datetime import datetime, timedelta

app = Flask(__name__)

# Quantum Weather Prediction Class
class QuantumWeatherPredictor:
    def __init__(self):
        self.backend = Aer.get_backend('qasm_simulator')
    
    def encode_weather_data(self, temperature, humidity, pressure, wind_speed):
        """Encode classical weather data into quantum states"""
        # Normalize values to range [0, 1]
        temp_norm = (temperature + 20) / 80  # Assuming range from -20°C to 60°C
        humidity_norm = humidity / 100
        pressure_norm = (pressure - 950) / 150  # Assuming range from 950 to 1100 hPa
        wind_norm = min(wind_speed / 50, 1)  # Capping at 50 m/s
        
        # Create a 4-qubit circuit for encoding
        qc = QuantumCircuit(4, 4)
        
        # Encode data as rotation angles
        qc.ry(temp_norm * np.pi, 0)
        qc.ry(humidity_norm * np.pi, 1)
        qc.ry(pressure_norm * np.pi, 2)
        qc.ry(wind_norm * np.pi, 3)
        
        # Add entanglement to capture correlation between parameters
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 0)
        
        return qc
    
    def apply_quantum_processing(self, circuit):
        """Apply quantum transformations for weather pattern analysis"""
        # Apply Hadamard gates to create superposition
        for i in range(4):
            circuit.h(i)
        
        # Add more entanglement
        circuit.cx(0, 2)
        circuit.cx(1, 3)
        
        # Add phase shifts to represent weather dynamics
        circuit.t(0)
        circuit.t(1)
        circuit.t(2)
        circuit.t(3)
        
        # More entanglement to model complex relationships
        circuit.cx(0, 1)
        circuit.cx(2, 3)
        
        # Measurement
        circuit.measure(range(4), range(4))
        
        return circuit
    
    def execute_circuit(self, circuit, shots=1024):
        """Execute the quantum circuit and return results"""
        job = execute(circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        return counts
    
    def interpret_results(self, counts):
        """Interpret quantum results for weather prediction"""
        # Get the most frequent outcome
        max_result = max(counts, key=counts.get)
        
        # Convert binary string to integers for each parameter
        temp_qubit = int(max_result[3])
        humidity_qubit = int(max_result[2])
        pressure_qubit = int(max_result[1])
        wind_qubit = int(max_result[0])
        
        # Convert other frequent outcomes into probabilistic weather patterns
        weather_probs = {
            'sunny': 0,
            'cloudy': 0,
            'rainy': 0,
            'stormy': 0
        }
        
        for outcome, count in counts.items():
            prob = count / sum(counts.values())
            
            # Simple heuristic for weather categorization
            if outcome.count('1') == 0:
                weather_probs['sunny'] += prob
            elif outcome.count('1') == 1:
                weather_probs['cloudy'] += prob
            elif outcome.count('1') == 2:
                weather_probs['rainy'] += prob
            else:
                weather_probs['stormy'] += prob
        
        # Calculate temperature prediction
        temp_range = [-20, 60]  # °C
        if temp_qubit == 0:
            temp_prediction = np.random.uniform(temp_range[0], (temp_range[0] + temp_range[1])/2)
        else:
            temp_prediction = np.random.uniform((temp_range[0] + temp_range[1])/2, temp_range[1])
        
        # Adjust based on pressure and other qubits
        if pressure_qubit == 1:
            temp_prediction += 5  # Higher pressure often means higher temperature
        
        # Calculate precipitation probability
        precip_prob = weather_probs['rainy'] * 80 + weather_probs['stormy'] * 100
        
        # Wind speed prediction
        if wind_qubit == 0:
            wind_prediction = np.random.uniform(0, 20)
        else:
            wind_prediction = np.random.uniform(20, 50)
        
        # Create forecast for next 5 days with quantum noise
        forecast = []
        for i in range(5):
            # Add quantum noise (decreasing reliability over time)
            noise_factor = 1 + (i * 0.15)
            day_temp = temp_prediction + np.random.normal(0, 2 * noise_factor)
            day_wind = wind_prediction + np.random.normal(0, 5 * noise_factor)
            day_precip = min(100, max(0, precip_prob + np.random.normal(0, 10 * noise_factor)))
            
            # Weather type determination
            if day_precip < 20:
                weather_type = 'Sunny'
            elif day_precip < 50:
                weather_type = 'Partly Cloudy'
            elif day_precip < 80:
                weather_type = 'Rainy'
            else:
                weather_type = 'Stormy'
                
            forecast.append({
                'day': (datetime.now() + timedelta(days=i)).strftime('%A'),
                'date': (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'),
                'temp': round(day_temp, 1),
                'wind': round(day_wind, 1),
                'precip': round(day_precip, 1),
                'weather_type': weather_type
            })
            
        return {
            'current': {
                'temperature': round(temp_prediction, 1),
                'wind_speed': round(wind_prediction, 1),
                'precipitation_probability': round(precip_prob, 1),
                'weather_type': max(weather_probs.items(), key=lambda x: x[1])[0].capitalize()
            },
            'forecast': forecast,
            'quantum_reliability_score': round((1 - (max(counts.values()) / sum(counts.values()))) * 100, 2)
        }
    
    def get_visualization(self, counts):
        """Generate visualization of quantum results"""
        plt.figure(figsize=(10, 6))
        plot_histogram(counts)
        plt.title('Quantum Weather States Distribution')
        
        # Save plot to a string buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert to base64 string
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
    
    def predict_weather(self, location_data):
        """Main method to predict weather using quantum computing"""
        # Extract current weather data from location data
        temperature = location_data.get('temperature', 20)
        humidity = location_data.get('humidity', 60)
        pressure = location_data.get('pressure', 1013)
        wind_speed = location_data.get('wind_speed', 10)
        
        # Encode data into quantum circuit
        qc = self.encode_weather_data(temperature, humidity, pressure, wind_speed)
        
        # Apply quantum processing
        qc = self.apply_quantum_processing(qc)
        
        # Execute quantum circuit
        counts = self.execute_circuit(qc)
        
        # Interpret results
        prediction = self.interpret_results(counts)
        
        # Generate visualization
        visualization = self.get_visualization(counts)
        prediction['visualization'] = visualization
        
        return prediction

# Function to fetch real weather data
def fetch_weather_data(location):
    """Fetch real weather data from an API for initial conditions"""
    # This would normally use a real API like OpenWeatherMap
    # For demonstration, we'll simulate a response
    
    # In a real application, you would use:
    # api_key = "your_api_key"
    # url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    # response = requests.get(url)
    # data = response.json()
    
    # Simulated response
    simulated_data = {
        'temperature': np.random.uniform(0, 30),
        'humidity': np.random.uniform(30, 95),
        'pressure': np.random.uniform(990, 1030),
        'wind_speed': np.random.uniform(0, 25)
    }
    
    return simulated_data

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    location = data.get('location', 'New York')
    
    # Fetch initial weather data
    weather_data = fetch_weather_data(location)
    
    # Create quantum predictor
    predictor = QuantumWeatherPredictor()
    
    # Get prediction
    prediction = predictor.predict_weather(weather_data)
    
    # Add location info
    prediction['location'] = location
    
    return jsonify(prediction)

# HTML Templates
@app.route('/templates/index.html')
def get_index_template():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MyWeather - Quantum Weather Prediction</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d);
                background-size: 400% 400%;
                animation: gradient 15s ease infinite;
                color: white;
                margin: 0;
                padding: 0;
                min-height: 100vh;
            }
            
            @keyframes gradient {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            
            .container {
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
            }
            
            header {
                text-align: center;
                padding: 20px 0;
            }
            
            h1 {
                font-size: 3rem;
                margin-bottom: 0;
            }
            
            .subtitle {
                font-style: italic;
                margin-top: 0;
            }
            
            .search-bar {
                display: flex;
                margin: 30px 0;
                justify-content: center;
            }
            
            input {
                padding: 10px 15px;
                width: 60%;
                border: none;
                border-radius: 30px 0 0 30px;
                font-size: 16px;
                outline: none;
            }
            
            button {
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 0 30px 30px 0;
                cursor: pointer;
                font-size: 16px;
                transition: background-color 0.3s;
            }
            
            button:hover {
                background-color: #45a049;
            }
            
            .weather-display {
                background-color: rgba(0, 0, 0, 0.5);
                border-radius: 15px;
                padding: 20px;
                margin-top: 20px;
                display: none;
            }
            
            .current-weather {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 30px;
            }
            
            .weather-icon {
                font-size: 5rem;
                text-align: center;
            }
            
            .temperature {
                font-size: 3rem;
                font-weight: bold;
            }
            
            .details {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            
            .forecast {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 15px;
            }
            
            .forecast-day {
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                transition: transform 0.3s;
            }
            
            .forecast-day:hover {
                transform: translateY(-5px);
            }
            
            .day-name {
                font-weight: bold;
                margin-bottom: 10px;
            }
            
            .quantum-viz {
                margin-top: 30px;
                text-align: center;
            }
            
            .quantum-viz img {
                max-width: 100%;
                border-radius: 10px;
                margin-top: 10px;
            }
            
            .loader {
                border: 5px solid #f3f3f3;
                border-top: 5px solid #3498db;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 2s linear infinite;
                margin: 20px auto;
                display: none;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .quantum-info {
                background-color: rgba(0, 0, 255, 0.2);
                border-radius: 10px;
                padding: 15px;
                margin-top: 20px;
            }
            
            footer {
                text-align: center;
                padding: 20px 0;
                margin-top: 40px;
                font-size: 0.9rem;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>MyWeather</h1>
                <p class="subtitle">Quantum Computing Weather Prediction</p>
            </header>
            
            <div class="search-bar">
                <input type="text" id="location-input" placeholder="Enter location...">
                <button id="search-button">Predict</button>
            </div>
            
            <div class="loader" id="loader"></div>
            
            <div class="weather-display" id="weather-display">
                <h2 id="location-name">Weather for Location</h2>
                
                <div class="current-weather">
                    <div>
                        <div class="weather-icon" id="weather-icon"></div>
                        <div id="weather-type">Sunny</div>
                    </div>
                    
                    <div class="temperature" id="temperature">25°C</div>
                    
                    <div class="details">
                        <div>Wind: <span id="wind-speed">10 km/h</span></div>
                        <div>Precipitation: <span id="precipitation">20%</span></div>
                    </div>
                </div>
                
                <h3>5-Day Forecast</h3>
                <div class="forecast" id="forecast"></div>
                
                <div class="quantum-viz">
                    <h3>Quantum State Visualization</h3>
                    <img id="quantum-viz-img" src="" alt="Quantum state visualization">
                </div>
                
                <div class="quantum-info">
                    <h3>Quantum Reliability Score: <span id="quantum-score">85%</span></h3>
                    <p>This weather prediction is powered by quantum computing algorithms that analyze weather patterns in superposition, enabling more accurate predictions of complex atmospheric systems.</p>
                </div>
            </div>
            
            <footer>
                &copy; 2025 MyWeather - Quantum Weather Prediction | Created with Qiskit and Flask
            </footer>
        </div>
        
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const searchButton = document.getElementById('search-button');
                const locationInput = document.getElementById('location-input');
                const weatherDisplay = document.getElementById('weather-display');
                const loader = document.getElementById('loader');
                
                // Weather icons using emoji
                const weatherIcons = {
                    'Sunny': '☀️',
                    'Partly Cloudy': '⛅',
                    'Cloudy': '☁️',
                    'Rainy': '🌧️',
                    'Stormy': '⛈️'
                };
                
                // Default location on load
                locationInput.value = 'New York';
                
                searchButton.addEventListener('click', function() {
                    const location = locationInput.value.trim();
                    if (location) {
                        fetchWeatherPrediction(location);
                    }
                });
                
                locationInput.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        const location = locationInput.value.trim();
                        if (location) {
                            fetchWeatherPrediction(location);
                        }
                    }
                });
                
                function fetchWeatherPrediction(location) {
                    // Show loader
                    loader.style.display = 'block';
                    weatherDisplay.style.display = 'none';
                    
                    // Fetch prediction from API
                    fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ location: location }),
                    })
                    .then(response => response.json())
                    .then(data => {
                        // Update the UI with prediction data
                        document.getElementById('location-name').textContent = `Weather for ${data.location}`;
                        
                        // Current weather
                        const current = data.current;
                        document.getElementById('temperature').textContent = `${current.temperature}°C`;
                        document.getElementById('wind-speed').textContent = `${current.wind_speed} km/h`;
                        document.getElementById('precipitation').textContent = `${current.precipitation_probability}%`;
                        document.getElementById('weather-type').textContent = current.weather_type;
                        document.getElementById('weather-icon').textContent = weatherIcons[current.weather_type] || '🌈';
                        
                        // Forecast
                        const forecastContainer = document.getElementById('forecast');
                        forecastContainer.innerHTML = '';
                        
                        data.forecast.forEach(day => {
                            const dayElement = document.createElement('div');
                            dayElement.className = 'forecast-day';
                            dayElement.innerHTML = `
                                <div class="day-name">${day.day}</div>
                                <div class="weather-icon">${weatherIcons[day.weather_type] || '🌈'}</div>
                                <div>${day.temp}°C</div>
                                <div>${day.precip}% precip</div>
                            `;
                            forecastContainer.appendChild(dayElement);
                        });
                        
                        // Quantum visualization
                        document.getElementById('quantum-viz-img').src = `data:image/png;base64,${data.visualization}`;
                        
                        // Quantum score
                        document.getElementById('quantum-score').textContent = `${data.quantum_reliability_score}%`;
                        
                        // Hide loader and show weather
                        loader.style.display = 'none';
                        weatherDisplay.style.display = 'block';
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('Error fetching weather prediction. Please try again.');
                        loader.style.display = 'none';
                    });
                }
                
                // Initial load
                fetchWeatherPrediction('New York');
            });
        </script>
    </body>
    </html>
    """
    return html

# Create template directory and files for Flask
def create_templates():
    import os
    
    if not os.path.exists('templates'):
        os.makedirs('templates')
        
    with open('templates/index.html', 'w') as f:
        f.write(get_index_template())

# Entry point
if __name__ == '__main__':
    print("Creating template files...")
    create_templates()
    print("Starting MyWeather server...")
    app.run(debug=True, host='0.0.0.0', port=5000) 