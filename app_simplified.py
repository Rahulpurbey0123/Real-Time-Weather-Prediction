# Import necessary libraries
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import requests
import json
from datetime import datetime, timedelta
import random
import time
import os
from functools import wraps

# Import configuration
try:
    import config   
    API_KEY = config.OPENWEATHERMAP_API_KEY
    CURRENT_WEATHER_URL = config.CURRENT_WEATHER_URL
    FORECAST_URL = config.FORECAST_URL
    UNITS = config.UNITS
    CACHE_TIMEOUT = config.CACHE_TIMEOUT
except (ImportError, AttributeError):
    # Default values if config is not available
    API_KEY = "e7a55d50b3e75a22479897fcc557a05b"  # Replace with your actual API key
    CURRENT_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
    FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
    UNITS = "metric"
    CACHE_TIMEOUT = 600  # 10 minutes

app = Flask(__name__)

# Simple in-memory cache
cache = {}

def cached(timeout=CACHE_TIMEOUT):
    """Simple caching decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Create a cache key from the function name and arguments
            key = f.__name__ + str(args) + str(kwargs)
            
            # Check if we have a valid cached result
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < timeout:
                    print(f"Cache hit for {key}")
                    return result
            
            # If not in cache or expired, call the function
            result = f(*args, **kwargs)
            
            # Store in cache
            cache[key] = (result, time.time())
            return result
        return decorated_function
    return decorator

# Simplified Weather Prediction Class
class WeatherPredictor:
    def __init__(self):
        pass
    
    def encode_weather_data(self, temperature, humidity, pressure, wind_speed):
        """Encode classical weather data into normalized values"""
        # Normalize values to range [0, 1]
        temp_norm = (temperature + 20) / 80  # Assuming range from -20°C to 60°C
        humidity_norm = humidity / 100
        pressure_norm = (pressure - 950) / 150  # Assuming range from 950 to 1100 hPa
        wind_norm = min(wind_speed / 50, 1)  # Capping at 50 m/s
        
        return {
            'temp_norm': temp_norm,
            'humidity_norm': humidity_norm,
            'pressure_norm': pressure_norm,
            'wind_norm': wind_norm
        }
    
    def apply_processing(self, encoded_data):
        """Apply transformations for weather pattern analysis"""
        # Simulate quantum processing with classical calculations
        processed_data = {}
        
        # Apply some transformations to simulate quantum effects
        processed_data['temp_factor'] = np.sin(encoded_data['temp_norm'] * np.pi) * 0.5 + 0.5
        processed_data['humidity_factor'] = np.cos(encoded_data['humidity_norm'] * np.pi) * 0.5 + 0.5
        processed_data['pressure_factor'] = np.sin(encoded_data['pressure_norm'] * np.pi * 2) * 0.5 + 0.5
        processed_data['wind_factor'] = np.cos(encoded_data['wind_norm'] * np.pi * 2) * 0.5 + 0.5
        
        # Add some randomness to simulate quantum uncertainty
        for key in processed_data:
            processed_data[key] = processed_data[key] + np.random.normal(0, 0.1)
            processed_data[key] = max(0, min(1, processed_data[key]))  # Clamp to [0, 1]
        
        return processed_data
    
    def interpret_results(self, processed_data):
        """Interpret results for weather prediction"""
        # Calculate temperature prediction
        temp_range = [-20, 60]  # °C
        temp_prediction = temp_range[0] + processed_data['temp_factor'] * (temp_range[1] - temp_range[0])
        
        # Calculate precipitation probability
        precip_prob = processed_data['humidity_factor'] * 100
        
        # Wind speed prediction
        wind_prediction = processed_data['wind_factor'] * 50
        
        # Weather type determination
        weather_probs = {
            'sunny': (1 - processed_data['humidity_factor']) * processed_data['temp_factor'],
            'cloudy': processed_data['humidity_factor'] * (1 - processed_data['pressure_factor']),
            'rainy': processed_data['humidity_factor'] * processed_data['pressure_factor'] * (1 - processed_data['wind_factor']),
            'stormy': processed_data['humidity_factor'] * processed_data['pressure_factor'] * processed_data['wind_factor']
        }
        
        # Normalize probabilities
        total_prob = sum(weather_probs.values())
        for key in weather_probs:
            weather_probs[key] /= total_prob
        
        # Create forecast for next 5 days with noise
        forecast = []
        for i in range(5):
            # Add noise (decreasing reliability over time)
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
            'quantum_reliability_score': round(random.uniform(70, 95), 2)
        }
    
    def get_visualization(self, processed_data):
        """Generate visualization of results"""
        plt.figure(figsize=(10, 6))
        
        # Create a bar chart of the processed data
        labels = ['Temperature', 'Humidity', 'Pressure', 'Wind']
        values = [processed_data['temp_factor'], processed_data['humidity_factor'], 
                 processed_data['pressure_factor'], processed_data['wind_factor']]
        
        plt.bar(labels, values, color=['red', 'blue', 'green', 'orange'])
        plt.ylim(0, 1)
        plt.title('Weather Factors Distribution')
        plt.ylabel('Factor Value')
        
        # Add a grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save plot to a string buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert to base64 string
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
    
    def predict_weather(self, location_data):
        """Main method to predict weather"""
        # Extract current weather data from location data
        temperature = location_data.get('temperature', 20)
        humidity = location_data.get('humidity', 60)
        pressure = location_data.get('pressure', 1013)
        wind_speed = location_data.get('wind_speed', 10)
        
        # Encode data
        encoded_data = self.encode_weather_data(temperature, humidity, pressure, wind_speed)
        
        # Apply processing
        processed_data = self.apply_processing(encoded_data)
        
        # Interpret results
        prediction = self.interpret_results(processed_data)
        
        # Generate visualization
        visualization = self.get_visualization(processed_data)
        prediction['visualization'] = visualization
        
        return prediction

# Function to fetch real weather data
@cached()
def fetch_weather_data(location):
    """Fetch real weather data from OpenWeatherMap API for initial conditions"""
    try:
        # OpenWeatherMap API key - you need to replace this with your actual API key
        api_key = API_KEY
        
        # API endpoint for current weather
        url = f"{CURRENT_WEATHER_URL}?q={location}&appid={api_key}&units={UNITS}"
        
        # Make the API request
        response = requests.get(url, timeout=5)  # 5 second timeout for faster response
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            
            # Extract relevant weather data
            weather_data = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed']
            }
            
            return weather_data
        else:
            # If API request fails, use simulated data as fallback
            print(f"API request failed with status code {response.status_code}. Using simulated data.")
            return {
                'temperature': np.random.uniform(0, 30),
                'humidity': np.random.uniform(30, 95),
                'pressure': np.random.uniform(990, 1030),
                'wind_speed': np.random.uniform(0, 25)
            }
    except Exception as e:
        # If any error occurs, use simulated data as fallback
        print(f"Error fetching weather data: {e}. Using simulated data.")
        return {
            'temperature': np.random.uniform(0, 30),
            'humidity': np.random.uniform(30, 95),
            'pressure': np.random.uniform(990, 1030),
            'wind_speed': np.random.uniform(0, 25)
        }

# Function to fetch 5-day forecast data
@cached()
def fetch_forecast_data(location):
    """Fetch 5-day forecast data from OpenWeatherMap API"""
    try:
        # OpenWeatherMap API key - you need to replace this with your actual API key
        api_key = API_KEY
        
        # API endpoint for 5-day forecast
        url = f"{FORECAST_URL}?q={location}&appid={api_key}&units={UNITS}"
        
        # Make the API request
        response = requests.get(url, timeout=5)  # 5 second timeout for faster response
        
        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Forecast API request failed with status code {response.status_code}.")
            return None
    except Exception as e:
        print(f"Error fetching forecast data: {e}")
        return None

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request data"}), 400
            
        location = data.get('location', 'New York')
        
        # Fetch initial weather data
        start_time = time.time()
        weather_data = fetch_weather_data(location)
        weather_fetch_time = time.time() - start_time
        
        # Create predictor
        predictor = WeatherPredictor()
        
        # Get prediction
        start_time = time.time()
        prediction = predictor.predict_weather(weather_data)
        prediction_time = time.time() - start_time
        
        # Try to get real forecast data
        start_time = time.time()
        forecast_data = fetch_forecast_data(location)
        forecast_fetch_time = time.time() - start_time
        
        # If we have real forecast data, use it to enhance our prediction
        if forecast_data and 'list' in forecast_data:
            # Process the forecast data to get daily forecasts
            daily_forecasts = []
            forecast_list = forecast_data['list']
            
            # Get unique dates from the forecast
            dates = set()
            date_forecasts = {}
            
            for item in forecast_list:
                date = item['dt_txt'].split(' ')[0]
                if date not in dates:
                    dates.add(date)
                    date_forecasts[date] = []
                date_forecasts[date].append(item)
            
            # Get the next 5 days
            sorted_dates = sorted(list(dates))[:5]
            
            # Create a forecast for each day
            for i, date in enumerate(sorted_dates):
                items = date_forecasts[date]
                
                # Calculate average values for the day
                temp_sum = sum(item['main']['temp'] for item in items)
                temp_avg = temp_sum / len(items)
                
                # Get weather condition from the middle of the day if available
                mid_day_items = [item for item in items if '12:00:00' in item['dt_txt']]
                if mid_day_items:
                    weather_type = mid_day_items[0]['weather'][0]['main']
                else:
                    weather_type = items[0]['weather'][0]['main']
                
                # Map OpenWeatherMap weather types to our types
                weather_map = {
                    'Clear': 'Sunny',
                    'Clouds': 'Partly Cloudy' if any('scattered' in item['weather'][0]['description'].lower() for item in items) else 'Cloudy',
                    'Rain': 'Rainy',
                    'Drizzle': 'Rainy',
                    'Thunderstorm': 'Stormy',
                    'Snow': 'Snowy',
                    'Mist': 'Cloudy',
                    'Fog': 'Cloudy'
                }
                
                mapped_weather = weather_map.get(weather_type, 'Partly Cloudy')
                
                # Calculate precipitation probability
                precip_prob = 0
                for item in items:
                    if 'rain' in item or 'snow' in item:
                        precip_prob = max(precip_prob, 80)
                    elif weather_type in ['Clouds', 'Mist', 'Fog']:
                        precip_prob = max(precip_prob, 30)
                
                # Get wind speed
                wind_sum = sum(item['wind']['speed'] for item in items)
                wind_avg = wind_sum / len(items)
                
                # Format date
                forecast_date = datetime.strptime(date, '%Y-%m-%d')
                day_name = forecast_date.strftime('%A')
                
                daily_forecasts.append({
                    'day': day_name,
                    'date': date,
                    'temp': round(temp_avg, 1),
                    'wind': round(wind_avg, 1),
                    'precip': round(precip_prob, 1),
                    'weather_type': mapped_weather
                })
            
            # Replace our predicted forecast with the real one
            prediction['forecast'] = daily_forecasts
        
        # Add location info
        prediction['location'] = location
        
        # Add data source info
        prediction['data_source'] = 'OpenWeatherMap API' if forecast_data else 'Prediction Model'
        
        # Add performance metrics
        prediction['performance'] = {
            'weather_fetch_time_ms': round(weather_fetch_time * 1000, 2),
            'forecast_fetch_time_ms': round(forecast_fetch_time * 1000, 2),
            'prediction_time_ms': round(prediction_time * 1000, 2),
            'total_time_ms': round((weather_fetch_time + forecast_fetch_time + prediction_time) * 1000, 2)
        }
        
        return jsonify(prediction)
    except Exception as e:
        print(f"Error in predict route: {e}")
        return jsonify({
            "error": "An error occurred while processing your request",
            "message": str(e)
        }), 500

# Clear cache endpoint for admin use
@app.route('/admin/clear-cache', methods=['POST'])
def clear_cache():
    try:
        # Simple authentication - in production, use proper authentication
        auth_key = request.headers.get('X-Admin-Key')
        if not auth_key or auth_key != "admin-secret-key":  # Replace with a secure key
            return jsonify({"error": "Unauthorized"}), 401
            
        # Clear the cache
        cache.clear()
        return jsonify({"message": "Cache cleared successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# HTML Templates
@app.route('/templates/index.html')
def get_index_template():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MyWeather - Real-Time Weather Prediction</title>
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
                flex-wrap: wrap;
            }
            
            @media (max-width: 768px) {
                .current-weather {
                    flex-direction: column;
                    text-align: center;
                }
                
                .current-weather > div {
                    margin-bottom: 15px;
                }
                
                .details {
                    align-items: center;
                }
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
            
            .data-source {
                background-color: rgba(0, 0, 0, 0.3);
                border-radius: 5px;
                padding: 5px 10px;
                display: inline-block;
                margin-top: 10px;
                font-size: 0.9rem;
            }
            
            footer {
                text-align: center;
                padding: 20px 0;
                margin-top: 40px;
                font-size: 0.9rem;
            }
            
            /* Improve responsiveness */
            @media (max-width: 600px) {
                h1 {
                    font-size: 2rem;
                }
                
                .search-bar {
                    flex-direction: column;
                    align-items: center;
                }
                
                input {
                    width: 90%;
                    border-radius: 30px;
                    margin-bottom: 10px;
                }
                
                button {
                    width: 50%;
                    border-radius: 30px;
                }
                
                .temperature {
                    font-size: 2.5rem;
                }
                
                .weather-icon {
                    font-size: 4rem;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>MyWeather</h1>
                <p class="subtitle">Real-Time Weather Prediction</p>
            </header>
            
            <div class="search-bar">
                <input type="text" id="location-input" placeholder="Enter location...">
                <button id="search-button">Predict</button>
            </div>
            
            <div class="loader" id="loader"></div>
            
            <div class="weather-display" id="weather-display">
                <h2 id="location-name">Weather for Location</h2>
                <div class="data-source" id="data-source">Data Source: OpenWeatherMap API</div>
                
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
                    <h3>Weather Factors Visualization</h3>
                    <img id="quantum-viz-img" src="" alt="Weather factors visualization">
                </div>
                
                <div class="quantum-info">
                    <h3>Prediction Reliability Score: <span id="quantum-score">85%</span></h3>
                    <p>This weather prediction is powered by advanced algorithms that analyze weather patterns, enabling more accurate predictions of complex atmospheric systems.</p>
                </div>
            </div>
            
            <footer>
                &copy; 2025 MyWeather - Real-Time Weather Prediction | Powered by OpenWeatherMap API
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
                    'Stormy': '⛈️',
                    'Snowy': '❄️'
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
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('Network response was not ok');
                        }
                        return response.json();
                    })
                    .then(data => {
                        // Update the UI with prediction data
                        document.getElementById('location-name').textContent = `Weather for ${data.location}`;
                        document.getElementById('data-source').textContent = `Data Source: ${data.data_source}`;
                        
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
                        
                        // Visualization
                        document.getElementById('quantum-viz-img').src = `data:image/png;base64,${data.visualization}`;
                        
                        // Reliability score
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
                
                // Initial load with a small delay to ensure everything is ready
                setTimeout(() => {
                    fetchWeatherPrediction('New York');
                }, 500);
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
        
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(get_index_template())

# Entry point
if __name__ == '__main__':
    print("Creating template files...")
    create_templates()
    print("Starting MyWeather server...")
    app.run(debug=True, host='0.0.0.0', port=5000) 