<<<<<<< HEAD
# Real-Time Weather Prediction

This application provides real-time weather predictions using data from the OpenWeatherMap API, enhanced with advanced prediction algorithms.

## Features

- Real-time weather data from OpenWeatherMap API
- 5-day forecast with temperature, precipitation, and wind speed
- Visualization of weather factors
- Prediction reliability score
- Responsive design for mobile and desktop
- Fast response times with caching

## Requirements

- Python 3.7+
- Flask
- NumPy
- Pandas
- Matplotlib
- Requests
- OpenWeatherMap API key

## Installation

1. Clone this repository or download the source code.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Get an API key from [OpenWeatherMap](https://openweathermap.org/api) (free tier available)
4. Update the `config.py` file with your API key:

```python
OPENWEATHERMAP_API_KEY = "your_api_key_here"
```

## Running the Application

1. Run the Flask application:

```bash
python app_simplified.py
```

2. Open your web browser and navigate to:

```
http://localhost:5000
```

3. Enter a location in the search bar and click "Predict" to get a weather forecast.

## How It Works

The application fetches current weather data from OpenWeatherMap API and uses it as input for the prediction model. It then enhances the prediction with the 5-day forecast data from the API.

Key components:
- **Data Fetching**: Real-time weather data from OpenWeatherMap API
- **Caching**: Responses are cached to improve performance
- **Prediction Model**: Advanced algorithms process the weather data
- **Visualization**: Weather factors are visualized for better understanding

## Performance Optimization

- API responses are cached to reduce API calls and improve response times
- Timeout settings for API requests to ensure fast responses
- Error handling with fallback to simulated data if API is unavailable

## API Configuration

You can customize the API settings in the `config.py` file:

```python
# Units (metric, imperial, standard)
UNITS = "metric"

# Cache settings (in seconds)
CACHE_TIMEOUT = 600  # 10 minutes
```

## Note

This application uses the free tier of OpenWeatherMap API, which has a limit of 60 calls per minute or 1,000,000 calls per month. The caching mechanism helps stay within these limits. 
=======
# Real-Time-Weather-Prediction
This application provides real-time weather predictions using data from the OpenWeatherMap API, enhanced with advanced prediction algorithms.
>>>>>>> 645794b96a470d72ee72979743a4edbeb27fad5f
