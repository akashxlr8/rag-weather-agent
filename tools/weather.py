import os
import requests
from typing import Dict, Any

def get_weather(city: str) -> str:
    """
    Fetches the current weather for a given city using the OpenWeatherMap API.
    """
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: OPENWEATHER_API_KEY not found in environment variables."

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return parse_weather_response(data)
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {e}"

def parse_weather_response(data: Dict[str, Any]) -> str:
    """
    Parses the OpenWeatherMap API response into a human-readable string.
    """
    try:
        city_name = data.get("name", "Unknown City")
        weather_desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]

        return (
            f"Weather in {city_name}: {weather_desc}. "
            f"Temperature: {temp}°C. "
            f"Humidity: {humidity}%. "
            f"Wind Speed: {wind_speed} m/s."
        )
    except (KeyError, IndexError) as e:
        return f"Error parsing weather data: {e}"
