import unittest
import requests
from unittest.mock import patch, MagicMock
from tools.weather import get_weather, parse_weather_response

class TestWeatherTool(unittest.TestCase):

    @patch('tools.weather.requests.get')
    @patch('tools.weather.os.environ.get')
    def test_get_weather_success(self, mock_env_get, mock_get):
        mock_env_get.return_value = "fake_api_key"
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "London",
            "weather": [{"description": "cloudy"}],
            "main": {"temp": 15, "humidity": 80},
            "wind": {"speed": 5}
        }
        mock_get.return_value = mock_response

        result = get_weather("London")
        self.assertIn("Weather in London: cloudy", result)
        self.assertIn("Temperature: 15°C", result)

    @patch('tools.weather.os.environ.get')
    def test_get_weather_no_api_key(self, mock_env_get):
        mock_env_get.return_value = None
        result = get_weather("London")
        self.assertIn("Error: OPENWEATHER_API_KEY not found", result)

    @patch('tools.weather.requests.get')
    @patch('tools.weather.os.environ.get')
    def test_get_weather_api_error(self, mock_env_get, mock_get):
        mock_env_get.return_value = "fake_api_key"
        
        mock_response = MagicMock()
        # Mock a RequestException instead of a generic Exception
        mock_response.raise_for_status.side_effect = requests.exceptions.RequestException("API Error")
        mock_get.return_value = mock_response

        result = get_weather("London")
        self.assertIn("Error fetching weather data", result)

if __name__ == '__main__':
    unittest.main()
