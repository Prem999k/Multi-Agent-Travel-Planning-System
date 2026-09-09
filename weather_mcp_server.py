# Install dependencies:
# pip install mcp requests python-dotenv

import os

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables from .env
load_dotenv()

# Create the MCP weather server.
mcp = FastMCP("Weather Server")

# Read the OpenWeather API key from .env.
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


# Get the current weather for a city.
@mcp.tool()
def get_current_weather(city: str):
    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "q": city,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
        },
        timeout=15,
    )

    data = response.json()

    # Return the API error when the request fails.
    if response.status_code != 200:
        return {
            "error": data.get("message", "Unable to fetch weather"),
            "status_code": response.status_code,
        }

    return {
        "city": data["name"],
        "temperature_c": data["main"]["temp"],
        "feels_like_c": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"],
    }


# Get a short weather forecast for a city.
@mcp.tool()
def get_forecast(city: str):
    response = requests.get(
        "https://api.openweathermap.org/data/2.5/forecast",
        params={
            "q": city,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
        },
        timeout=15,
    )

    data = response.json()

    # Return the API error when the request fails.
    if response.status_code != 200:
        return {
            "error": data.get("message", "Unable to fetch forecast"),
            "status_code": response.status_code,
        }

    # Return the first 5 forecast entries.
    forecast = []

    for item in data.get("list", [])[:5]:
        forecast.append(
            {
                "datetime": item["dt_txt"],
                "temperature": item["main"]["temp"],
                "weather": item["weather"][0]["description"],
            }
        )

    return {
        "city": data.get("city", {}).get("name", city),
        "forecast": forecast,
    }


# Start the MCP server.
if __name__ == "__main__":
    mcp.run()