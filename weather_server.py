import random

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


# Pydantic models
class WeatherInfo(BaseModel):
    location: str
    temperature: str
    unit: str
    forecast: list[str]


class Location(BaseModel):
    location: str
    unit: str


# Fake data
forecasts = ["sunny", "cloudy", "rainy", "windy", "snowy", "foggy", "hazy"]
temperatures = list(range(30, 101))  # temperatures from 30 to 100


@app.post("/get_weather", response_model=WeatherInfo)
async def get_weather(location: Location):
    print(location)
    forecast = random.choices(forecasts, k=2)
    temperature = str(random.choice(temperatures))
    return {
        "location": location.location,
        "temperature": temperature,
        "unit": location.unit,
        "forecast": forecast,
    }
