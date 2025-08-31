import pandas as pd
import numpy as np
from faker import Faker
import random
from pathlib import Path

# Paths
DATA_SYNTHETIC = Path("/mnt/data/agriculture_suitability.csv")

# Faker setup
fake = Faker()

# Config
n_rows = 200
crop_types = ["Wheat", "Rice", "Maize", "Cotton", "Sugarcane"]
soil_types = ["Loamy", "Sandy", "Clay", "Alluvial", "Black"]
irrigation_types = ["Canal", "Tube-well", "Rain-fed", "Drip"]
seasons = ["Kharif", "Rabi", "Zaid"]

# Generate synthetic dataset
data = []
for i in range(1, n_rows + 1):
    crop = random.choice(crop_types)
    soil = random.choice(soil_types)
    irrigation = random.choice(irrigation_types)
    season = random.choice(seasons)
    
    farm_area = round(random.uniform(1, 50), 2)
    fertilizer = round(random.uniform(0.1, 5.0), 2)
    pesticide = round(random.uniform(0.5, 20.0), 2)
    water_usage = round(random.uniform(100, 10000), 2)
    
    # yield depends loosely on crop type and soil fertility
    base_yield = {
        "Wheat": 3.0, "Rice": 4.5, "Maize": 3.8, "Cotton": 2.5, "Sugarcane": 6.0
    }[crop]
    yield_tons = round(base_yield * farm_area * random.uniform(0.6, 1.4), 2)
    
    # Suitability rule (simplified heuristic)
    suitable = (
        (soil in ["Loamy", "Alluvial", "Black"]) and 
        (water_usage > 500) and 
        (fertilizer >= 0.5) and 
        (yield_tons / farm_area > 2.0)
    )
    suitability = "Suitable" if suitable else "Not Suitable"
    
    data.append([
        f"FARM_{i:04d}", crop, farm_area, irrigation, fertilizer, pesticide,
        yield_tons, soil, season, water_usage, suitability
    ])

columns = [
    "Farm_ID", "Crop_Type", "Farm_Area(acres)", "Irrigation_Type", 
    "Fertilizer_Used(tons)", "Pesticide_Used(kg)", "Yield(tons)",
    "Soil_Type", "Season", "Water_Usage(cubic meters)", "Suitability"
]

df = pd.DataFrame(data, columns=columns)

# Save synthetic dataset
df.to_csv(DATA_SYNTHETIC, index=False)

from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("Synthetic Agriculture Suitability Dataset", df.head(10))
