from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load model
class EnsembleModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.scaler = self.model.scaler

    def predict(self, input_df):
        scaled = self.scaler.transform(input_df)
        rf_pred = self.model.rf_model.predict(scaled)
        xgb_pred = self.model.xgb_model.predict(scaled)
        return (rf_pred + xgb_pred) / 2  # Ensemble average

ensemble_model = EnsembleModel("ensemble_model.pkl")

# Input model for prediction
class PredictionRequest(BaseModel):
    variety: str
    soil_ph: float
    nitrogen: float
    phosphorus: float
    potassium: float
    organic_matter: float
    beneficial_microbes: float
    harmful_microbes: float
    microbial_biomass: float
    soil_organic_carbon: float
    microbial_activity: str
    soil_enzyme_activity: str
    disease_present: str
    disease_name: str | None = None
    nutrient_deficiency: str
    nutrient_deficiency_name: str | None = None

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Feature engineering
        df = pd.DataFrame({
            'Variety_Mangala': [1 if request.variety == 'Mangala' else 0],
            'Variety_SK_Local': [1 if request.variety == 'SK Local' else 0],
            'Variety_Shreemangala': [1 if request.variety == 'Shreemangala' else 0],
            'Variety_Sumangala': [1 if request.variety == 'Sumangala' else 0],
            'Soil_pH': [request.soil_ph],
            'N_Nitrogen': [request.nitrogen],
            'P_Phosphorus': [request.phosphorus],
            'K_Potassium': [request.potassium],
            'Organic_Matter_kg_compost': [request.organic_matter],
            'Temperature_°C': [25.0],  # Mean value
            'Rainfall_mm': [2000.0],  # Mean value
            'Elevation_m': [300.0],  # Mean value
            'Beneficial_Microbes_CFU_g': [request.beneficial_microbes],
            'Harmful_Microbes_CFU_g': [request.harmful_microbes],
            'Microbial_Biomass_C_g_kg': [request.microbial_biomass],
            'Soil_Organic_Carbon': [request.soil_organic_carbon / 100],
            'Microbial_Activity_High': [1 if request.microbial_activity == 'High' else 0],
            'Microbial_Activity_Moderate': [1 if request.microbial_activity == 'Moderate' else 0],
            'Microbial_Activity_Low': [1 if request.microbial_activity == 'Low' else 0],
            'Soil_Enzyme_Activity_High': [1 if request.soil_enzyme_activity == 'High' else 0],
            'Soil_Enzyme_Activity_Moderate': [1 if request.soil_enzyme_activity == 'Moderate' else 0],
            'Soil_Enzyme_Activity_Low': [1 if request.soil_enzyme_activity == 'Low' else 0],
            'Disease_Yes_No_Yes': [1 if request.disease_present == 'Yes' else 0],
            'Disease_Name_Koleroga': [1 if request.disease_present == 'Yes' and request.disease_name == 'Koleroga (Mahali)' else 0],
            'Disease_Name_Spindle_Bug': [1 if request.disease_present == 'Yes' and request.disease_name == 'Spindle Bug' else 0],
            'Nutrient_Deficiency_Nitrogen': [1 if request.nutrient_deficiency == 'Yes' and request.nutrient_deficiency_name == 'Nitrogen Deficiency' else 0],
            'Nutrient_Deficiency_Phosphorus': [1 if request.nutrient_deficiency == 'Yes' and request.nutrient_deficiency_name == 'Phosphorus Deficiency' else 0],
            'Nutrient_Deficiency_Potassium': [1 if request.nutrient_deficiency == 'Yes' and request.nutrient_deficiency_name == 'Potassium Deficiency' else 0],
        })

        df = df.reindex(columns=ensemble_model.model.scaler.feature_names_in_, fill_value=0)
        yield_value = ensemble_model.predict(df)

        return {"predicted_yield": round(yield_value[0], 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llm-insights")
async def generate_insight(request: PredictionRequest):
    try:
        # Get prediction
        df = pd.DataFrame({
            'Variety_Mangala': [1 if request.variety == 'Mangala' else 0],
            'Variety_SK_Local': [1 if request.variety == 'SK Local' else 0],
            'Variety_Shreemangala': [1 if request.variety == 'Shreemangala' else 0],
            'Variety_Sumangala': [1 if request.variety == 'Sumangala' else 0],
            'Soil_pH': [request.soil_ph],
            'N_Nitrogen': [request.nitrogen],
            'P_Phosphorus': [request.phosphorus],
            'K_Potassium': [request.potassium],
            'Organic_Matter_kg_compost': [request.organic_matter],
            'Temperature_°C': [25.0],  # Mean value
            'Rainfall_mm': [2000.0],  # Mean value
            'Elevation_m': [300.0],  # Mean value
            'Beneficial_Microbes_CFU_g': [request.beneficial_microbes],
            'Harmful_Microbes_CFU_g': [request.harmful_microbes],
            'Microbial_Biomass_C_g_kg': [request.microbial_biomass],
            'Soil_Organic_Carbon': [request.soil_organic_carbon / 100],
            'Microbial_Activity_High': [1 if request.microbial_activity == 'High' else 0],
            'Microbial_Activity_Moderate': [1 if request.microbial_activity == 'Moderate' else 0],
            'Microbial_Activity_Low': [1 if request.microbial_activity == 'Low' else 0],
            'Soil_Enzyme_Activity_High': [1 if request.soil_enzyme_activity == 'High' else 0],
            'Soil_Enzyme_Activity_Moderate': [1 if request.soil_enzyme_activity == 'Moderate' else 0],
            'Soil_Enzyme_Activity_Low': [1 if request.soil_enzyme_activity == 'Low' else 0],
            'Disease_Yes_No_Yes': [1 if request.disease_present == 'Yes' else 0],
            'Disease_Name_Koleroga': [1 if request.disease_present == 'Yes' and request.disease_name == 'Koleroga (Mahali)' else 0],
            'Disease_Name_Spindle_Bug': [1 if request.disease_present == 'Yes' and request.disease_name == 'Spindle Bug' else 0],
            'Nutrient_Deficiency_Nitrogen': [1 if request.nutrient_deficiency == 'Yes' and request.nutrient_deficiency_name == 'Nitrogen Deficiency' else 0],
            'Nutrient_Deficiency_Phosphorus': [1 if request.nutrient_deficiency == 'Yes' and request.nutrient_deficiency_name == 'Phosphorus Deficiency' else 0],
            'Nutrient_Deficiency_Potassium': [1 if request.nutrient_deficiency == 'Yes' and request.nutrient_deficiency_name == 'Potassium Deficiency' else 0],
        })
        df = df.reindex(columns=ensemble_model.model.scaler.feature_names_in_, fill_value=0)
        yield_value = ensemble_model.predict(df)

        # Compose automatic prompt
        prompt = f"""
        The arecanut crop input parameters are:
        Variety: {request.variety}, Soil pH: {request.soil_ph}, N: {request.nitrogen}, P: {request.phosphorus}, K: {request.potassium},
        Organic Matter: {request.organic_matter}, Beneficial Microbes: {request.beneficial_microbes}, Harmful Microbes: {request.harmful_microbes},
        Microbial Biomass: {request.microbial_biomass}, SOC: {request.soil_organic_carbon}, Microbial Activity: {request.microbial_activity},
        Soil Enzyme Activity: {request.soil_enzyme_activity}, Disease: {request.disease_present}, Nutrient Deficiency: {request.nutrient_deficiency}

        Given these inputs, and a predicted arecanut yield of {round(yield_value[0], 2)} kg/palm,
        provide agronomic and biological interpretation, potential risks, and advice to improve yield.
        """

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=2048,
            top_p=1,
            stream=False
        )

        return {"llm_insight": completion.choices[0].message.content.strip(), "predicted_yield": round(yield_value[0], 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Arecanut Yield Prediction API"}