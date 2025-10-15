import sys
import os
import certifi
from dotenv import load_dotenv
import pymongo
import pandas as pd

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from uvicorn import run as app_run

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.constant.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME
)

# -----------------------
# ✅ Environment setup
# -----------------------
ca = certifi.where()
load_dotenv()

mongo_db_url = os.getenv("MONGODB_URL_KEY")
print(f"MongoDB URL: {mongo_db_url}")

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# -----------------------
# ✅ FastAPI App Setup
# -----------------------
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Absolute path for templates (Fixes your error)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# -----------------------
# ✅ Routes
# -----------------------
@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("✅ Training completed successfully.")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        print(df.iloc[0])
        y_pred = network_model.predict(df)
        df["predicted_column"] = y_pred

        output_dir = os.path.join(BASE_DIR, "prediction_output")
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "output.csv"), index=False)

        table_html = df.to_html(classes="table table-striped", index=False)
        return templates.TemplateResponse(
            "table.html", {"request": request, "table": table_html}
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys)


# -----------------------
# ✅ Run App
# -----------------------
if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
