from fastapi import FastAPI, File, UploadFile
from realtime_pipeline_cpu import InferencePipeline
from PIL import Image
import io

app = FastAPI()
pipeline = InferencePipeline(model_path="models/int8_model.pth")

@app.get("/")
def root():
    return {"message": "Military Asset Detection API"}

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    # Read the uploaded file
    image_bytes = file.file.read()
    # Open as PIL Image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Run prediction
    result = pipeline.predict(image)
    return result 