from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch
import timm
from torchvision import transforms
import json
import io

app = FastAPI(title="Skin Disease Detection API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 7

model = timm.create_model(
"efficientnet_b4",
pretrained=False,
num_classes=NUM_CLASSES
)

state_dict = torch.load("model/skin_model.pth", map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()


with open("data/disease_info.json") as f:
   disease_db = json.load(f)


classes = [
"Actinic Keratoses",
"Basal Cell Carcinoma",
"Benign Keratosis",
"Dermatofibroma",
"Melanoma",
"Melanocytic Nevus",
"Vascular Lesion"
]


transform = transforms.Compose([
transforms.Resize((380, 380)),
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
)
])


@app.get("/")
def health_check():
    return {"status": "Skin Disease Detection API running"}

# -------------------------

# Prediction Endpoint

# -------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess image
        img = transform(image).unsqueeze(0).to(device)

        # Model inference
        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

        # Get predicted disease
        disease = classes[pred.item()]

        # Lookup disease info
        info = disease_db.get(
            disease,
            {
                "advice": "Consult a dermatologist",
                "medicines": []
            }
        )

        # Return response
        return {
            "prediction": disease,
            "confidence": float(confidence),
            "advice": info["advice"],
            "medicines": info["medicines"],
            "disclaimer": "This AI system is not a medical diagnosis. Consult a certified dermatologist."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
