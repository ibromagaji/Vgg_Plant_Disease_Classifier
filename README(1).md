# 🌿 LeafScan — Plant Disease Classifier

A classy, full-screen Streamlit app for classifying plant diseases from leaf images.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Wiring Your Model

Open `app.py` and replace the `run_model()` function stub with your real inference code.

The function receives `image_bytes: bytes` (raw image data) and must return a dict:

```python
def run_model(image_bytes: bytes) -> dict:
    # Load your model
    # img = Image.open(io.BytesIO(image_bytes))
    # pred = your_model.predict(img)

    return {
        "disease":     "Late Blight",        # str — display name
        "confidence":  94.2,                 # float — 0 to 100
        "severity":    "High",               # "Low" | "Moderate" | "High"
        "description": "...",                # str — paragraph about the disease
        "symptoms":    ["symptom 1", ...],   # list[str]
        "treatment":   ["step 1", ...],      # list[str]
        "prevention":  ["tip 1", ...],       # list[str]
    }
```

## Pages

| Page | Description |
|------|-------------|
| **Upload** | Hero header, drag-and-drop image upload, Analyse button |
| **Result** | Full diagnostic report — disease name, confidence bar, severity badge, description, symptoms, treatment, prevention panels |

The back button on the result page resets state and returns to the upload screen.
