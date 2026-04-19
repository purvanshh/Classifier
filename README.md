# BanaScope

A minimal web app that classifies South Indian banana varieties from an image using a fine-tuned EfficientNet-B0 model.

**Supported varieties:** 2390 · Grand Naine · Ney Poovan · Poovan

## Demo

Upload any banana image and get an instant prediction with confidence score.

## Stack

- **Model:** EfficientNet-B0 (PyTorch + timm), fine-tuned on a custom dataset
- **App:** Streamlit
- **Inference:** CPU / GPU (auto-detected)

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
├── app.py                  # Streamlit app
├── EfficientNet B0.pth     # Trained model weights
├── requirements.txt
└── .streamlit/
    └── config.toml         # Theme config
```

## Model details

| Property | Value |
|---|---|
| Architecture | EfficientNet-B0 |
| Input size | 224 × 224 |
| Classes | 4 |
| Framework | PyTorch + timm |
