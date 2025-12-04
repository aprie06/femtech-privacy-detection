# Demo

This folder contains interactive demonstrations of the privacy policy deception detection system.

## Option 1: Google Colab (Recommended for Presentations)

**Best for:** Class presentations, live demos, no setup required

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aprie06/femtech-privacy-disaster-detection/blob/main/demo/Femtech_Privacy_Demo_1.ipynb)

**To use:**
1. Click the badge above or upload `Femtech_Privacy_Demo_1.ipynb` to Google Colab
2. Run all cells (Runtime > Run all)
3. Wait for model to load (~2-3 minutes first time)
4. Try the interactive examples

**Features:**
- Pre-loaded real violation examples (Flo, Ovia, Glow)
- Interactive custom input testing
- Batch analysis of all documented violations
- No installation required

---

## Option 2: Streamlit App (Local Web Interface)

**Best for:** Interactive exploration, custom demos

**To run locally:**

```bash
# Install Streamlit
pip install streamlit

# Run the app
streamlit run demo/streamlit_app.py
```

Then open http://localhost:8501 in your browser.

**Features:**
- Web-based interface
- Real-time contradiction detection
- Pre-loaded examples with one-click loading
- Visual severity indicators
