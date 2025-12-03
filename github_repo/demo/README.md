# Demo

This folder contains interactive demonstrations of the privacy policy deception detection system.

## Option 1: Google Colab (Recommended for Presentations)

**Best for:** Class presentations, live demos, no setup required

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aprie06/femtech-privacy-disaster-detection/blob/main/demo/Femtech_Privacy_Demo.ipynb)

**To use:**
1. Click the badge above or upload `Femtech_Privacy_Demo.ipynb` to Google Colab
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

---

## Demo Script for Presentations

**Setup (before presentation):**
1. Open Colab notebook
2. Run cells 1-3 (install, load model, define detector)
3. Keep tab open - model stays loaded

**During presentation:**
1. Show Slide explaining NLI approach
2. Switch to Colab
3. Run Example 1 (Flo) - show CRITICAL result
4. Run Example 2 (Ovia) - show CRITICAL result  
5. Run Example 4 (Apple) - show LOW result (control works!)
6. Optional: Run custom example from audience
7. Run batch analysis showing all results
8. Return to slides

**Talking points:**
- "The model correctly identifies Flo's deceptive claim with 0.89 contradiction score"
- "Apple Health serves as our control - the model correctly shows LOW risk"
- "93.3% of documented violations were rated HIGH or CRITICAL"
- "This is zero-shot - no domain-specific training required"
