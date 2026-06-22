[README.md](https://github.com/user-attachments/files/23917838/README.md)

## Femtech Privacy Detection: Automated Deception Detection in Reproductive Health App Policies

**Author:** Alexis Prieto  
MS Computer Science Research, Texas A&M University-San Antonio, 2025 



---

## Project Overview

This project applies deep learning techniques to detect privacy policy deception in reproductive health (femtech) applications. We frame privacy policy violations as a **human-induced data disaster** affecting over 100 million users, with documented harms including:

- **$59.5 million** class action settlement (Flo Health)
- **$250,000** California AG fine (Glow)
- Exposure of sensitive reproductive data to Facebook, employers, and data brokers
- Criminal exposure risks in post-Dobbs legal environment

## Why This Matters Now

The Supreme Court's 2022 decision in Dobbs v. Jackson Women's Health Organization fundamentally changed the legal stakes of reproductive health data. In at least 13 states, seeking or providing abortion services carries criminal penalties. Femtech applications collect some of the most sensitive data a person can generate, including menstrual cycles, pregnancy status, and fertility patterns, yet the industry has a documented pattern of sharing that data with third parties while claiming otherwise in their privacy policies. This is not a theoretical risk. FTC enforcement actions, state AG settlements, and investigative journalism have confirmed that the gap between what these apps promise and what they actually do is real, measurable, and harmful. This project builds a system to detect that gap automatically, before regulators catch it.


---

## Key Results

| Model/Approach | Metric | Result |
|----------------|--------|--------|
| Initial BERT | Classification Accuracy | 37.8% |
| Weighted BERT | Classification Accuracy | **59.5%** |
| BERT on Violations | Detection Rate | 22.2% |
| Hybrid (BERT + Rules) | Detection Rate | 52.0% |
| **NLI (DeBERTa)** | **Contradiction Score** | **76.4%** |
| Integrated Risk Scoring | Regulatory Alignment | **100%** |

### Novel Contribution

First application of **Natural Language Inference (NLI)** to privacy policy deception detection, achieving 76.4% average contradiction scores and 100% alignment with regulatory enforcement outcomes.

---
## Repository Structure

```
├── src/
│   ├── bert_classifier.py      # BERT classification with class weighting
│   ├── nli_detector.py         # DeBERTa NLI contradiction detection
│   └── requirements.txt        # Python dependencies
├── data/
│   ├── sample_labeled_sentences.csv    # Sample training data
│   └── violations_policy_pairs.csv     # Violation-policy pairs for NLI
├── results/
│   ├── nli_results.csv         # NLI contradiction scores
│   ├── risk_scores.csv         # Integrated risk rankings
│   └── bert_training_results.txt # Training metrics
├── models/
│   ├── model_config.json       # Model hyperparameters
│   └── README.md               # How to load models
├── visualizations/
│   ├── 01_privacy_risk_ranking_vertical.png
│   ├── 02_privacy_risk_ranking_horizontal.png
│   ├── 03_category_distribution_all_apps.png
│   ├── ... (13 total visualizations)
│   └── README.md               # Chart descriptions
├── demo/
│   ├── Femtech_Privacy_Demo.ipynb  # Google Colab demo
│   ├── streamlit_app.py            # Streamlit web demo
│   └── README.md                   # Demo instructions
├── .github/workflows/
│   └── ml-pipeline.yml         # CI/CD pipeline
└── README.md
```

---

## Live Demo

### Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aprie06/femtech-privacy-disaster-detection/blob/main/demo/Femtech_Privacy_Demo.ipynb)

Click the badge above to run the interactive demo in Google Colab. No installation required.


## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/aprie06/femtech-privacy-disaster-detection.git
cd femtech-privacy-disaster-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r src/requirements.txt
```

---

## Usage

### 1. BERT Classification

Train BERT classifier on labeled privacy policy sentences:

```bash
python src/bert_classifier.py \
    --data data/sample_labeled_sentences.csv \
    --output results/bert_output
```

**Output:**
- `best_model.pt` - Trained model checkpoint
- `confusion_matrix.png` - Classification visualization
- `training_history.png` - Loss/accuracy curves
- `results_summary.txt` - Performance metrics

### 2. NLI Contradiction Detection

Analyze policy-violation pairs for contradictions:

```bash
python src/nli_detector.py \
    --violations data/violations_policy_pairs.csv \
    --output results/nli_output
```

**Output:**
- `nli_results.csv` - Contradiction scores for each pair
- `risk_scores.csv` - Integrated risk rankings by app
- `nli_analysis.png` - Visualization dashboard

---

## Dataset

### Privacy Policy Sentences (1,516 total)

| Application | Sentences | Documented Violations |
|-------------|-----------|----------------------|
| Flo | 312 | 5 (FTC, Class Action) |
| Clue | 245 | 2 (Data Broker) |
| Ovia | 228 | 3 (Employer Sharing) |
| Glow | 198 | 2 (CA AG Fine) |
| Period Tracker | 167 | 1 |
| Maya | 142 | 2 (Facebook SDK) |
| MIA Fem | 118 | 2 (Facebook Sharing) |
| Apple Health | 106 | 0 (Control) |

### 9-Category Classification Schema

| Label | Category | Distribution |
|-------|----------|--------------|
| 0 | Data Collection | 37.1% |
| 1 | Data Sharing | 26.6% |
| 2 | Data Retention | 8.2% |
| 3 | User Rights | 7.8% |
| 4 | Security Measures | 6.4% |
| 5 | Third Party | 5.1% |
| 6 | Policy Changes | 4.6% |
| 7 | Contact Info | 2.9% |
| 8 | Legal Compliance | 1.3% |

---

## Methodology

### Approach

1. What privacy categories appear most frequently in femtech policies?
2. How effectively can BERT classify privacy policy sentences?
3. **Can NLI detect semantic contradictions between policy claims and practices?** (Primary)
4. How do classification and contradiction detection complement each other?
5. Can integrated risk scoring predict regulatory enforcement likelihood?

### Stage 1: BERT Classification

- Model: `bert-base-uncased` (110M parameters)
- Training: 3 epochs, lr=2e-5, batch_size=16
- **Key Innovation:** Inverse frequency class weighting to handle 9.8x class imbalance
- Result: 59.5% accuracy (improved from 37.8% baseline)

### Stage 2: NLI Contradiction Detection

- Model: `microsoft/deberta-large-mnli` (400M parameters)
- Approach: Zero-shot (no domain fine-tuning)
- Formulation:
  - **Premise:** Documented violation (e.g., "Flo shared pregnancy data with Facebook")
  - **Hypothesis:** Policy claim (e.g., "We do not share health information")
- Result: 76.4% average contradiction score

### Stage 3: Integrated Risk Scoring

| Component | Weight |
|-----------|--------|
| NLI Contradiction Score | 40% |
| Documented Violation Count | 30% |
| Severity Distribution | 20% |
| High-Risk Violation Types | 10% |

---

## Results Validation

The integrated risk scoring achieved **100% alignment** with regulatory enforcement:

| App | Risk Score | Level | Regulatory Action |
|-----|------------|-------|-------------------|
| Flo | 73.5 | CRITICAL | FTC + $59.5M Settlement |
| Ovia | 72.6 | CRITICAL | WashPost Investigation |
| Glow | 58.2 | HIGH | $250K CA AG Fine |
| Apple Health | 12.4 | LOW | None (Control) |

---

## Ethical Considerations

1. **Disaster Context:** Privacy violations in reproductive health apps constitute a human-induced data disaster with real harms to millions of users.

2. **Bias Considerations:** Model trained on English-language policies from US-market apps; may not generalize to other regions.

3. **Validation:** All findings validated against public regulatory enforcement data (FTC, state AGs, investigative journalism).

4. **Limitations:** Cannot detect undiscovered violations; relies on documented evidence as ground truth.

---

## Citation

```bibtex
@article{prieto2025femtech,
  title={Detecting Privacy Policy Deception in Reproductive Health Applications: 
         A Deep Learning Approach to Human-Induced Data Disasters},
  author={Prieto, Alexis},
  journal={ML/Deep Learning Research},
  year={2025}
}
```

---

## References

1. Dong, S., et al. (2022). Privacy Analysis of Period Tracking Applications. USENIX Security.
2. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. NAACL-HLT.
3. He, P., et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. ICLR.
4. FTC. (2021). FTC Finalizes Order with Flo Health.
5. Consumer Reports. (2020). Menstrual-Tracking App Glow Faces $250,000 Fine.

---

## License

This project is for educational and research purposes. MIT License.

---

## Contact

For questions or collaboration inquiries, please open an issue on this repository.
