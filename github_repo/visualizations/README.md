# Visualizations

This directory contains all visualizations generated during the femtech privacy policy analysis.

## Chart Descriptions

### Privacy Risk Analysis

| File | Description |
|------|-------------|
| `01_privacy_risk_ranking_vertical.png` | Bar chart showing privacy risk percentage by app (vertical orientation) |
| `02_privacy_risk_ranking_horizontal.png` | Bar chart showing privacy risk percentage by app (horizontal orientation) |

### Category Distribution

| File | Description |
|------|-------------|
| `03_category_distribution_all_apps.png` | Distribution of 9 privacy categories across all 1,516 sentences |
| `04_data_collection_by_app.png` | Percentage of Data Collection sentences by application |

### Analysis Summaries

| File | Description |
|------|-------------|
| `05_summary_tables_analysis.png` | Summary tables showing privacy risk by app and category breakdown |
| `06_analysis_output.png` | Detailed analysis output with confidence scores |

### Data Sharing Analysis

| File | Description |
|------|-------------|
| `07_data_sharing_analysis.png` | Analysis of protective vs. concerning data sharing language |
| `08_data_sharing_observations.png` | Key observations about data sharing practices with examples |

### Violation Detection

| File | Description |
|------|-------------|
| `09_violation_detection.png` | Hybrid approach violation detection results |
| `10_detection_rates_examples.png` | Detection rates by severity with example discrepancies |

### NLI & Risk Scoring

| File | Description |
|------|-------------|
| `11_nli_analysis.png` | NLI contradiction detection results |
| `12_risk_scoring.png` | Integrated risk scoring system output |
| `13_final_results.png` | Final risk rankings and regulatory alignment |

## Key Metrics Shown

- **Privacy Risk Percentage**: Clue (77.1%), Ovia (70.4%), Maya (62.1%), Flo (60.4%), Glow (57.9%)
- **Category Distribution**: Data Collection (37.1%), Data Sharing (26.6%), User Rights (20.4%)
- **Detection Improvement**: Baseline 22.2% → Hybrid 52.0% (+29.8 pts)
- **NLI Performance**: 76.4% average contradiction score
- **Regulatory Alignment**: 100%

## Generating New Visualizations

To regenerate visualizations, run:

```python
# See src/bert_classifier.py for confusion matrix and training curves
# See src/nli_detector.py for NLI analysis visualizations
```
