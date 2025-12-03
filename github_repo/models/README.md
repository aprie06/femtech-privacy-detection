# Models Directory

This directory contains trained model checkpoints and configuration files.

## Files

### `best_bert_model.pt`
- Fine-tuned BERT model for privacy policy classification
- Base model: `bert-base-uncased` (110M parameters)
- Trained on 1,213 sentences with class weighting
- Achieves 59.5% classification accuracy

### `model_config.json`
- Model configuration and hyperparameters
- Required for loading the saved model

## DeBERTa Model

The NLI model (DeBERTa) is NOT included in this repository due to size (1.5GB).

To use DeBERTa for contradiction detection, load directly from Hugging Face:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "microsoft/deberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

## Loading the BERT Model

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Initialize model architecture
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=9
)

# Load trained weights
model.load_state_dict(torch.load('models/best_bert_model.pt'))
model.eval()
```

## Label Mapping

```python
LABEL_NAMES = {
    0: 'Data Collection',
    1: 'Data Sharing',
    2: 'Data Retention',
    3: 'User Rights',
    4: 'Security Measures',
    5: 'Third Party',
    6: 'Policy Changes',
    7: 'Contact Info',
    8: 'Legal Compliance'
}
```
