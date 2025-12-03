"""
Femtech Privacy Policy Deception Detection - NLI Contradiction Detection
=========================================================================
This module implements Natural Language Inference using DeBERTa to detect
semantic contradictions between privacy policy claims and documented
data practices.

Novel contribution: First application of NLI to privacy policy deception
detection, achieving 76.4% average contradiction score on documented violations.

Author: Alexis Prieto
Course: ML/Deep Learning Research - Fall 2025
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os

# Configuration
NLI_CONFIG = {
    'model_name': 'microsoft/deberta-large-mnli',
    'max_length': 512,
    'severity_thresholds': {
        'CRITICAL': 0.75,
        'HIGH': 0.55,
        'MEDIUM': 0.35,
        'LOW': 0.0
    },
    'label_map': {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
}


class NLIContradictionDetector:
    """
    Detect contradictions between privacy policy claims and documented violations
    using Natural Language Inference with DeBERTa.
    
    The key insight is that privacy deception can be formulated as an NLI problem:
    - Premise: The documented violation (what actually happened)
    - Hypothesis: The policy claim (what was promised)
    
    A high contradiction score indicates the policy claim is deceptive.
    """
    
    def __init__(self, model_name: str = NLI_CONFIG['model_name']):
        """Initialize the NLI detector with DeBERTa model."""
        print(f"Loading {model_name}...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.label_map = NLI_CONFIG['label_map']
        print("NLI detector initialized successfully")
    
    def detect_contradiction(
        self, 
        policy_claim: str, 
        violation_description: str
    ) -> Dict[str, float]:
        """
        Detect if a policy claim contradicts documented violation.
        
        Args:
            policy_claim: The privacy policy statement (hypothesis)
            violation_description: The documented violation (premise)
            
        Returns:
            Dictionary with contradiction, neutral, entailment scores
            and severity classification
        """
        # Tokenize with premise (violation) and hypothesis (policy claim)
        inputs = self.tokenizer(
            violation_description,  # premise - what actually happened
            policy_claim,           # hypothesis - what policy claimed
            return_tensors='pt',
            truncation=True,
            max_length=NLI_CONFIG['max_length'],
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
        
        scores = {
            'contradiction': probs[0].item(),
            'neutral': probs[1].item(),
            'entailment': probs[2].item()
        }
        
        # Determine predicted label and confidence
        scores['predicted_label'] = max(scores, key=lambda k: scores[k] if k in self.label_map.values() else 0)
        scores['confidence'] = max(scores['contradiction'], scores['neutral'], scores['entailment'])
        
        # Classify severity based on contradiction score
        scores['severity'] = self._classify_severity(scores['contradiction'])
        
        return scores
    
    def _classify_severity(self, contradiction_score: float) -> str:
        """Classify contradiction severity based on score thresholds."""
        thresholds = NLI_CONFIG['severity_thresholds']
        
        if contradiction_score >= thresholds['CRITICAL']:
            return 'CRITICAL'
        elif contradiction_score >= thresholds['HIGH']:
            return 'HIGH'
        elif contradiction_score >= thresholds['MEDIUM']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def analyze_violations(self, violations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze all violation-policy pairs in a DataFrame.
        
        Expected columns:
        - policy_sentence: The privacy policy claim
        - violation_description: The documented violation
        - app_name: Application name
        - violation_type: Type of violation
        """
        results = []
        
        for idx, row in violations_df.iterrows():
            print(f"Processing {idx+1}/{len(violations_df)}: {row.get('app_name', 'Unknown')}")
            
            scores = self.detect_contradiction(
                policy_claim=row['policy_sentence'],
                violation_description=row['violation_description']
            )
            
            result = {
                'app_name': row.get('app_name', 'Unknown'),
                'violation_type': row.get('violation_type', 'Unknown'),
                'policy_sentence': row['policy_sentence'],
                'violation_description': row['violation_description'],
                'contradiction_score': scores['contradiction'],
                'neutral_score': scores['neutral'],
                'entailment_score': scores['entailment'],
                'predicted_label': scores['predicted_label'],
                'confidence': scores['confidence'],
                'severity': scores['severity']
            }
            
            # Copy any additional columns
            for col in violations_df.columns:
                if col not in result:
                    result[col] = row[col]
            
            results.append(result)
        
        return pd.DataFrame(results)


def compute_risk_scores(nli_results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute integrated privacy risk scores (0-100) for each application.
    
    Components:
    - NLI contradiction score (40% weight)
    - Documented violation count (30% weight)
    - Severity distribution (20% weight)
    - High-risk violation types (10% weight)
    """
    apps = nli_results['app_name'].unique()
    risk_scores = []
    
    severity_weights = {'CRITICAL': 1.0, 'HIGH': 0.7, 'MEDIUM': 0.4, 'LOW': 0.1}
    high_risk_types = ['Data Sharing', 'Employer Sharing', 'Facebook SDK', 'Location']
    
    for app in apps:
        app_data = nli_results[nli_results['app_name'] == app]
        
        # Component 1: Average NLI contradiction score (0-40 points)
        avg_contradiction = app_data['contradiction_score'].mean()
        nli_component = avg_contradiction * 40
        
        # Component 2: Violation count normalized (0-30 points)
        violation_count = len(app_data)
        violation_component = min(violation_count / 10, 1) * 30
        
        # Component 3: Severity distribution (0-20 points)
        severity_scores = app_data['severity'].map(severity_weights)
        severity_component = severity_scores.mean() * 20
        
        # Component 4: High-risk violation types (0-10 points)
        high_risk_count = app_data[app_data['violation_type'].isin(high_risk_types)].shape[0]
        type_component = min(high_risk_count / 5, 1) * 10
        
        total_score = nli_component + violation_component + severity_component + type_component
        
        # Determine risk level
        if total_score >= 70:
            risk_level = 'CRITICAL'
        elif total_score >= 50:
            risk_level = 'HIGH'
        elif total_score >= 30:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        risk_scores.append({
            'app_name': app,
            'risk_score': round(total_score, 1),
            'risk_level': risk_level,
            'nli_component': round(nli_component, 1),
            'violation_component': round(violation_component, 1),
            'severity_component': round(severity_component, 1),
            'type_component': round(type_component, 1),
            'avg_contradiction': round(avg_contradiction, 3),
            'violation_count': violation_count
        })
    
    return pd.DataFrame(risk_scores).sort_values('risk_score', ascending=False)


def plot_nli_results(nli_results: pd.DataFrame, output_dir: str):
    """Generate visualizations for NLI analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('NLI Contradiction Analysis: Privacy Policy Deception', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Contradiction scores by app
    ax1 = axes[0, 0]
    app_scores = nli_results.groupby('app_name')['contradiction_score'].mean().sort_values(ascending=False)
    colors = ['#dc2626' if x >= 0.6 else '#ea580c' if x >= 0.4 else '#16a34a' for x in app_scores.values]
    bars = ax1.bar(app_scores.index, app_scores.values, color=colors, edgecolor='black')
    ax1.set_ylabel('Average Contradiction Score')
    ax1.set_title('Policy-Practice Contradiction by App')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Severity distribution
    ax2 = axes[0, 1]
    severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    severity_colors = {'CRITICAL': '#dc2626', 'HIGH': '#ea580c', 'MEDIUM': '#ca8a04', 'LOW': '#16a34a'}
    severity_counts = nli_results['severity'].value_counts().reindex(severity_order, fill_value=0)
    ax2.pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%',
            colors=[severity_colors[s] for s in severity_counts.index], startangle=90)
    ax2.set_title('Violation Severity Distribution')
    
    # Plot 3: Contradiction by violation type
    ax3 = axes[1, 0]
    type_scores = nli_results.groupby('violation_type')['contradiction_score'].mean().sort_values(ascending=False)
    ax3.barh(type_scores.index, type_scores.values, color='steelblue', edgecolor='black')
    ax3.set_xlabel('Average Contradiction Score')
    ax3.set_title('Contradiction by Violation Type')
    ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
    
    # Plot 4: Score distribution histogram
    ax4 = axes[1, 1]
    ax4.hist(nli_results['contradiction_score'], bins=20, color='steelblue', 
             edgecolor='black', alpha=0.7)
    ax4.axvline(x=0.75, color='#dc2626', linestyle='--', label='CRITICAL threshold')
    ax4.axvline(x=0.55, color='#ea580c', linestyle='--', label='HIGH threshold')
    ax4.set_xlabel('Contradiction Score')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Contradiction Scores')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'nli_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"NLI analysis visualization saved to {output_dir}/nli_analysis.png")


def main(violations_path: str, output_dir: str = './results'):
    """
    Main NLI analysis pipeline.
    
    Args:
        violations_path: Path to CSV with violation-policy pairs
        output_dir: Directory for saving results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load violations data
    print("Loading violation-policy pairs...")
    violations_df = pd.read_csv(violations_path)
    print(f"Loaded {len(violations_df)} violation-policy pairs")
    
    # Initialize NLI detector
    detector = NLIContradictionDetector()
    
    # Analyze all violations
    print("\nAnalyzing contradictions...")
    nli_results = detector.analyze_violations(violations_df)
    
    # Save NLI results
    nli_results.to_csv(os.path.join(output_dir, 'nli_results.csv'), index=False)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("NLI CONTRADICTION ANALYSIS SUMMARY")
    print("="*60)
    print(f"Average contradiction score: {nli_results['contradiction_score'].mean():.3f}")
    print(f"Violations detected as contradictions: {(nli_results['predicted_label'] == 'contradiction').sum()}/{len(nli_results)}")
    print(f"\nSeverity breakdown:")
    print(nli_results['severity'].value_counts())
    print(f"\nBy app:")
    print(nli_results.groupby('app_name')['contradiction_score'].mean().sort_values(ascending=False))
    
    # Compute risk scores
    print("\nComputing integrated risk scores...")
    risk_scores = compute_risk_scores(nli_results)
    risk_scores.to_csv(os.path.join(output_dir, 'risk_scores.csv'), index=False)
    
    print("\n" + "="*60)
    print("INTEGRATED PRIVACY RISK RANKINGS")
    print("="*60)
    print(risk_scores[['app_name', 'risk_score', 'risk_level', 'violation_count']].to_string(index=False))
    
    # Generate visualizations
    plot_nli_results(nli_results, output_dir)
    
    print(f"\nAll results saved to {output_dir}")
    
    return nli_results, risk_scores


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='NLI Contradiction Detection for Privacy Policies')
    parser.add_argument('--violations', type=str, required=True, help='Path to violations CSV')
    parser.add_argument('--output', type=str, default='./results', help='Output directory')
    args = parser.parse_args()
    
    main(args.violations, args.output)
