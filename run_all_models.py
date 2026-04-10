import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path('.')

MODELS = [
    {
        'name': 'CNN Computer Vision',
        'script': 'cnn_computer_vision_cattle.py',
        'output_dir': 'cnn_cv_cattle_outputs',
    },
    {
        'name': 'Ensemble Learning + CV',
        'script': 'ensemble_learning_computer_vision_cattle.py',
        'output_dir': 'ensemble_cv_cattle_outputs',
    },
    {
        'name': 'Transfer Learning CNN',
        'script': 'transfer_learning_cnn_cattle.py',
        'output_dir': 'tl_cattle_outputs',
    },
    {
        'name': 'Transfer Learning + Attention',
        'script': 'transfer_learning_attention_cattle.py',
        'output_dir': 'tl_attention_cattle_outputs',
    },
]


def run_script(script_name: str):
    print(f"\n{'='*80}")
    print(f"Running: {script_name}")
    print(f"{'='*80}\n")
    result = subprocess.run([sys.executable, script_name], text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {script_name}")



def load_metrics(output_dir: Path, model_name: str) -> dict:
    csv_path = output_dir / 'final_results.csv'
    json_path = output_dir / 'final_results.json'
    summary = {
        'Model': model_name,
        'Accuracy': None,
        'Precision': None,
        'Recall': None,
        'F1': None,
        'AUC': None,
        'Output_Folder': str(output_dir),
    }

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}
        # Use first row or best row depending on file format
        row = df.iloc[0]
        for key, target in [('accuracy', 'Accuracy'), ('acc', 'Accuracy'),
                            ('precision', 'Precision'), ('precision_macro', 'Precision'),
                            ('recall', 'Recall'), ('recall_macro', 'Recall'),
                            ('f1', 'F1'), ('f1_macro', 'F1'), ('auc', 'AUC')]:
            if key in cols:
                summary[target] = float(row[cols[key]])
        return summary

    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Try common shapes
        if isinstance(data, dict):
            for key, target in [('accuracy', 'Accuracy'), ('acc', 'Accuracy'),
                                ('precision', 'Precision'), ('precision_macro', 'Precision'),
                                ('recall', 'Recall'), ('recall_macro', 'Recall'),
                                ('f1', 'F1'), ('f1_macro', 'F1'), ('auc', 'AUC')]:
                if key in data and isinstance(data[key], (int, float)):
                    summary[target] = float(data[key])
        return summary

    return summary



def save_summary(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'master_final_results.csv'
    json_path = out_dir / 'master_final_results.json'
    txt_path = out_dir / 'master_best_model_summary.txt'

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient='records', indent=2)

    available = df.dropna(subset=['Accuracy'])
    if not available.empty:
        best_idx = available['Accuracy'].astype(float).idxmax()
        best_row = available.loc[best_idx]
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('Best Model Summary\n')
            f.write('=' * 40 + '\n')
            for col in df.columns:
                f.write(f"{col}: {best_row[col]}\n")
    else:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('No valid accuracy values found.\n')

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")



def plot_metric(df: pd.DataFrame, metric: str, out_dir: Path):
    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.bar(plot_df['Model'], plot_df[metric])
    plt.title(f'{metric} Comparison')
    plt.ylabel(metric)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    path = out_dir / f'master_{metric.lower()}_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")



def plot_all_metrics(df: pd.DataFrame, out_dir: Path):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    available = [m for m in metrics if m in df.columns and df[m].notna().any()]
    if not available:
        return
    plot_df = df[['Model'] + available].copy().set_index('Model')
    plt.figure(figsize=(12, 6))
    plot_df.plot(kind='bar')
    plt.title('All Model Metrics Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=25, ha='right')
    plt.legend()
    plt.tight_layout()
    path = out_dir / 'master_all_metrics_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")



def main():
    summary_rows = []
    for item in MODELS:
        script = BASE_DIR / item['script']
        if not script.exists():
            print(f"Skipping missing script: {script}")
            continue
        run_script(str(script))
        summary_rows.append(load_metrics(BASE_DIR / item['output_dir'], item['name']))

    if not summary_rows:
        raise RuntimeError('No scripts were run. Make sure the model files are present.')

    df = pd.DataFrame(summary_rows)
    master_dir = BASE_DIR / 'master_model_outputs'
    save_summary(df, master_dir)
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']:
        plot_metric(df, metric, master_dir)
    plot_all_metrics(df, master_dir)

    print('\nFinal comparison table:')
    print(df)
    print(f"\nAll master outputs saved in: {master_dir.resolve()}")


if __name__ == '__main__':
    main()
