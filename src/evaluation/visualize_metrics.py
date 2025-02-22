import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List
import logging
from rich.logging import RichHandler
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler()],
)
logger = logging.getLogger('rich')

class MetricsVisualizer:
    def __init__(
        self,
        base_results_dir: str,
        qlora_results_dir: str,
        output_dir: str = "metrics/visualizations"
    ):
        self.base_dir = Path(base_results_dir)
        self.qlora_dir = Path(qlora_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set custom style for all plots
        plt.style.use('seaborn-v0_8-darkgrid')
        self.set_plot_style()
        
        # Custom colors
        self.colors = {
            'base': '#2ecc71',     # Green
            'qlora': '#3498db',    # Blue
            'grid': '#dddddd',     # Light gray for grid
            'text': '#2c3e50',     # Dark blue-gray for text
            'background': '#f8f9fa' # Light background
        }
        
        # Load data
        self.base_data = self.load_results(self.base_dir)
        self.qlora_data = self.load_results(self.qlora_dir)
        
        # Timestamp for saved files
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def set_plot_style(self):
        """Set consistent style for all plots"""
        plt.rcParams.update({
            'figure.facecolor': '#f8f9fa',
            'axes.facecolor': '#f8f9fa',
            'axes.grid': True,
            'grid.color': '#dddddd',
            'grid.linestyle': '--',
            'grid.alpha': 0.7,
            'axes.labelcolor': '#2c3e50',
            'axes.edgecolor': '#2c3e50',
            'xtick.color': '#2c3e50',
            'ytick.color': '#2c3e50',
            'text.color': '#2c3e50',
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'legend.frameon': True,
            'legend.framealpha': 0.8,
            'legend.edgecolor': '#dddddd',
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.facecolor': '#f8f9fa',
            'figure.titlesize': 16,
            'figure.titleweight': 'bold'
        })

    def load_results(self, results_dir: Path) -> Dict:
        """Load and validate results from a directory"""
        try:
            results_file = results_dir / "results.json"
            metrics_file = results_dir / "aggregated_metrics.csv"
            samples_file = results_dir / "sample_metrics.csv"
            
            if not all(f.exists() for f in [results_file, metrics_file, samples_file]):
                raise FileNotFoundError("Missing required results files")
            
            data = {
                'results': json.loads(results_file.read_text()),
                'aggregated': pd.read_csv(metrics_file),
                'samples': pd.read_csv(samples_file)
            }
            return data
        except Exception as e:
            logger.error(f"Error loading results from {results_dir}: {e}")
            raise

    def plot_metric_comparison(self, metric: str):
        """Enhanced comparison plot with error bars and styling"""
        plt.figure(figsize=(12, 6))
        
        base_data = self.base_data['aggregated']
        qlora_data = self.qlora_data['aggregated']
        
        base_metric = base_data[base_data['metric'] == metric]
        qlora_metric = qlora_data[qlora_data['metric'] == metric]
        
        datasets = base_metric['dataset'].unique()
        x = np.arange(len(datasets))
        width = 0.35
        
        # Create bars with enhanced styling
        plt.bar(x - width/2, base_metric['mean'], width, 
                label='Base Model', color=self.colors['base'], alpha=0.8,
                edgecolor='white', linewidth=1)
        plt.bar(x + width/2, qlora_metric['mean'], width,
                label='QLora Model', color=self.colors['qlora'], alpha=0.8,
                edgecolor='white', linewidth=1)
        
        # Add error bars
        plt.errorbar(x - width/2, base_metric['mean'], yerr=base_metric['std'],
                    fmt='none', color=self.colors['text'], capsize=5, alpha=0.5,
                    capthick=1, linewidth=1)
        plt.errorbar(x + width/2, qlora_metric['mean'], yerr=qlora_metric['std'],
                    fmt='none', color=self.colors['text'], capsize=5, alpha=0.5,
                    capthick=1, linewidth=1)
        
        # Enhance plot aesthetics
        plt.xlabel('Dataset', fontsize=12, color=self.colors['text'])
        plt.ylabel(f'{metric.upper()} Score', fontsize=12, color=self.colors['text'])
        plt.title(f'Comparison of {metric.upper()} Scores\nBase vs QLora Model',
                 pad=20, fontsize=14, color=self.colors['text'])
        
        plt.xticks(x, datasets, rotation=45, ha='right')
        plt.legend(frameon=True, facecolor=self.colors['background'])
        
        # Add value labels on bars
        for i, v in enumerate(base_metric['mean']):
            plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom',
                    color=self.colors['text'], fontsize=9)
        for i, v in enumerate(qlora_metric['mean']):
            plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom',
                    color=self.colors['text'], fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{metric}_comparison_{self.timestamp}.png")
        plt.close()

    def plot_overall_improvements(self):
        """Create a heatmap showing improvements across all metrics and datasets"""
        base_data = self.base_data['aggregated']
        qlora_data = self.qlora_data['aggregated']
        
        # Calculate improvements
        improvements = pd.DataFrame()
        for dataset in base_data['dataset'].unique():
            for metric in base_data['metric'].unique():
                base_value = base_data[(base_data['dataset'] == dataset) & 
                                     (base_data['metric'] == metric)]['mean'].iloc[0]
                qlora_value = qlora_data[(qlora_data['dataset'] == dataset) & 
                                       (qlora_data['metric'] == metric)]['mean'].iloc[0]
                
                improvement = ((qlora_value - base_value) / base_value) * 100
                improvements.loc[dataset, metric] = improvement
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(improvements, annot=True, cmap='RdYlGn', center=0,
                    fmt='.1f', cbar_kws={'label': 'Improvement %'})
        
        plt.title('Percentage Improvement: QLora vs Base Model')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"overall_improvements_{self.timestamp}.png", dpi=300)
        plt.close()

    def plot_response_distribution(self):
        """Enhanced distribution plots with better styling"""
        # First, let's check what columns we actually have
        logger.info("Available columns in base samples:")
        logger.info(self.base_data['samples'].columns.tolist())
        logger.info("Available columns in qlora samples:")
        logger.info(self.qlora_data['samples'].columns.tolist())

        # Use the actual metrics from our data
        metrics = [
            ('rouge1', 'ROUGE-1 Score'),
            ('rouge2', 'ROUGE-2 Score'),
            ('rougeL', 'ROUGE-L Score'),
            ('bleu', 'BLEU Score')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Response Quality Distribution', y=1.02)
        
        for ax, (metric, title) in zip(axes.flat, metrics):
            # Create KDE plots with enhanced styling
            sns.kdeplot(data=self.base_data['samples'], x=metric,
                       label='Base Model', color=self.colors['base'],
                       ax=ax, fill=True, alpha=0.3)
            sns.kdeplot(data=self.qlora_data['samples'], x=metric,
                       label='QLora Model', color=self.colors['qlora'],
                       ax=ax, fill=True, alpha=0.3)
            
            # Calculate and display means
            base_mean = self.base_data['samples'][metric].mean()
            qlora_mean = self.qlora_data['samples'][metric].mean()
            
            ax.axvline(base_mean, color=self.colors['base'], linestyle='--', alpha=0.8,
                      label=f'Base Mean: {base_mean:.3f}')
            ax.axvline(qlora_mean, color=self.colors['qlora'], linestyle='--', alpha=0.8,
                      label=f'QLora Mean: {qlora_mean:.3f}')
            
            # Enhanced axis labels and styling
            ax.set_title(title, pad=10)
            ax.set_xlabel('Score', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.legend(frameon=True, facecolor=self.colors['background'],
                     title='Model Type', title_fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis limits based on the data
            x_min = min(self.base_data['samples'][metric].min(),
                       self.qlora_data['samples'][metric].min())
            x_max = max(self.base_data['samples'][metric].max(),
                       self.qlora_data['samples'][metric].max())
            ax.set_xlim(x_min, x_max)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"response_distributions_{self.timestamp}.png")
        plt.close()

    def create_summary_table(self) -> pd.DataFrame:
        """Create a summary table of key metrics"""
        metrics = ['rouge1', 'rouge2', 'rougeL', 'bleu']
        summary_data = []
        
        for metric in metrics:
            base_mean = self.base_data['aggregated'][
                self.base_data['aggregated']['metric'] == metric]['mean'].mean()
            qlora_mean = self.qlora_data['aggregated'][
                self.qlora_data['aggregated']['metric'] == metric]['mean'].mean()
            improvement = ((qlora_mean - base_mean) / base_mean) * 100
            
            summary_data.append({
                'Metric': metric,
                'Base Model': f"{base_mean:.4f}",
                'QLora Model': f"{qlora_mean:.4f}",
                'Improvement %': f"{improvement:.2f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / f"metrics_summary_{self.timestamp}.csv", index=False)
        return summary_df

    def generate_report(self):
        """Generate a comprehensive visualization report"""
        logger.info("Generating visualization report...")
        
        # Create individual metric comparisons
        metrics = ['rouge1', 'rouge2', 'rougeL', 'bleu']
        for metric in metrics:
            logger.info(f"Plotting comparison for {metric}")
            self.plot_metric_comparison(metric)
        
        # Create overall improvements heatmap
        logger.info("Plotting overall improvements heatmap")
        self.plot_overall_improvements()
        
        # Create response distribution plots
        logger.info("Plotting response distributions")
        self.plot_response_distribution()
        
        # Create summary table
        logger.info("Creating summary table")
        summary_df = self.create_summary_table()
        
        logger.info(f"Report generated successfully. Files saved to {self.output_dir}")
        return summary_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True,
                      help="Directory containing base model results")
    parser.add_argument("--qlora_dir", type=str, required=True,
                      help="Directory containing QLora model results")
    parser.add_argument("--output_dir", type=str, default="metrics/visualizations",
                      help="Directory to save visualizations")
    args = parser.parse_args()
    
    visualizer = MetricsVisualizer(
        base_results_dir=args.base_dir,
        qlora_results_dir=args.qlora_dir,
        output_dir=args.output_dir
    )
    
    summary = visualizer.generate_report()
    print("\nMetrics Summary:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
