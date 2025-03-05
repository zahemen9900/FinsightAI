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
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap

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
        
        # Set research-quality visualization style
        self.set_publication_style()
        
        # Enhanced color palette for research publications
        self.colors = {
            'base': '#1A5276',      # Deep blue for base model
            'qlora': '#C0392B',     # Dark red for qlora model
            'grid': '#E5E8E8',      # Light gray for grid
            'text': '#17202A',      # Almost black for text
            'background': '#FFFFFF', # White background for publications
            'accent1': '#2471A3',   # Additional accent colors
            'accent2': '#943126',
            'light_base': '#AED6F1', # Lighter versions for fills
            'light_qlora': '#F5B7B1'
        }
        
        # Load data
        self.base_data = self.load_results(self.base_dir)
        self.qlora_data = self.load_results(self.qlora_dir)
        
        # Timestamp for saved files
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create custom colormaps for better data visualization
        self.create_custom_colormaps()

    def create_custom_colormaps(self):
        """Create custom colormaps for more visually appealing plots"""
        # Create a custom diverging colormap for heatmaps
        self.cmap_diverging = LinearSegmentedColormap.from_list(
            'custom_diverging',
            ['#D32F2F', '#FFFFFF', '#388E3C'], 
            N=256
        )
        
        # Create sequential colormaps
        self.cmap_sequential_blue = LinearSegmentedColormap.from_list(
            'seq_blue',
            ['#FFFFFF', '#1A5276'],
            N=256
        )
        
        self.cmap_sequential_red = LinearSegmentedColormap.from_list(
            'seq_red',
            ['#FFFFFF', '#C0392B'],
            N=256
        )

    def set_publication_style(self):
        """Set publication-ready style for matplotlib plots"""
        # Use LaTeX for high-quality text rendering if available
        try:
            plt.rcParams['text.usetex'] = True
            # Fall back to standard serif font if LaTeX unavailable
        except:
            plt.rcParams['text.usetex'] = False
        
        # Set publication-quality parameters
        plt.rcParams.update({
            # Figure properties
            'figure.figsize': (8, 6),  # Default figure size
            'figure.dpi': 300,         # Publication-quality DPI
            'figure.facecolor': '#FFFFFF',
            'figure.autolayout': True, # Better layout management
            
            # Font properties
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
            'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
            'font.size': 11,        # Base font size
            
            # Axes properties
            'axes.facecolor': '#FFFFFF',
            'axes.linewidth': 0.8,  # Thinner, more elegant spines
            'axes.grid': True,
            'axes.grid.axis': 'both',
            'axes.grid.which': 'major',
            'axes.titlesize': 14,   # Slightly larger title
            'axes.labelsize': 12,   # Label size
            'axes.labelpad': 8,     # More space for labels
            'axes.formatter.use_mathtext': True,  # Better math rendering
            'axes.axisbelow': True, # Grid lines behind data
            
            # Grid properties
            'grid.color': '#E5E8E8',
            'grid.linestyle': '--',
            'grid.linewidth': 0.5,
            'grid.alpha': 0.7,
            
            # Legend properties
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'legend.facecolor': '#FFFFFF',
            'legend.edgecolor': '#CCCCCC',
            'legend.fancybox': True,
            'legend.fontsize': 10,
            'legend.title_fontsize': 11,
            'legend.borderpad': 0.7,
            'legend.labelspacing': 0.7,
            'legend.handletextpad': 0.5,
            'legend.markerscale': 1.0,
            'legend.loc': 'best',
            
            # Tick properties
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'xtick.major.size': 3.5,
            'ytick.major.size': 3.5,
            'xtick.minor.size': 2.0,
            'ytick.minor.size': 2.0,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.minor.width': 0.6,
            'ytick.minor.width': 0.6,
            'xtick.major.pad': 3.5,
            'ytick.major.pad': 3.5,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            
            # Save figure properties
            'savefig.dpi': 600,     # Very high DPI for publication
            'savefig.format': 'pdf', # PDF is better for publications
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'savefig.transparent': False,
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
        """Publication-quality comparison plot with error bars and styling"""
        # Create figure with appropriate size for publications
        fig, ax = plt.subplots(figsize=(10, 6))
        
        base_data = self.base_data['aggregated']
        qlora_data = self.qlora_data['aggregated']
        
        base_metric = base_data[base_data['metric'] == metric]
        qlora_metric = qlora_data[qlora_data['metric'] == metric]
        
        datasets = base_metric['dataset'].unique()
        x = np.arange(len(datasets))
        width = 0.35
        
        # Create bars with publication-quality styling
        bars1 = ax.bar(x - width/2, base_metric['mean'], width, 
                label='Base Model', color=self.colors['base'], alpha=0.9,
                edgecolor='white', linewidth=0.8, zorder=3)
        bars2 = ax.bar(x + width/2, qlora_metric['mean'], width,
                label='QLoRA Model', color=self.colors['qlora'], alpha=0.9,
                edgecolor='white', linewidth=0.8, zorder=3)
        
        # Add error bars
        ax.errorbar(x - width/2, base_metric['mean'], yerr=base_metric['std'],
                    fmt='none', color=self.colors['text'], capsize=3, alpha=0.7,
                    capthick=0.8, linewidth=0.8, zorder=4)
        ax.errorbar(x + width/2, qlora_metric['mean'], yerr=qlora_metric['std'],
                    fmt='none', color=self.colors['text'], capsize=3, alpha=0.7,
                    capthick=0.8, linewidth=0.8, zorder=4)
        
        # Enhance aesthetics for publication
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel(f'{metric.upper()} Score', fontsize=12)
        ax.set_title(f'Comparison of {metric.upper()} Performance', fontsize=14)
        
        # Format x-axis labels
        formatted_datasets = [d.replace('_', ' ').title() for d in datasets]
        ax.set_xticks(x)
        ax.set_xticklabels(formatted_datasets, rotation=30, ha='right')
        
        # Add a horizontal grid for better readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        
        # Format y-axis with limited decimal places
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
        
        # Add value labels on bars with slightly better placement
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', va='bottom',
                   fontsize=9, color=self.colors['text'])
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', va='bottom',
                   fontsize=9, color=self.colors['text'])
        
        # Enhanced legend
        legend = ax.legend(frameon=True, fancybox=True, framealpha=0.9,
                         loc='upper right', title="Model Type")
        legend.get_title().set_fontweight('bold')
        
        # Set y-axis to start at 0 for proper comparison
        ax.set_ylim(bottom=0)
        
        # Add subtle border around the plot
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(0.8)
        
        # Add a source note for publication context
        plt.figtext(0.99, 0.01, f'FinsightAI Evaluation {self.timestamp}',
                  ha='right', va='bottom', fontsize=8, style='italic', alpha=0.7)
        
        # Save with publication-level quality
        plt.tight_layout()
        # Save as PDF (for publication) and PNG (for easy viewing)
        plt.savefig(self.output_dir / f"{metric}_comparison_{self.timestamp}.pdf")
        plt.savefig(self.output_dir / f"{metric}_comparison_{self.timestamp}.png", dpi=300)
        plt.close(fig)

    def plot_overall_improvements(self):
        """Create a publication-quality heatmap showing improvements"""
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
        
        # Format dataset and metric names for better display
        improvements.index = [idx.replace('_', ' ').title() for idx in improvements.index]
        improvements.columns = [col.upper() for col in improvements.columns]
        
        # Determine symmetric range for colormap - show positive/negative with same scale
        abs_max = max(abs(improvements.min().min()), abs(improvements.max().max()))
        
        # Create figure with appropriate size
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap with enhanced styling
        cmap = self.cmap_diverging
        sns.heatmap(improvements, annot=True, cmap=cmap, center=0,
                    fmt='.2f', cbar_kws={'label': 'Improvement %', 'shrink': 0.8},
                    linewidths=0.5, linecolor='#FFFFFF',
                    annot_kws={"size": 10, "weight": "bold"},
                    vmin=-abs_max, vmax=abs_max,
                    ax=ax)
        
        # Enhanced title and layout
        ax.set_title('QLoRA Model Performance Improvement (%)\nCompared to Base Model',
                  fontsize=16, pad=20)
        
        # Rotate y-axis labels for better readability
        plt.yticks(rotation=0)
        
        # Add title to x and y axes
        ax.set_xlabel('Evaluation Metric', fontsize=12, labelpad=10)
        ax.set_ylabel('Dataset', fontsize=12, labelpad=10)
        
        # Custom annotation to highlight best improvements
        max_val = improvements.max().max()
        max_loc = np.where(improvements.values == max_val)
        if len(max_loc[0]) > 0:
            i, j = max_loc[0][0], max_loc[1][0]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
        
        # Custom colorbar ticks
        cbar = ax.collections[0].colorbar
        tick_values = np.linspace(-abs_max, abs_max, 7)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f"{x:.1f}%" for x in tick_values])
        
        # Add source note for academic context
        plt.figtext(0.99, 0.01, f'FinsightAI Evaluation {self.timestamp}',
                  ha='right', va='bottom', fontsize=8, style='italic', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"overall_improvements_{self.timestamp}.pdf")
        plt.savefig(self.output_dir / f"overall_improvements_{self.timestamp}.png", dpi=300)
        plt.close(fig)

    def plot_response_distribution(self):
        """Publication-quality distribution plots for research papers"""
        # First, check what columns we actually have
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
        
        # Create single plot with separate axes for better layout control
        fig = plt.figure(figsize=(12, 10))
        
        # Add a common title for the entire figure
        fig.suptitle('Model Response Quality Distribution Comparison', 
                   fontsize=16, fontweight='bold', y=0.98)
        
        # Add subtle caption with paper-like description
        plt.figtext(0.5, 0.94, 
                  'Kernel density estimate plots showing distribution of evaluation metrics',
                  ha='center', fontsize=10, fontstyle='italic')
        
        for i, (metric, title) in enumerate(metrics):
            ax = plt.subplot(2, 2, i+1)
            
            # Create publication-quality KDE plots with better colors and styling
            sns.kdeplot(data=self.base_data['samples'], x=metric,
                       label='Base Model', color=self.colors['base'],
                       ax=ax, fill=True, alpha=0.2, linewidth=1.5)
            sns.kdeplot(data=self.qlora_data['samples'], x=metric,
                       label='QLoRA Model', color=self.colors['qlora'],
                       ax=ax, fill=True, alpha=0.2, linewidth=1.5)
            
            # Calculate and display means with enhanced visibility
            base_mean = self.base_data['samples'][metric].mean()
            qlora_mean = self.qlora_data['samples'][metric].mean()
            
            ax.axvline(base_mean, color=self.colors['base'], linestyle='-', linewidth=1.5,
                      label=f'Base Mean: {base_mean:.3f}')
            ax.axvline(qlora_mean, color=self.colors['qlora'], linestyle='-', linewidth=1.5,
                      label=f'QLoRA Mean: {qlora_mean:.3f}')
            
            # Calculate improvement percentage for annotation
            improvement_pct = ((qlora_mean - base_mean) / base_mean) * 100
            
            # Add improvement annotation
            if improvement_pct > 0:
                improvement_text = f"+{improvement_pct:.2f}%"
                color = 'darkgreen'
            else:
                improvement_text = f"{improvement_pct:.2f}%"
                color = 'darkred'
                
            # Add text annotation to show improvement
            ax.annotate(improvement_text,
                       xy=(0.7, 0.9), xycoords='axes fraction',
                       bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.8),
                       ha='center', va='center', fontsize=10, fontweight='bold', color=color)
            
            # Enhanced axis labels and styling for publication
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel('Score', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            
            # Custom legend with better styling
            handles, labels = ax.get_legend_handles_labels()
            # Reorder to group means with their distributions
            order = [0, 2, 1, 3]
            handles = [handles[i] for i in order]
            labels = [labels[i] for i in order]
            ax.legend(handles, labels, frameon=True, facecolor='white',
                    framealpha=0.8, fontsize=9, loc='upper left')
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Set x-axis limits based on the data with some padding
            x_min = min(self.base_data['samples'][metric].min(),
                       self.qlora_data['samples'][metric].min()) - 0.02
            x_max = max(self.base_data['samples'][metric].max(),
                       self.qlora_data['samples'][metric].max()) + 0.02
            ax.set_xlim(max(0, x_min), x_max)  # Ensure we don't go below 0 for metrics
            
            # Format tick labels
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
            
        # Add source annotation for publication context
        plt.figtext(0.99, 0.01, f'FinsightAI Evaluation {self.timestamp}',
                  ha='right', va='bottom', fontsize=8, style='italic', alpha=0.7)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])  # Adjust for the overall title
        plt.savefig(self.output_dir / f"response_distributions_{self.timestamp}.pdf")
        plt.savefig(self.output_dir / f"response_distributions_{self.timestamp}.png", dpi=300)
        plt.close(fig)

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

    def generate_radar_chart(self):
        """Generate a publication-quality radar chart comparing model performance"""
        metrics = ['rouge1', 'rouge2', 'rougeL', 'bleu']
        
        # Calculate average performance across datasets
        base_means = []
        qlora_means = []
        
        for metric in metrics:
            base_mean = self.base_data['aggregated'][
                self.base_data['aggregated']['metric'] == metric]['mean'].mean()
            qlora_mean = self.qlora_data['aggregated'][
                self.qlora_data['aggregated']['metric'] == metric]['mean'].mean()
                
            base_means.append(base_mean)
            qlora_means.append(qlora_mean)
        
        # Formatted labels for the chart
        labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU']
        
        # Set up radar chart with LaTeX-quality styling
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Compute angle for each metric
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        # Close the polygon
        base_means += [base_means[0]]
        qlora_means += [qlora_means[0]]
        angles += [angles[0]]
        labels += [labels[0]]
        
        # Plot data
        ax.plot(angles, base_means, 'o-', linewidth=2, label='Base Model', 
               color=self.colors['base'], alpha=0.8)
        ax.fill(angles, base_means, color=self.colors['light_base'], alpha=0.25)
        
        ax.plot(angles, qlora_means, 'o-', linewidth=2, label='QLoRA Model',
               color=self.colors['qlora'], alpha=0.8)
        ax.fill(angles, qlora_means, color=self.colors['light_qlora'], alpha=0.25)
        
        # Fix axis to start from top
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Set labels with enhanced styling
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels[:-1], fontsize=12, fontweight='bold')
        
        # Set limits for better visualization
        max_val = max(max(base_means), max(qlora_means)) * 1.1
        ax.set_ylim(0, max_val)
        
        # Add subtle gridlines with custom styling
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Custom y-ticks to show score values
        y_ticks = np.linspace(0, max_val, 6)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{x:.2f}' for x in y_ticks], fontsize=9)
        
        # Add legend with enhanced styling
        legend = ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1),
                         frameon=True, framealpha=0.8, facecolor='white',
                         edgecolor='#CCCCCC', fontsize=10)
                         
        # Add title
        plt.title('Model Performance Comparison Across Metrics', 
                y=1.08, fontsize=14, fontweight='bold')
        
        # Add caption for publication context
        plt.figtext(0.5, 0.01, 
                  'Radar chart showing average performance scores across evaluation metrics',
                  ha='center', fontsize=9, fontstyle='italic')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"radar_chart_{self.timestamp}.pdf")
        plt.savefig(self.output_dir / f"radar_chart_{self.timestamp}.png", dpi=300)
        plt.close(fig)

    def generate_report(self):
        """Generate a comprehensive visualization report"""
        logger.info("Generating publication-quality visualization report...")
        
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
        
        # Create radar chart
        logger.info("Generating radar chart")
        self.generate_radar_chart()
        
        # Create summary table
        logger.info("Creating summary table")
        summary_df = self.create_summary_table()
        
        logger.info(f"Report generated successfully. Publication-quality visualizations saved to {self.output_dir}")
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
