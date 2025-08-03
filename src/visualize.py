"""
Data visualization module for breast cancer prediction.
Provides comprehensive EDA and model explainability visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

# Set style for all plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class DataVisualizer:
    """Handles all data visualization for breast cancer prediction."""
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        
    def plot_target_distribution(self, data: pd.DataFrame, target_col: str = "diagnosis",
                                save_path: Optional[str] = None) -> None:
        """
        Plot the distribution of the target variable.
        
        Args:
            data: Input DataFrame
            target_col: Name of the target column
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Create count plot
        ax = sns.countplot(data=data, x=target_col, palette=['lightblue', 'lightcoral'])
        
        # Add value labels on bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', 
                       (p.get_x() + p.get_width()/2., p.get_height()), 
                       ha='center', va='bottom')
        
        plt.title('Distribution of Breast Cancer Diagnosis', fontsize=16, fontweight='bold')
        plt.xlabel('Diagnosis (0: Benign, 1: Malignant)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        # Add percentage labels
        total = len(data)
        for i, p in enumerate(ax.patches):
            percentage = f'{(p.get_height()/total)*100:.1f}%'
            ax.text(p.get_x() + p.get_width()/2., p.get_height() + 5, 
                   percentage, ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Target distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_distributions(self, data: pd.DataFrame, feature_cols: list,
                                 target_col: str = "diagnosis",
                                 save_path: Optional[str] = None) -> None:
        """
        Plot feature distributions by target class.
        
        Args:
            data: Input DataFrame
            feature_cols: List of feature columns to plot
            target_col: Name of the target column
            save_path: Optional path to save the plot
        """
        n_features = len(feature_cols)
        fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 5))
        
        if n_features == 1:
            axes = [axes]
        
        for i, feature in enumerate(feature_cols):
            # Create violin plot
            sns.violinplot(data=data, x=target_col, y=feature, ax=axes[i], 
                          palette=['lightblue', 'lightcoral'])
            
            axes[i].set_title(f'{feature.replace("_", " ").title()} Distribution', 
                             fontweight='bold')
            axes[i].set_xlabel('Diagnosis (0: Benign, 1: Malignant)')
            axes[i].set_ylabel(feature.replace("_", " ").title())
        
        plt.suptitle('Feature Distributions by Diagnosis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature distributions plot saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, data: pd.DataFrame, feature_cols: list,
                              save_path: Optional[str] = None) -> None:
        """
        Plot correlation matrix heatmap.
        
        Args:
            data: Input DataFrame
            feature_cols: List of feature columns to include
            save_path: Optional path to save the plot
        """
        # Calculate correlation matrix
        corr_data = data[feature_cols + ['diagnosis']]
        corr_matrix = corr_data.corr()
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation matrix plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_names: list, importance_scores: list,
                               title: str = "Feature Importance",
                               save_path: Optional[str] = None) -> None:
        """
        Plot feature importance scores.
        
        Args:
            feature_names: List of feature names
            importance_scores: List of importance scores
            title: Plot title
            save_path: Optional path to save the plot
        """
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=True)
        
        plt.figure(figsize=(10, 6))
        
        # Create horizontal bar plot
        bars = plt.barh(importance_df['Feature'], importance_df['Importance'], 
                       color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, conf_matrix: np.ndarray, 
                            class_names: list = ['Benign', 'Malignant'],
                            title: str = "Confusion Matrix",
                            save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix heatmap.
        
        Args:
            conf_matrix: Confusion matrix array
            class_names: Names of the classes
            title: Plot title
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, Any]],
                            save_path: Optional[str] = None) -> None:
        """
        Plot comparison of different models.
        
        Args:
            results: Dictionary containing model results
            save_path: Optional path to save the plot
        """
        # Extract metrics for comparison
        models = list(results.keys())
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        
        # Create comparison DataFrame
        comparison_data = []
        for model in models:
            model_metrics = results[model]
            comparison_data.append({
                'Model': model.replace('_', ' ').title(),
                'Accuracy': model_metrics['accuracy'],
                'F1 Score': model_metrics['f1_score'],
                'Precision': model_metrics['precision'],
                'Recall': model_metrics['recall']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            metric_title = metric.replace('_', ' ').title()
            bars = axes[i].bar(comparison_df['Model'], comparison_df[metric_title], 
                              color=['lightblue', 'lightcoral'], alpha=0.7)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            
            axes[i].set_title(f'{metric_title} Comparison', fontweight='bold')
            axes[i].set_ylabel(metric_title)
            axes[i].set_ylim(0, 1.1)
            axes[i].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def create_comprehensive_eda(self, data: pd.DataFrame, feature_cols: list,
                               save_dir: Optional[str] = None) -> None:
        """
        Create comprehensive exploratory data analysis plots.
        
        Args:
            data: Input DataFrame
            feature_cols: List of feature columns
            save_dir: Optional directory to save plots
        """
        logger.info("Creating comprehensive EDA plots...")
        
        # Target distribution
        target_path = f"{save_dir}/target_distribution.png" if save_dir else None
        self.plot_target_distribution(data, save_path=target_path)
        
        # Feature distributions
        feature_path = f"{save_dir}/feature_distributions.png" if save_dir else None
        self.plot_feature_distributions(data, feature_cols, save_path=feature_path)
        
        # Correlation matrix
        corr_path = f"{save_dir}/correlation_matrix.png" if save_dir else None
        self.plot_correlation_matrix(data, feature_cols, save_path=corr_path)
        
        logger.info("EDA plots completed successfully")


def create_visualization_report(data: pd.DataFrame, results: Dict[str, Dict[str, Any]],
                              feature_cols: list, save_dir: str = "plots") -> None:
    """
    Create a comprehensive visualization report.
    
    Args:
        data: Input DataFrame
        results: Model evaluation results
        feature_cols: List of feature columns
        save_dir: Directory to save plots
    """
    import os
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    visualizer = DataVisualizer()
    
    # Create EDA plots
    visualizer.create_comprehensive_eda(data, feature_cols, save_dir)
    
    # Create model comparison plot
    comparison_path = f"{save_dir}/model_comparison.png"
    visualizer.plot_model_comparison(results, save_path=comparison_path)
    
    # Create confusion matrices for each model
    for model_name, model_results in results.items():
        conf_matrix = model_results['confusion_matrix']
        conf_path = f"{save_dir}/{model_name}_confusion_matrix.png"
        visualizer.plot_confusion_matrix(
            conf_matrix, 
            title=f"{model_name.replace('_', ' ').title()} Confusion Matrix",
            save_path=conf_path
        )
    
    logger.info(f"Visualization report saved to {save_dir}") 