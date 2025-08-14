# compare_data.py
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import ks_2samp
from typing import Dict, Tuple, List

class DataComparator:
    def __init__(self):
        # Load all necessary data
        with open('synthetic_data.pkl', 'rb') as f:
            self.synthetic_encoded = pickle.load(f)
            
        with open('encoded_data.pkl', 'rb') as f:
            self.real_encoded = pickle.load(f)
            
        with open('normalized_data.pkl', 'rb') as f:
            self.normalized_data = pickle.load(f)
            
        # Store dimensions
        self.n_real = len(self.real_encoded)
        self.n_synthetic = len(self.synthetic_encoded)
        
    def plot_distribution_comparison(self, save_path: str = 'distribution_comparison.png'):
        """Compare distributions of encoded features"""
        n_features = min(10, self.real_encoded.shape[1])  # Plot first 10 features
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.ravel()
        
        for i in range(n_features):
            # Real data distribution
            sns.kdeplot(
                data=self.real_encoded[:, i],
                ax=axes[i],
                label='Real',
                color='blue',
                alpha=0.5
            )
            
            # Synthetic data distribution
            sns.kdeplot(
                data=self.synthetic_encoded[:, i],
                ax=axes[i],
                label='Synthetic',
                color='red',
                alpha=0.5
            )
            
            axes[i].set_title(f'Feature {i+1}')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def compute_statistical_tests(self) -> pd.DataFrame:
        """Compute statistical tests between real and synthetic data"""
        results = []
        
        for i in range(self.real_encoded.shape[1]):
            # Perform Kolmogorov-Smirnov test
            ks_stat, p_value = ks_2samp(
                self.real_encoded[:, i],
                self.synthetic_encoded[:, i]
            )
            
            # Compute mean and std differences
            mean_diff = np.abs(
                np.mean(self.real_encoded[:, i]) - 
                np.mean(self.synthetic_encoded[:, i])
            )
            std_diff = np.abs(
                np.std(self.real_encoded[:, i]) - 
                np.std(self.synthetic_encoded[:, i])
            )
            
            results.append({
                'Feature': f'Feature_{i}',
                'KS_Statistic': ks_stat,
                'P_Value': p_value,
                'Mean_Difference': mean_diff,
                'Std_Difference': std_diff
            })
        
        return pd.DataFrame(results)
    
    def plot_dimensionality_reduction(self, save_path: str = 'dim_reduction.png'):
        """Compare real and synthetic data in reduced dimensional space"""
        # Combine data for fitting
        combined_data = np.vstack([self.real_encoded, self.synthetic_encoded])
        
        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined_data)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(combined_data)
        
        # Create labels
        labels = ['Real'] * self.n_real + ['Synthetic'] * self.n_synthetic
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # PCA plot
        sns.scatterplot(
            x=pca_result[:self.n_real, 0],
            y=pca_result[:self.n_real, 1],
            label='Real',
            alpha=0.5,
            ax=ax1
        )
        sns.scatterplot(
            x=pca_result[self.n_real:, 0],
            y=pca_result[self.n_real:, 1],
            label='Synthetic',
            alpha=0.5,
            ax=ax1
        )
        ax1.set_title('PCA Visualization')
        
        # t-SNE plot
        sns.scatterplot(
            x=tsne_result[:self.n_real, 0],
            y=tsne_result[:self.n_real, 1],
            label='Real',
            alpha=0.5,
            ax=ax2
        )
        sns.scatterplot(
            x=tsne_result[self.n_real:, 0],
            y=tsne_result[self.n_real:, 1],
            label='Synthetic',
            alpha=0.5,
            ax=ax2
        )
        ax2.set_title('t-SNE Visualization')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def compute_correlation_difference(self, save_path: str = 'correlation_diff.png'):
        """Compare correlation matrices between real and synthetic data"""
        # Compute correlation matrices
        real_corr = np.corrcoef(self.real_encoded.T)
        synthetic_corr = np.corrcoef(self.synthetic_encoded.T)
        
        # Compute difference
        correlation_diff = np.abs(real_corr - synthetic_corr)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_diff,
            cmap='coolwarm',
            center=0,
            vmin=0,
            vmax=1
        )
        plt.title('Correlation Matrix Difference\n(Real - Synthetic)')
        plt.savefig(save_path)
        plt.close()
        
        return np.mean(correlation_diff)
    
    def print_summary_statistics(self):
        """Print summary statistics for both datasets"""
        print("\nSummary Statistics:")
        print("\nReal Data:")
        print(pd.DataFrame(self.real_encoded).describe())
        print("\nSynthetic Data:")
        print(pd.DataFrame(self.synthetic_encoded).describe())

def main():
    print("Starting data comparison analysis...")
    
    # Initialize comparator
    comparator = DataComparator()
    
    # Generate visualizations
    print("\nGenerating distribution comparisons...")
    comparator.plot_distribution_comparison()
    
    print("\nGenerating dimensionality reduction visualizations...")
    comparator.plot_dimensionality_reduction()
    
    print("\nComputing correlation differences...")
    mean_corr_diff = comparator.compute_correlation_difference()
    print(f"Mean correlation difference: {mean_corr_diff:.4f}")
    
    # Compute statistical tests
    print("\nComputing statistical tests...")
    test_results = comparator.compute_statistical_tests()
    print("\nStatistical Test Results:")
    print(test_results)
    
    # Print summary statistics
    comparator.print_summary_statistics()
    
    print("\nAnalysis complete! Visualization files have been saved.")

if __name__ == "__main__":
    main()