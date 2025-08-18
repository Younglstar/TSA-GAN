import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

class DataComparator:
    def __init__(self):
        # Create output directory
        self.output_dir = 'data_comparisons'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load original data
        print("Loading original data...")
        self.original_df = pd.read_csv('Emergency_Cohort_with_Followup_500_subject.csv')
        
        # Load decoded synthetic data
        print("Loading decoded synthetic data...")
        with open('decoded_synthetic_data.pkl', 'rb') as f:
            self.decoded_data = pickle.load(f)
        
        # Load normalization parameters
        print("Loading normalization parameters...")
        with open('normalization_params.pkl', 'rb') as f:
            self.norm_params = pickle.load(f)
        
        # Load processed data to get feature information
        print("Loading processed data...")
        with open('processed_data.pkl', 'rb') as f:
            self.processed_data = pickle.load(f)
    
    def convert_to_dataframe(self) -> pd.DataFrame:
        """Convert synthetic data back to original format"""
        # Initialize empty DataFrame
        synthetic_df = pd.DataFrame(columns=self.original_df.columns)
        synthetic_df['DOSSIER_HASH'] = [f'SYNTH_{i:06d}' for i in range(len(self.decoded_data['static']))]
        
        # Load categorical embeddings
        print("\nLoading categorical embeddings...")
        with open('categorical_embeddings.pkl', 'rb') as f:
            cat_embeddings = pickle.load(f)
        
        # Process static features
        static_data = self.decoded_data['static']
        current_pos = 0
        
        print("\nMapping features...")
        for col in self.original_df.columns:
            if col == 'DOSSIER_HASH':
                continue
            
            print(f"\nProcessing column: {col}")
            if col in cat_embeddings['static_embeddings']:
                print(f"Processing categorical column: {col}")
                # Get embedding dimension for this feature
                embed_dim = cat_embeddings['static_embeddings'][col].shape[1]
                feature_data = static_data[:, current_pos:current_pos + embed_dim]
                
                # Get unique categories with their counts
                unique_cats = pd.Series(self.original_df[col].fillna('MISSING')).value_counts()
                print(f"Original unique categories: {len(unique_cats)}")
                print(f"Category distribution:\n{unique_cats}")
                
                # Map embeddings back to categories
                synthetic_df[col] = self.map_embeddings_to_categories(
                    feature_data,
                    cat_embeddings['static_embeddings'][col],
                    unique_cats.index.values
                )
                current_pos += embed_dim
                
            elif col in self.norm_params:
                print(f"Processing numerical column: {col}")
                synthetic_df[col] = self.inverse_normalize(
                    static_data[:, current_pos],
                    self.norm_params[col]
                )
                current_pos += 1
                
            else:
                print(f"Skipping column: {col} (no mapping information found)")
        
        return synthetic_df
    
    def inverse_normalize(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Inverse normalize data using stored parameters"""
        return data * params['scale'] + params['min']
    
    def map_embeddings_to_categories(
        self,
        embeddings: np.ndarray,
        original_embeddings: np.ndarray,
        categories: np.ndarray
    ) -> np.ndarray:
        """Map embeddings back to categorical values using nearest neighbor"""
        print(f"\nMapping embeddings for categories...")
        print(f"Input shapes:")
        print(f"- Embeddings to map: {embeddings.shape}")
        print(f"- Original embeddings: {original_embeddings.shape}")
        print(f"- Number of categories: {len(categories)}")
        
        # Normalize embeddings (to ensure distance calculations are meaningful)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        original_embeddings_norm = original_embeddings / np.linalg.norm(original_embeddings, axis=1)[:, np.newaxis]
        
        # Compute pairwise distances
        distances = np.dot(embeddings_norm, original_embeddings_norm.T)
        
        # Get indices of nearest neighbors
        indices = np.argmax(distances, axis=1)

        # Clip indices to valid range
        indices = np.clip(indices, 0, len(categories) - 1)
        
        # Map to categories
        result = categories[indices]
        
        print(f"Mapping complete. Result shape: {result.shape}")
        print(f"Unique categories in result: {np.unique(result)}")
        
        return result
    
    def compare_distributions(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame):
        """Create comparison plots for each column"""
        for col in original_df.columns:
            plt.figure(figsize=(12, 6))
            
            if pd.api.types.is_numeric_dtype(original_df[col]):
                # Numerical column
                plt.subplot(1, 2, 1)
                sns.histplot(data=original_df, x=col, label='Original', alpha=0.5)
                sns.histplot(data=synthetic_df, x=col, label='Synthetic', alpha=0.5)
                plt.legend()
                plt.title(f'Distribution of {col}')
                
                # Q-Q plot
                plt.subplot(1, 2, 2)
                stats.probplot(original_df[col].dropna(), dist="norm", plot=plt)
                plt.title(f'Q-Q Plot of {col}')
                
            else:
                # Categorical column
                orig_counts = original_df[col].value_counts(normalize=True)
                synth_counts = synthetic_df[col].value_counts(normalize=True)
                
                # Combine and fill missing categories
                all_categories = pd.Index(set(orig_counts.index) | set(synth_counts.index))
                orig_counts = orig_counts.reindex(all_categories, fill_value=0)
                synth_counts = synth_counts.reindex(all_categories, fill_value=0)
                
                # Plot
                plt.bar(np.arange(len(all_categories)) - 0.2, 
                       orig_counts, 0.4, label='Original', alpha=0.5)
                plt.bar(np.arange(len(all_categories)) + 0.2, 
                       synth_counts, 0.4, label='Synthetic', alpha=0.5)
                plt.xticks(range(len(all_categories)), all_categories, rotation=45)
                plt.title(f'Distribution of {col}')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{col}_comparison.png'))
            plt.close()
    
    def generate_summary_statistics(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
        """Generate detailed summary statistics for each column"""
        stats_list = []
        
        for col in original_df.columns:
            if col == 'DOSSIER_HASH':
                continue
                
            stats_dict = {'Column': col}
            
            if pd.api.types.is_numeric_dtype(original_df[col]):
                # Numerical column statistics
                orig_data = original_df[col].dropna()
                synth_data = synthetic_df[col].dropna()
                
                stats_dict.update({
                    'Type': 'Numerical',
                    'Original_Count': len(orig_data),
                    'Synthetic_Count': len(synth_data),
                    'Original_Mean': orig_data.mean(),
                    'Synthetic_Mean': synth_data.mean(),
                    'Original_Std': orig_data.std(),
                    'Synthetic_Std': synth_data.std(),
                    'Original_Min': orig_data.min(),
                    'Synthetic_Min': synth_data.min(),
                    'Original_Max': orig_data.max(),
                    'Synthetic_Max': synth_data.max(),
                    'Original_Missing_%': (original_df[col].isna().sum() / len(original_df)) * 100,
                    'Synthetic_Missing_%': (synthetic_df[col].isna().sum() / len(synthetic_df)) * 100
                })
                
                # Add KS test if enough data points
                if len(orig_data) > 0 and len(synth_data) > 0:
                    ks_stat, p_value = stats.ks_2samp(orig_data, synth_data)
                    stats_dict.update({
                        'KS_Statistic': ks_stat,
                        'KS_P_Value': p_value
                    })
                
            else:
                # Categorical column statistics
                orig_counts = original_df[col].value_counts(normalize=True)
                synth_counts = synthetic_df[col].value_counts(normalize=True)
                
                # Get all categories
                all_categories = sorted(set(orig_counts.index) | set(synth_counts.index))
                
                stats_dict.update({
                    'Type': 'Categorical',
                    'Original_Categories': len(orig_counts),
                    'Synthetic_Categories': len(synth_counts),
                    'Category_Match_%': (len(set(orig_counts.index) & set(synth_counts.index)) / 
                                      len(set(orig_counts.index) | set(synth_counts.index)) * 100),
                    'Original_Missing_%': (original_df[col].isna().sum() / len(original_df)) * 100,
                    'Synthetic_Missing_%': (synthetic_df[col].isna().sum() / len(synthetic_df)) * 100
                })
                
                # Add category-wise comparisons
                for cat in all_categories:
                    orig_pct = orig_counts.get(cat, 0) * 100
                    synth_pct = synth_counts.get(cat, 0) * 100
                    stats_dict[f'Original_%_{cat}'] = orig_pct
                    stats_dict[f'Synthetic_%_{cat}'] = synth_pct
            
            stats_list.append(stats_dict)
        
        return pd.DataFrame(stats_list)
    
    def generate_report(self):
        """Generate comprehensive comparison report"""
        print("Converting synthetic data to DataFrame format...")
        synthetic_df = self.convert_to_dataframe()
        
        print("Saving synthetic data to CSV...")
        synthetic_df.to_csv(os.path.join(self.output_dir, 'synthetic_data.csv'), index=False)
        
        print("Generating distribution comparisons...")
        self.compare_distributions(self.original_df, synthetic_df)
        
        print("Calculating summary statistics...")
        stats_df = self.generate_summary_statistics(self.original_df, synthetic_df)
        stats_df.to_csv(os.path.join(self.output_dir, 'column_statistics.csv'), index=False)
        
        # Generate detailed report
        with open(os.path.join(self.output_dir, 'summary_report.txt'), 'w') as f:
            f.write("Synthetic Data Generation Summary Report\n")
            f.write("=====================================\n\n")
            f.write(f"Number of original records: {len(self.original_df)}\n")
            f.write(f"Number of synthetic records: {len(synthetic_df)}\n\n")
            
            # Numerical columns summary
            num_cols = stats_df[stats_df['Type'] == 'Numerical']
            f.write("Numerical Columns Summary:\n")
            f.write("------------------------\n")
            for _, row in num_cols.iterrows():
                f.write(f"\nColumn: {row['Column']}\n")
                f.write(f"Mean: {row['Original_Mean']:.2f} (orig) vs {row['Synthetic_Mean']:.2f} (synth)\n")
                f.write(f"Std: {row['Original_Std']:.2f} (orig) vs {row['Synthetic_Std']:.2f} (synth)\n")
                if 'KS_P_Value' in row:
                    if 'KS_P_Value' in row:
                        f.write(f"KS test p-value: {row['KS_P_Value']:.4f}\n")
                        f.write(f"Missing values: {row['Original_Missing_%']:.1f}% (orig) vs {row['Synthetic_Missing_%']:.1f}% (synth)\n")
                        # Categorical columns summary
                        cat_cols = stats_df[stats_df['Type'] == 'Categorical']
                        f.write("\nCategorical Columns Summary:\n")
                        f.write("---------------------------\n")
                        for _, row in cat_cols.iterrows():
                            f.write(f"\nColumn: {row['Column']}\n")
                            f.write(f"Categories: {row['Original_Categories']} (orig) vs {row['Synthetic_Categories']} (synth)\n")
                            f.write(f"Category match: {row['Category_Match_%']:.1f}%\n")
                            f.write(f"Missing values: {row['Original_Missing_%']:.1f}% (orig) vs {row['Synthetic_Missing_%']:.1f}% (synth)\n")

def main():
    try:
        comparator = DataComparator()
        comparator.generate_report()
        print("\nComparison complete! Check the 'data_comparisons' directory for results.")
    except Exception as e:
        print(f"Error during execution: {e}")
        raise e

if __name__ == "__main__":
    main()