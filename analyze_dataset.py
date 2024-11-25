import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_bci_iv_2a.dataset import BciIvCsvParser
import numpy as np
from typing import Dict, List

class DatasetAnalyzer:
    def __init__(self):
        self.class_names = {
            0: "Rest",
            1: "Left Hand",
            2: "Right Hand",
            3: "Feet",
            4: "Tongue"
        }
        
    def load_subject_data(self, subject_number: int) -> pd.DataFrame:
        """Load and combine both training and evaluation data for a subject"""
        subject_str = f"0{subject_number}" if subject_number < 10 else str(subject_number)
        
        # Load evaluation and training data
        evaluation_data = BciIvCsvParser(f"dataset_bci_iv_2a/A{subject_str}E.csv").get_dataframe()
        training_data = BciIvCsvParser(f"dataset_bci_iv_2a/A{subject_str}T.csv").get_dataframe()
        
        # Adjust recording numbers for training data to avoid overlap
        recording_offset = evaluation_data["Recording"].max() + 1
        training_data["Recording"] += recording_offset
        
        # Combine datasets
        return pd.concat([evaluation_data, training_data])

    def analyze_class_distribution(self, df: pd.DataFrame, subject_number: int) -> Dict:
        """Analyze class distribution for a single subject"""
        label_counts = df['Label'].value_counts().sort_index()
        total_samples = len(df)
        
        # Calculate percentages
        distribution_stats = {
            'counts': label_counts,
            'percentages': (label_counts / total_samples * 100).round(2),
            'total': total_samples
        }
        
        # Print distribution
        print(f"\nClass Distribution for Subject {subject_number}:")
        print("-" * 50)
        for label, count in label_counts.items():
            class_name = self.class_names.get(label, f"Class {label}")
            percentage = distribution_stats['percentages'][label]
            print(f"{class_name}: {count} samples ({percentage}%)")
        print("-" * 50)
        print(f"Total samples: {total_samples}")
        
        return distribution_stats

    def plot_class_distribution(self, stats: Dict, subject_number: int, save_path: str = None):
        """Create a bar plot of class distribution"""
        plt.figure(figsize=(12, 6))
        
        # Create bar plot
        bars = plt.bar(
            [self.class_names[i] for i in stats['counts'].index],
            stats['counts'].values
        )
        
        # Customize plot
        plt.title(f'Class Distribution in BCI2a Dataset - Subject {subject_number}')
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{int(height)}\n({height/stats["total"]*100:.1f}%)',
                ha='center',
                va='bottom'
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def analyze_all_subjects(self, num_subjects: int = 9):
        """Analyze class distribution for all subjects"""
        all_stats = {}
        
        # Create a figure for all subjects
        fig, axes = plt.subplots(3, 3, figsize=(20, 20))
        fig.suptitle('Class Distribution Across All Subjects', fontsize=16)
        
        for subject in range(1, num_subjects + 1):
            try:
                # Load and analyze data
                df = self.load_subject_data(subject)
                stats = self.analyze_class_distribution(df, subject)
                all_stats[subject] = stats
                
                # Plot in subplot
                ax = axes[(subject-1)//3, (subject-1)%3]
                ax.bar(
                    [self.class_names[i] for i in stats['counts'].index],
                    stats['counts'].values
                )
                ax.set_title(f'Subject {subject}')
                ax.set_xticklabels([self.class_names[i] for i in stats['counts'].index], rotation=45)
                
                # Add percentage labels
                for i, v in enumerate(stats['counts'].values):
                    ax.text(
                        i,
                        v,
                        f'{v}\n({v/stats["total"]*100:.1f}%)',
                        ha='center',
                        va='bottom'
                    )
            
            except Exception as e:
                print(f"Error processing subject {subject}: {str(e)}")
        
        plt.tight_layout()
        plt.savefig('all_subjects_distribution.png')
        plt.close()
        
        return all_stats

    def plot_heatmap(self, all_stats: Dict):
        """Create a heatmap of class distributions across all subjects"""
        # Prepare data for heatmap
        data = []
        for subject in all_stats:
            percentages = all_stats[subject]['percentages']
            data.append(percentages)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            data,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            xticklabels=[self.class_names[i] for i in range(5)],
            yticklabels=[f'Subject {i}' for i in range(1, len(all_stats) + 1)]
        )
        plt.title('Class Distribution Heatmap (Percentages)')
        plt.tight_layout()
        plt.savefig('class_distribution_heatmap.png')
        plt.close()

    def analyze_single_subject(self, subject_number: int):
        """Detailed analysis of a single subject's data"""
        print(f"\nAnalyzing Subject {subject_number}...")
        
        # Load evaluation and training data separately
        subject_str = f"0{subject_number}" if subject_number < 10 else str(subject_number)
        evaluation_data = BciIvCsvParser(f"dataset_bci_iv_2a/A{subject_str}E.csv").get_dataframe()
        training_data = BciIvCsvParser(f"dataset_bci_iv_2a/A{subject_str}T.csv").get_dataframe()
        
        # Analyze each set separately
        print("\nEvaluation Set (E):")
        eval_stats = self.analyze_class_distribution(evaluation_data, f"{subject_number}E")
        
        print("\nTraining Set (T):")
        train_stats = self.analyze_class_distribution(training_data, f"{subject_number}T")
        
        # Plot separate distributions
        self.plot_class_distribution(eval_stats, f"{subject_number}E", f'subject_{subject_number}_eval_distribution.png')
        self.plot_class_distribution(train_stats, f"{subject_number}T", f'subject_{subject_number}_train_distribution.png')
        
        # Combined analysis
        combined_data = pd.concat([evaluation_data, training_data])
        print("\nCombined (E+T):")
        combined_stats = self.analyze_class_distribution(combined_data, f"{subject_number}_combined")
        self.plot_class_distribution(combined_stats, f"{subject_number}_combined", f'subject_{subject_number}_combined_distribution.png')
        
        return {
            'evaluation': eval_stats,
            'training': train_stats,
            'combined': combined_stats
        }

if __name__ == "__main__":
    analyzer = DatasetAnalyzer()
    
    # Analyze only subject 1 (which is what your model uses)
    subject_stats = analyzer.analyze_single_subject(1)
    
    print("\nAnalysis complete! Check the generated PNG files for visualizations.") 