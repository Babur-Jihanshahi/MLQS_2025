import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def load_combined_data(filepath="all_modes_raw_long.csv"):
    """Load the combined raw data"""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")
    print(f"Modes found: {df['mode'].unique()}")
    return df

def select_features_for_plotting(df):
    """Select numeric features to plot, excluding time and metadata columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude time columns and other metadata
    exclude_patterns = ['time', 't_', 'index', 'timestamp']
    features = [col for col in numeric_cols 
                if not any(pattern in col.lower() for pattern in exclude_patterns)]
    
    return features

def plot_distributions_by_mode(df, features=None, max_features=12):
    """Create distribution plots for each mode"""
    
    if features is None:
        features = select_features_for_plotting(df)
    
    # Limit number of features for readability
    if len(features) > max_features:
        print(f"Limiting to first {max_features} features. Total available: {len(features)}")
        features = features[:max_features]
    
    modes = df['mode'].unique()
    n_modes = len(modes)
    n_features = len(features)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_features, n_modes, figsize=(5*n_modes, 4*n_features))
    
    # Handle single feature/mode case
    if n_features == 1:
        axes = axes.reshape(1, -1)
    if n_modes == 1:
        axes = axes.reshape(-1, 1)
    
    # Color palette for modes
    colors = sns.color_palette("husl", n_modes)
    
    for i, feature in enumerate(features):
        for j, mode in enumerate(modes):
            ax = axes[i, j]
            
            # Get data for this mode and feature
            mode_data = df[df['mode'] == mode][feature].dropna()
            
            if len(mode_data) > 0:
                # Create histogram with KDE
                sns.histplot(data=mode_data, kde=True, bins=30, 
                           color=colors[j], alpha=0.7, ax=ax)
                
                # Add statistics
                mean_val = mode_data.mean()
                median_val = mode_data.median()
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_val:.2f}')
                
                # Formatting
                ax.set_title(f'{feature} - {mode.capitalize()}', fontsize=10)
                ax.set_xlabel('')
                ax.set_ylabel('Frequency' if j == 0 else '')
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{feature} - {mode.capitalize()}', fontsize=10)
    
    plt.suptitle('Feature Distributions by Transport Mode', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_file = "distributions_by_mode.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.show()

def plot_overlayed_distributions(df, features=None, max_features=9):
    """Create overlayed distribution plots comparing all modes"""
    
    if features is None:
        features = select_features_for_plotting(df)
    
    # Limit features
    if len(features) > max_features:
        features = features[:max_features]
    
    # Calculate subplot grid
    n_features = len(features)
    cols = 3
    rows = int(np.ceil(n_features / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    modes = df['mode'].unique()
    colors = sns.color_palette("husl", len(modes))
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        # Plot each mode
        for j, mode in enumerate(modes):
            mode_data = df[df['mode'] == mode][feature].dropna()
            
            if len(mode_data) > 0:
                # Create KDE plot for overlay
                sns.kdeplot(data=mode_data, color=colors[j], 
                          label=mode.capitalize(), ax=ax, alpha=0.7, linewidth=2)
        
        ax.set_title(f'Distribution of {feature}', fontsize=12)
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Feature Distributions - All Modes Overlayed', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    output_file = "distributions_overlayed.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.show()

def plot_boxplots_comparison(df, features=None, max_features=9):
    """Create boxplots comparing distributions across modes"""
    
    if features is None:
        features = select_features_for_plotting(df)
    
    # Select most interesting features based on variance across modes
    if len(features) > max_features:
        # Calculate coefficient of variation for each feature across modes
        feature_scores = []
        for feature in features:
            mode_means = df.groupby('mode')[feature].mean()
            if mode_means.std() > 0:
                cv = mode_means.std() / mode_means.mean()
                feature_scores.append((feature, cv))
        
        # Sort by coefficient of variation and take top features
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        features = [f[0] for f in feature_scores[:max_features]]
        print(f"Selected {max_features} features with highest variance across modes")
    
    # Create subplots
    cols = 3
    rows = int(np.ceil(len(features) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    axes = axes.flatten() if len(features) > 1 else [axes]
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        # Prepare data for boxplot
        plot_data = []
        labels = []
        
        for mode in df['mode'].unique():
            mode_data = df[df['mode'] == mode][feature].dropna()
            if len(mode_data) > 0:
                plot_data.append(mode_data)
                labels.append(mode.capitalize())
        
        if plot_data:
            # Create boxplot
            bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
            
            # Color boxes
            colors = sns.color_palette("husl", len(labels))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(f'{feature}', fontsize=12)
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            # Rotate labels if needed
            if len(labels) > 3:
                ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Hide empty subplots
    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Feature Comparison Across Transport Modes (Boxplots)', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    output_file = "distributions_boxplots.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.show()

def plot_sensor_specific_distributions(df):
    """Create separate distribution plots for GPS and Gyro features"""
    
    # Separate GPS and Gyro features
    gps_features = [col for col in df.select_dtypes(include=[np.number]).columns 
                    if 'gyro_' not in col and col not in ['time_s', 'time', 't']]
    gyro_features = [col for col in df.select_dtypes(include=[np.number]).columns 
                     if 'gyro_' in col]
    
    # Plot GPS features
    if gps_features:
        print("\nPlotting GPS features...")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(gps_features[:6]):  # Limit to 6 features
            ax = axes[i]
            
            for mode in df['mode'].unique():
                mode_data = df[df['mode'] == mode][feature].dropna()
                if len(mode_data) > 0:
                    sns.kdeplot(data=mode_data, label=mode.capitalize(), ax=ax, alpha=0.7)
            
            ax.set_title(f'{feature}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(gps_features[:6]), 6):
            axes[i].set_visible(False)
            
        plt.suptitle('GPS Feature Distributions by Mode', fontsize=16)
        plt.tight_layout()
        plt.savefig("distributions_gps_features.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    # Plot Gyro features
    if gyro_features:
        print("\nPlotting Gyroscope features...")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        gyro_xyz = ['gyro_x', 'gyro_y', 'gyro_z']
        
        for i, feature in enumerate(gyro_xyz):
            if feature in gyro_features:
                ax = axes[i]
                
                # Only plot for walking mode (as it has gyro data)
                walking_data = df[df['mode'] == 'walking'][feature].dropna()
                if len(walking_data) > 0:
                    sns.histplot(data=walking_data, kde=True, bins=50, 
                               color='blue', alpha=0.7, ax=ax)
                    
                    # Add statistics
                    mean_val = walking_data.mean()
                    std_val = walking_data.std()
                    ax.axvline(mean_val, color='red', linestyle='--', 
                             label=f'Mean: {mean_val:.3f}')
                    ax.axvspan(mean_val - std_val, mean_val + std_val, 
                             alpha=0.2, color='red', label=f'±1 STD')
                    
                ax.set_title(f'{feature} (Walking)', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Gyroscope Feature Distributions', fontsize=16)
        plt.tight_layout()
        plt.savefig("distributions_gyro_features.png", dpi=150, bbox_inches='tight')
        plt.show()

def create_summary_statistics_plot(df):
    """Create a visual summary of statistics across modes"""
    
    features = select_features_for_plotting(df)[:6]  # Top 6 features
    modes = df['mode'].unique()
    
    # Calculate statistics
    stats_data = []
    for mode in modes:
        for feature in features:
            mode_data = df[df['mode'] == mode][feature].dropna()
            if len(mode_data) > 0:
                stats_data.append({
                    'Mode': mode.capitalize(),
                    'Feature': feature,
                    'Mean': mode_data.mean(),
                    'Std': mode_data.std(),
                    'Count': len(mode_data)
                })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Create heatmap of means
    if not stats_df.empty:
        pivot_mean = stats_df.pivot(index='Feature', columns='Mode', values='Mean')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_mean, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, cbar_kws={'label': 'Mean Value'})
        plt.title('Mean Values Heatmap Across Modes and Features', fontsize=14)
        plt.tight_layout()
        plt.savefig("distributions_heatmap.png", dpi=150, bbox_inches='tight')
        plt.show()

# Main execution
if __name__ == "__main__":
    # Load data
    df = load_combined_data()
    
    # Get list of features
    features = select_features_for_plotting(df)
    print(f"\nFound {len(features)} plottable features:")
    print(features[:10], "..." if len(features) > 10 else "")
    
    # Choose which features to plot (you can customize this)
    # For GPS data
    gps_features = ['latitude', 'longitude', 'altitude', 'speed', 
                    'horizontal_accuracy', 'vertical_accuracy', 'distance']
    # For Gyro data  
    gyro_features = ['gyro_x', 'gyro_y', 'gyro_z']
    
    # Filter to available features
    selected_features = [f for f in gps_features + gyro_features if f in features]
    
    print(f"\nCreating distribution plots for: {selected_features}")
    
    # Create different types of plots
    print("\n1. Creating individual distribution plots by mode...")
    plot_distributions_by_mode(df, selected_features)
    
    print("\n2. Creating overlayed distribution plots...")
    plot_overlayed_distributions(df, selected_features)
    
    print("\n3. Creating boxplot comparisons...")
    plot_boxplots_comparison(df, selected_features)
    
    print("\nAll plots created successfully!")