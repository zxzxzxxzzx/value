"""
HDB Data Visualization utilities with organized output
Handles creation of charts and graphs with automatic export to graphs folder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os
from datetime import datetime

# Set non-interactive backend to prevent display issues
plt.switch_backend('Agg')

class HDBVisualizer:
    """Handles HDB data visualization with organized output to graphs folder"""
    
    def __init__(self):
        self.graphs_dir = 'graphs'
        self.exports_dir = 'exports'
        self.setup_directories()
        self.setup_style()
    
    def setup_directories(self):
        """Create graphs and exports directories"""
        os.makedirs(self.graphs_dir, exist_ok=True)
        os.makedirs(self.exports_dir, exist_ok=True)
    
    def setup_style(self):
        """Set up matplotlib and seaborn styling"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def save_plot(self, filename: str, dpi: int = 300):
        """Save plot to graphs directory"""
        filepath = os.path.join(self.graphs_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"📊 Exported data visualization to {filepath}")
    
    def plot_price_distribution_by_town(self, df: pd.DataFrame, top_n: int = 15):
        """Plot price distribution by town"""
        if 'town' not in df.columns or 'resale_price' not in df.columns:
            print("❌ Missing required columns for town price distribution")
            return
        
        # Get top N towns by transaction count
        top_towns = df['town'].value_counts().head(top_n).index
        df_filtered = df[df['town'].isin(top_towns)]
        
        plt.figure(figsize=(15, 10))
        sns.boxplot(data=df_filtered, x='resale_price', y='town', order=top_towns)
        plt.title(f'HDB Resale Price Distribution by Town (Top {top_n} by Volume)')
        plt.xlabel('Resale Price (SGD)')
        plt.ylabel('Town')
        
        # Add average price annotations
        for i, town in enumerate(top_towns):
            avg_price = df_filtered[df_filtered['town'] == town]['resale_price'].mean()
            plt.text(avg_price, i, f'${avg_price:,.0f}', 
                    ha='center', va='center', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        self.save_plot('price_distribution_by_town.png')
    
    def plot_price_distribution_by_flat_type(self, df: pd.DataFrame):
        """Plot price distribution by flat type"""
        if 'flat_type' not in df.columns or 'resale_price' not in df.columns:
            print("❌ Missing required columns for flat type price distribution")
            return
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x='flat_type', y='resale_price')
        plt.title('HDB Resale Price Distribution by Flat Type')
        plt.xlabel('Flat Type')
        plt.ylabel('Resale Price (SGD)')
        plt.xticks(rotation=45)
        
        # Add average price annotations
        for i, flat_type in enumerate(df['flat_type'].unique()):
            avg_price = df[df['flat_type'] == flat_type]['resale_price'].mean()
            plt.text(i, avg_price, f'${avg_price:,.0f}', 
                    ha='center', va='bottom', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        self.save_plot('price_distribution_by_flat_type.png')
    
    def plot_price_vs_floor_area(self, df: pd.DataFrame, sample_size: int = 2000):
        """Plot price vs floor area with trend line"""
        if 'floor_area_sqm' not in df.columns or 'resale_price' not in df.columns:
            print("❌ Missing required columns for price vs floor area plot")
            return
        
        # Sample data for better performance
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df
        
        plt.figure(figsize=(12, 8))
        
        # Scatter plot with color by flat type if available
        if 'flat_type' in df_sample.columns:
            scatter = plt.scatter(df_sample['floor_area_sqm'], df_sample['resale_price'], 
                                c=pd.Categorical(df_sample['flat_type']).codes, 
                                alpha=0.6, cmap='viridis')
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Flat Type')
        else:
            plt.scatter(df_sample['floor_area_sqm'], df_sample['resale_price'], alpha=0.6)
        
        # Add polynomial trend line (4th degree to match our model)
        x = df_sample['floor_area_sqm']
        y = df_sample['resale_price']
        
        # Fit polynomial trend line
        z = np.polyfit(x, y, 4)
        p = np.poly1d(z)
        x_trend = np.linspace(x.min(), x.max(), 100)
        plt.plot(x_trend, p(x_trend), 'r-', linewidth=2, label='4th Degree Polynomial Trend')
        
        plt.xlabel('Floor Area (sqm)')
        plt.ylabel('Resale Price (SGD)')
        plt.title('HDB Resale Price vs Floor Area (with 4th Degree Polynomial Trend)')
        plt.legend()
        
        self.save_plot('price_vs_floor_area.png')
    
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """Plot correlation heatmap of numerical features"""
        numerical_cols = ['floor_area_sqm', 'remaining_lease', 'resale_price', 'lease_commence_date']
        available_cols = [col for col in numerical_cols if col in df.columns]
        
        if len(available_cols) < 2:
            print("❌ Insufficient numerical columns for correlation heatmap")
            return
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[available_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={'shrink': .8})
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        
        self.save_plot('correlation_heatmap.png')
    
    def plot_price_trends_over_time(self, df: pd.DataFrame):
        """Plot price trends over time"""
        if 'month' not in df.columns or 'resale_price' not in df.columns:
            print("❌ Missing required columns for price trends over time")
            return
        
        # Convert month to datetime
        df = df.copy()
        df['month'] = pd.to_datetime(df['month'])
        
        # Calculate monthly average prices
        monthly_prices = df.groupby('month')['resale_price'].agg(['mean', 'median', 'count']).reset_index()
        
        plt.figure(figsize=(14, 8))
        
        # Plot average and median prices
        plt.plot(monthly_prices['month'], monthly_prices['mean'], 
                'b-', linewidth=2, label='Average Price', marker='o')
        plt.plot(monthly_prices['month'], monthly_prices['median'], 
                'r-', linewidth=2, label='Median Price', marker='s')
        
        plt.xlabel('Month')
        plt.ylabel('Resale Price (SGD)')
        plt.title('HDB Resale Price Trends Over Time')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        self.save_plot('price_trends_over_time.png')
    
    def plot_model_performance(self, model_metrics: Dict, predictions_vs_actual: Optional[tuple] = None):
        """Plot model performance metrics and predictions vs actual"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: R² comparison
        r2_values = [model_metrics.get('train_r2', 0), model_metrics.get('test_r2', 0)]
        axes[0, 0].bar(['Training', 'Testing'], r2_values, color=['skyblue', 'orange'])
        axes[0, 0].set_title('R² Score Comparison')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(r2_values):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Plot 2: MAE and RMSE comparison
        mae_values = [model_metrics.get('train_mae', 0), model_metrics.get('test_mae', 0)]
        rmse_values = [model_metrics.get('train_rmse', 0), model_metrics.get('test_rmse', 0)]
        
        x = np.arange(2)
        width = 0.35
        axes[0, 1].bar(x - width/2, mae_values, width, label='MAE', color='lightblue')
        axes[0, 1].bar(x + width/2, rmse_values, width, label='RMSE', color='lightcoral')
        axes[0, 1].set_title('Error Metrics Comparison')
        axes[0, 1].set_ylabel('Error (SGD)')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(['Training', 'Testing'])
        axes[0, 1].legend()
        
        # Plot 3: Model Information
        axes[1, 0].axis('off')
        info_text = f"""
        Polynomial Degree: {model_metrics.get('polynomial_degree', 'N/A')}
        Number of Samples: {model_metrics.get('n_samples', 'N/A'):,}
        Number of Features: {model_metrics.get('n_features', 'N/A')}
        
        Training R²: {model_metrics.get('train_r2', 0):.4f}
        Testing R²: {model_metrics.get('test_r2', 0):.4f}
        
        Test MAE: ${model_metrics.get('test_mae', 0):,.2f}
        Test RMSE: ${model_metrics.get('test_rmse', 0):,.2f}
        """
        axes[1, 0].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        axes[1, 0].set_title('Model Information')
        
        # Plot 4: Predictions vs Actual (if data provided)
        if predictions_vs_actual:
            y_actual, y_pred = predictions_vs_actual
            axes[1, 1].scatter(y_actual, y_pred, alpha=0.6)
            
            # Add perfect prediction line
            min_val = min(y_actual.min(), y_pred.min())
            max_val = max(y_actual.max(), y_pred.max())
            axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            axes[1, 1].set_xlabel('Actual Price (SGD)')
            axes[1, 1].set_ylabel('Predicted Price (SGD)')
            axes[1, 1].set_title('Predictions vs Actual Prices')
        else:
            axes[1, 1].axis('off')
            axes[1, 1].text(0.5, 0.5, 'Predictions vs Actual\n(No data provided)', 
                           ha='center', va='center', fontsize=12,
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        self.save_plot('model_performance.png')
    
    def create_comprehensive_dashboard(self, df: pd.DataFrame, model_metrics: Dict):
        """Create a comprehensive visualization dashboard"""
        print("🎨 Creating comprehensive visualization dashboard...")
        
        # Create individual visualizations
        self.plot_price_distribution_by_town(df)
        self.plot_price_distribution_by_flat_type(df)
        self.plot_price_vs_floor_area(df)
        self.plot_correlation_heatmap(df)
        self.plot_price_trends_over_time(df)
        self.plot_model_performance(model_metrics)
        
        print(f"✅ Dashboard created with 6 visualizations in {self.graphs_dir}/ folder")
    
    def export_data_summary(self, df: pd.DataFrame, filename: str = 'data_summary.csv'):
        """Export data summary to exports folder"""
        filepath = os.path.join(self.exports_dir, filename)
        
        # Create summary statistics
        summary_stats = df.describe(include='all')
        summary_stats.to_csv(filepath)
        
        print(f"📁 Exported data summary to {filepath}")
    
    def create_town_price_distribution(self, df: pd.DataFrame, town: str) -> bool:
        """Create price distribution chart for a specific town"""
        try:
            town_data = df[df['town'] == town].copy()
            if town_data.empty:
                return False
            
            plt.figure(figsize=(12, 8))
            
            # Create subplot layout
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Price distribution histogram
            ax1.hist(town_data['resale_price'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title(f'Price Distribution in {town}')
            ax1.set_xlabel('Resale Price (SGD)')
            ax1.set_ylabel('Frequency')
            ax1.axvline(town_data['resale_price'].mean(), color='red', linestyle='--', 
                       label=f'Mean: ${town_data["resale_price"].mean():,.0f}')
            ax1.legend()
            
            # Price by flat type in this town
            if 'flat_type' in town_data.columns:
                town_data.boxplot(column='resale_price', by='flat_type', ax=ax2)
                ax2.set_title(f'Price by Flat Type in {town}')
                ax2.set_xlabel('Flat Type')
                ax2.set_ylabel('Resale Price (SGD)')
            
            plt.suptitle('')
            self.save_plot(f'town_analysis_{town.lower().replace(" ", "_")}.png')
            return True
        except Exception:
            return False
    
    def create_flat_type_analysis(self, df: pd.DataFrame, flat_type: str) -> bool:
        """Create analysis for a specific flat type"""
        try:
            flat_data = df[df['flat_type'] == flat_type].copy()
            if flat_data.empty:
                return False
            
            plt.figure(figsize=(15, 10))
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Price distribution
            ax1.hist(flat_data['resale_price'], bins=25, alpha=0.7, color='lightgreen', edgecolor='black')
            ax1.set_title(f'{flat_type} Price Distribution')
            ax1.set_xlabel('Resale Price (SGD)')
            ax1.set_ylabel('Frequency')
            
            # 2. Price vs floor area
            if 'floor_area_sqm' in flat_data.columns:
                ax2.scatter(flat_data['floor_area_sqm'], flat_data['resale_price'], alpha=0.5)
                ax2.set_title(f'{flat_type}: Price vs Floor Area')
                ax2.set_xlabel('Floor Area (sqm)')
                ax2.set_ylabel('Resale Price (SGD)')
            
            # 3. Top towns for this flat type
            if 'town' in flat_data.columns:
                top_towns = flat_data.groupby('town')['resale_price'].mean().sort_values(ascending=False).head(10)
                ax3.bar(range(len(top_towns)), top_towns.values, color='coral')
                ax3.set_title(f'Top 10 Towns for {flat_type} (Avg Price)')
                ax3.set_xlabel('Towns')
                ax3.set_ylabel('Average Price (SGD)')
                ax3.set_xticks(range(len(top_towns)))
                ax3.set_xticklabels(top_towns.index, rotation=45, ha='right')
            
            # 4. Price by remaining lease
            if 'remaining_lease' in flat_data.columns:
                ax4.scatter(flat_data['remaining_lease'], flat_data['resale_price'], alpha=0.5, color='purple')
                ax4.set_title(f'{flat_type}: Price vs Remaining Lease')
                ax4.set_xlabel('Remaining Lease (years)')
                ax4.set_ylabel('Resale Price (SGD)')
            
            plt.tight_layout()
            self.save_plot(f'flat_type_analysis_{flat_type.lower().replace(" ", "_")}.png')
            return True
        except Exception:
            return False
    
    def create_price_vs_area_chart(self, df: pd.DataFrame, user_inputs: Dict) -> bool:
        """Create price vs area scatter plot with user's selection highlighted"""
        try:
            if 'floor_area_sqm' not in df.columns:
                return False
            
            plt.figure(figsize=(12, 8))
            
            # Filter similar properties
            similar_data = df.copy()
            if 'town' in user_inputs and 'town' in df.columns:
                similar_data = similar_data[similar_data['town'] == user_inputs['town']]
            
            # Scatter plot
            plt.scatter(df['floor_area_sqm'], df['resale_price'], alpha=0.3, color='lightblue', label='All Properties')
            
            if not similar_data.empty and len(similar_data) != len(df):
                plt.scatter(similar_data['floor_area_sqm'], similar_data['resale_price'], 
                           alpha=0.6, color='orange', label=f'Properties in {user_inputs.get("town", "Selected Area")}')
            
            # Highlight user's input
            user_area = user_inputs.get('floor_area_sqm', 0)
            if user_area > 0:
                plt.axvline(user_area, color='red', linestyle='--', linewidth=2, label=f'Your Selection: {user_area} sqm')
            
            plt.xlabel('Floor Area (sqm)')
            plt.ylabel('Resale Price (SGD)')
            plt.title('HDB Price vs Floor Area Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            self.save_plot('price_vs_floor_area.png')
            return True
        except Exception:
            return False
    
    def create_market_heatmap(self, df: pd.DataFrame, town: str, flat_type: str) -> bool:
        """Create market comparison heatmap"""
        try:
            # Create pivot table for heatmap
            if 'town' not in df.columns or 'flat_type' not in df.columns:
                return False
            
            pivot_data = df.groupby(['town', 'flat_type'])['resale_price'].mean().reset_index()
            heatmap_data = pivot_data.pivot(index='town', columns='flat_type', values='resale_price')
            
            plt.figure(figsize=(14, 10))
            
            # Create heatmap
            sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                       cbar_kws={'label': 'Average Resale Price (SGD)'})
            
            plt.title('HDB Average Prices: Towns vs Flat Types')
            plt.xlabel('Flat Type')
            plt.ylabel('Town')
            
            # Highlight user's selection if available
            if town in heatmap_data.index and flat_type in heatmap_data.columns:
                town_idx = list(heatmap_data.index).index(town)
                flat_idx = list(heatmap_data.columns).index(flat_type)
                plt.gca().add_patch(plt.Rectangle((flat_idx, town_idx), 1, 1, 
                                                fill=False, edgecolor='blue', lw=3))
            
            plt.tight_layout()
            self.save_plot('market_heatmap.png')
            return True
        except Exception:
            return False
