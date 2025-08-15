# SOURITRA SAMANTA (3C)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import os
from datetime import datetime

class HDBVisualizer:
    def __init__(self):
        self.output_dir = 'graphs'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set matplotlib style for better looking plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def generate_prediction_summary_visuals(self, model, inputs: Dict, prediction: float, contributions: Dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Feature Contribution Chart (Bar Chart)
        self._create_feature_contribution_chart(contributions, timestamp)
        
        # 2. Price Comparison Graph (Line Graph)
        self._create_price_comparison_graph(model, inputs, prediction, timestamp)
        
        # 3. Market Analysis Heatmap
        self._create_market_analysis_heatmap(model, inputs, timestamp)
        
        print(f"📊 Generated 3 visualizations in {self.output_dir}/ folder")
        return [
            f"{self.output_dir}/feature_contributions_{timestamp}.png",
            f"{self.output_dir}/price_comparison_{timestamp}.png", 
            f"{self.output_dir}/market_heatmap_{timestamp}.png"
        ]
    
    def _create_feature_contribution_chart(self, contributions: Dict, timestamp: str):
        plt.figure(figsize=(12, 8))
        
        # Prepare data for the chart
        features = list(contributions.keys())
        values = list(contributions.values())
        
        # Create horizontal bar chart
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(features)))
        bars = plt.barh(features, values, color=colors)
        
        # Customize the chart
        plt.title('Feature Contributions to HDB Price Prediction', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Contribution Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            plt.text(value + max(values) * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.1f}', ha='left', va='center', fontweight='bold')
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save the chart
        filename = f"{self.output_dir}/feature_contributions_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_price_comparison_graph(self, model, inputs: Dict, prediction: float, timestamp: str):
        plt.figure(figsize=(14, 8))
        
        # Get similar properties for comparison
        df = model.df
        similar_flats = df[
            (df['flat_type'] == inputs['flat_type']) & 
            (df['town'] == inputs['town'])
        ].copy()
        
        if len(similar_flats) > 0:
            # Create price range analysis
            area_ranges = np.arange(30, 201, 10)
            predicted_prices = []
            
            for area in area_ranges:
                try:
                    test_inputs = inputs.copy()
                    test_inputs['floor_area_sqm'] = area
                    pred_price, _ = model.predict_price(test_inputs)
                    predicted_prices.append(pred_price)
                except:
                    predicted_prices.append(np.nan)
            
            # Plot the prediction curve
            plt.plot(area_ranges, predicted_prices, 'b-', linewidth=3, label=f'Predicted Prices ({inputs["flat_type"]})', marker='o')
            
            # Plot actual market data points
            if len(similar_flats) > 0:
                plt.scatter(similar_flats['floor_area_sqm'], similar_flats['resale_price'], 
                           alpha=0.6, c='red', s=30, label=f'Actual Market Data ({inputs["town"]})')
            
            # Highlight the current prediction
            plt.scatter([inputs['floor_area_sqm']], [prediction], 
                       color='gold', s=200, marker='*', edgecolor='black', linewidth=2,
                       label=f'Your Prediction: ${prediction:,.0f}', zorder=5)
        
        plt.title(f'HDB Price Analysis: {inputs["flat_type"]} in {inputs["town"]}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Floor Area (sqm)', fontsize=12)
        plt.ylabel('Price (SGD)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Format y-axis to show currency
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        # Save the graph
        filename = f"{self.output_dir}/price_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_market_analysis_heatmap(self, model, inputs: Dict, timestamp: str):
        plt.figure(figsize=(16, 10))
        
        df = model.df
        
        # Create a price heatmap by town and flat type
        heatmap_data = df.groupby(['town', 'flat_type'])['resale_price'].median().unstack(fill_value=0)
        
        # Create the heatmap
        mask = heatmap_data == 0
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   fmt='.0f', 
                   cmap='YlOrRd', 
                   mask=mask,
                   cbar_kws={'label': 'Median Price (SGD)'},
                   annot_kws={'size': 8})
        
        # Highlight the user's selection
        try:
            town_idx = heatmap_data.index.get_loc(inputs['town'])
            flat_type_idx = heatmap_data.columns.get_loc(inputs['flat_type'])
            
            # Add a border around the selected cell
            plt.gca().add_patch(plt.Rectangle((flat_type_idx, town_idx), 1, 1, 
                                            fill=False, edgecolor='blue', lw=4))
        except (KeyError, ValueError):
            pass  # If the exact combination doesn't exist in the heatmap
        
        plt.title('HDB Market Analysis: Median Prices by Town and Flat Type', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Flat Type', fontsize=12)
        plt.ylabel('Town', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Add a note about the user's selection
        plt.figtext(0.02, 0.02, f"Your Selection: {inputs['flat_type']} in {inputs['town']} (highlighted in blue)", 
                   fontsize=10, style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        
        # Save the heatmap
        filename = f"{self.output_dir}/market_heatmap_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

# SOURITRA SAMANTA (3C)
