# SOURITRA SAMANTA (3C)

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class HDBDataProcessor:
    """Handles HDB data processing and feature engineering for polynomial regression"""
    
    def __init__(self):
        self.processed_data = None
        self.feature_stats = {}
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess HDB data"""
        cleaned_df = df.copy()
        
        # Remove duplicates
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        duplicates_removed = initial_count - len(cleaned_df)
        if duplicates_removed > 0:
            print(f"🧹 Removed {duplicates_removed} duplicate records")
        
        # Handle missing values
        missing_before = cleaned_df.isnull().sum().sum()
        cleaned_df = cleaned_df.dropna()
        missing_after = missing_before - cleaned_df.isnull().sum().sum()
        if missing_after > 0:
            print(f"🧹 Removed {missing_after} records with missing values")
        
        # Clean price data (remove extreme outliers for polynomial regression)
        if 'resale_price' in cleaned_df.columns:
            price_col = cleaned_df['resale_price']
            Q1 = price_col.quantile(0.005)  # More conservative for polynomial
            Q3 = price_col.quantile(0.995)
            
            outliers_mask = (price_col < Q1) | (price_col > Q3)
            outliers_count = outliers_mask.sum()
            
            cleaned_df = cleaned_df[~outliers_mask]
            if outliers_count > 0:
                print(f"🧹 Removed {outliers_count} price outliers for polynomial regression")
        
        # Clean floor area data
        if 'floor_area_sqm' in cleaned_df.columns:
            area_mask = (cleaned_df['floor_area_sqm'] >= 30) & (cleaned_df['floor_area_sqm'] <= 250)
            area_outliers = len(cleaned_df) - area_mask.sum()
            cleaned_df = cleaned_df[area_mask]
            if area_outliers > 0:
                print(f"🧹 Removed {area_outliers} unrealistic floor areas")
        
        # Standardize categorical data (including flat_model for accuracy)
        categorical_columns = ['town', 'flat_type', 'storey_range', 'flat_model']
        for col in categorical_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].str.upper().str.strip()
        
        print(f"✅ Data cleaning completed. Final dataset: {len(cleaned_df)} records")
        self.processed_data = cleaned_df
        return cleaned_df
    
    def generate_feature_stats(self, df: pd.DataFrame) -> Dict:
        """Generate statistical summary of features"""
        stats = {}
        
        # Numerical features
        numerical_cols = ['floor_area_sqm', 'remaining_lease', 'resale_price']
        for col in numerical_cols:
            if col in df.columns:
                stats[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'q25': df[col].quantile(0.25),
                    'q75': df[col].quantile(0.75),
                    'skewness': df[col].skew(),
                    'kurtosis': df[col].kurtosis()
                }
        
        # Categorical features (including flat_model for accuracy)
        categorical_cols = ['town', 'flat_type', 'storey_range', 'flat_model']
        for col in categorical_cols:
            if col in df.columns:
                stats[col] = {
                    'unique_count': df[col].nunique(),
                    'most_common': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                    'value_counts': df[col].value_counts().head(10).to_dict()
                }
        
        self.feature_stats = stats
        return stats
    
    def get_price_analysis_by_town(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze price statistics by town"""
        if 'town' not in df.columns or 'resale_price' not in df.columns:
            return pd.DataFrame()
        
        town_analysis = df.groupby('town')['resale_price'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        
        town_analysis.columns = ['Count', 'Mean_Price', 'Median_Price', 
                               'Std_Dev', 'Min_Price', 'Max_Price']
        
        return town_analysis.sort_values('Mean_Price', ascending=False)
    
    def get_price_analysis_by_flat_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze price statistics by flat type"""
        if 'flat_type' not in df.columns or 'resale_price' not in df.columns:
            return pd.DataFrame()
        
        flat_type_analysis = df.groupby('flat_type')['resale_price'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        
        flat_type_analysis.columns = ['Count', 'Mean_Price', 'Median_Price', 
                                    'Std_Dev', 'Min_Price', 'Max_Price']
        
        return flat_type_analysis.sort_values('Mean_Price', ascending=False)
    
    def create_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create correlation matrix for numerical features"""
        numerical_cols = ['floor_area_sqm', 'remaining_lease', 'resale_price']
        available_cols = [col for col in numerical_cols if col in df.columns]
        
        if len(available_cols) < 2:
            return pd.DataFrame()
        
        return df[available_cols].corr().round(3)
    
    def validate_input_data(self, inputs: Dict) -> Tuple[bool, List[str]]:
        """Validate user input data for polynomial regression"""
        errors = []
        
        # Check required fields (flat_model restored with limited options)
        required_fields = ['town', 'flat_type', 'floor_area_sqm', 
                          'storey_range', 'flat_model', 'remaining_lease']
        
        for field in required_fields:
            if field not in inputs or inputs[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate numerical ranges (stricter for polynomial regression)
        if 'floor_area_sqm' in inputs:
            area = inputs['floor_area_sqm']
            if not isinstance(area, (int, float)) or area < 30 or area > 250:
                errors.append("Floor area must be between 30 and 250 sqm")
        
        if 'remaining_lease' in inputs:
            lease = inputs['remaining_lease']
            if not isinstance(lease, int) or lease < 40 or lease > 99:
                errors.append("Remaining lease must be between 40 and 99 years")
        
        return len(errors) == 0, errors

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive data summary for polynomial regression analysis"""
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df['month'].min() if 'month' in df.columns else 'N/A',
                'end': df['month'].max() if 'month' in df.columns else 'N/A'
            },
            'towns': df['town'].nunique() if 'town' in df.columns else 0,
            'flat_types': df['flat_type'].nunique() if 'flat_type' in df.columns else 0,
            'price_range': {
                'min': df['resale_price'].min() if 'resale_price' in df.columns else 0,
                'max': df['resale_price'].max() if 'resale_price' in df.columns else 0,
                'mean': df['resale_price'].mean() if 'resale_price' in df.columns else 0
            }
        }
        return summary

# SOURITRA SAMANTA (3C)
