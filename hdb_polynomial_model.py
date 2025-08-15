# SOURITRA SAMANTA (3C)

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from typing import Dict, Any, List, Tuple
import os

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class HDBPolynomialPriceModel:
    def __init__(self):
        """Initialize the HDB polynomial price prediction model"""
        self.polynomial_pipeline = None
        self.label_encoders = {}
        self.df = None
        self.feature_names = []
        self.is_trained = False
        self.model_metrics = {}
        self.polynomial_degree = 4

    def load_data(self, filepath: str = 'sample_data.csv') -> pd.DataFrame:
        """Load HDB data from CSV file"""
        try:
            if os.path.exists(filepath):
                self.df = pd.read_csv(filepath)
                print(f"📊 Loaded {len(self.df)} HDB records from {filepath}")
                return self.df
            else:
                raise FileNotFoundError(f"Data file {filepath} not found")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise

    def preprocess_data(self):
        """Preprocess the data for polynomial machine learning"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Create a copy for processing
        processed_df = self.df.copy()
        
        # Handle missing values
        processed_df = processed_df.dropna()
        
        # Clean and prepare categorical variables (including flat_model for accuracy)
        categorical_columns = ['town', 'flat_type', 'storey_range', 'flat_model']
        
        # Encode categorical variables
        for col in categorical_columns:
            if col in processed_df.columns:
                # Clean the categorical data
                processed_df[col] = processed_df[col].str.upper().str.strip()
                
                # Encode
                le = LabelEncoder()
                processed_df[col + '_encoded'] = le.fit_transform(processed_df[col])
                self.label_encoders[col] = le
        
        # Prepare features and target
        feature_columns = [col + '_encoded' for col in categorical_columns if col in processed_df.columns]
        feature_columns.extend(['floor_area_sqm', 'remaining_lease'])
        
        # Filter available columns
        available_columns = [col for col in feature_columns if col in processed_df.columns]
        
        X = processed_df[available_columns]
        y = processed_df['resale_price']
        
        self.feature_names = available_columns
        
        return X, y

    def create_polynomial_pipeline(self):
        """Create polynomial regression pipeline with 4th degree features"""
        # Create pipeline: Polynomial Features -> Linear Regression
        self.polynomial_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly_features', PolynomialFeatures(degree=self.polynomial_degree, include_bias=False)),
            ('linear_reg', LinearRegression())
        ])
        
        print(f"✅ Created polynomial pipeline with degree {self.polynomial_degree}")

    def train_model(self):
        """Train the 4th degree polynomial regression model"""
        try:
            X, y = self.preprocess_data()
            
            # Create polynomial pipeline
            self.create_polynomial_pipeline()
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train the polynomial model
            print(f"🔄 Training {self.polynomial_degree}th degree polynomial regression model...")
            self.polynomial_pipeline.fit(X_train, y_train)
            
            # Calculate predictions
            y_pred_train = self.polynomial_pipeline.predict(X_train)
            y_pred_test = self.polynomial_pipeline.predict(X_test)
            
            # Calculate metrics
            self.model_metrics = {
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'n_samples': len(X),
                'n_features': len(self.feature_names),
                'polynomial_degree': self.polynomial_degree
            }
            
            self.is_trained = True
            print(f"✅ Polynomial model trained successfully!")
            print(f"   Training R² Score: {self.model_metrics['train_r2']:.4f}")
            print(f"   Testing R² Score: {self.model_metrics['test_r2']:.4f}")
            print(f"   Test MAE: SGD ${self.model_metrics['test_mae']:,.2f}")
            print(f"   Test RMSE: SGD ${self.model_metrics['test_rmse']:,.2f}")
            
        except Exception as e:
            print(f"❌ Error training polynomial model: {e}")
            raise

    

    def predict_price(self, inputs: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Predict HDB price using polynomial regression"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Prepare input data
        input_data = []
        feature_contributions = {}
        
        # Encode categorical variables (including flat_model for accuracy)
        categorical_columns = ['town', 'flat_type', 'storey_range', 'flat_model']
        
        for col in categorical_columns:
            if col in inputs:
                if col in self.label_encoders:
                    try:
                        # Clean input data
                        clean_input = str(inputs[col]).upper().strip()
                        encoded_value = self.label_encoders[col].transform([clean_input])[0]
                        input_data.append(encoded_value)
                    except ValueError:
                        # If value not seen during training, use most common value
                        encoded_value = 0
                        input_data.append(encoded_value)
                        print(f"⚠️  Warning: {col} value '{inputs[col]}' not seen during training")
        
        # Add numerical features
        numerical_features = ['floor_area_sqm', 'remaining_lease']
        for feature in numerical_features:
            if feature in inputs:
                input_data.append(inputs[feature])
        
        # Make prediction using polynomial pipeline
        input_array = np.array(input_data).reshape(1, -1)
        prediction = self.polynomial_pipeline.predict(input_array)[0]
        
        # Calculate feature importance (simplified for polynomial features)
        # We'll show the importance of original features before polynomial transformation
        scaler = self.polynomial_pipeline.named_steps['scaler']
        scaled_input = scaler.transform(input_array)
        
        for i, (feature, value) in enumerate(zip(self.feature_names, input_data)):
            if feature.endswith('_encoded'):
                original_feature = feature.replace('_encoded', '')
                feature_contributions[original_feature] = scaled_input[0][i] * 10000  # Scaled contribution
            else:
                feature_contributions[feature] = scaled_input[0][i] * 10000
        
        return prediction, feature_contributions

    def get_available_towns(self) -> List[str]:
        if self.df is not None:
            return sorted(self.df['town'].unique().tolist())
        return []

    def get_available_flat_types(self) -> List[str]:
        if self.df is not None:
            return sorted(self.df['flat_type'].unique().tolist())
        return []

    def get_available_storey_ranges(self) -> List[str]:
        if self.df is not None:
            return sorted(self.df['storey_range'].unique().tolist())
        return []

    def get_model_metrics(self) -> Dict[str, float]:
        return self.model_metrics.copy()

    def get_polynomial_equation_info(self) -> Dict[str, Any]:
        if not self.is_trained:
            return {
                'degree': self.polynomial_degree,
                'n_features': 0,
                'is_trained': False
            }
        
        poly_features = self.polynomial_pipeline.named_steps['poly_features']
        n_polynomial_features = poly_features.n_output_features_
        
        return {
            'degree': self.polynomial_degree,
            'n_features': n_polynomial_features,
            'is_trained': self.is_trained,
            'original_features': len(self.feature_names)
        }
    
# SOURITRA SAMANTA (3C)
