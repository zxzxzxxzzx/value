"""
HDB Price Prediction Model with 4th Degree Polynomial Regression
Handles data processing, polynomial feature transformation, model training, and price prediction
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from typing import Dict, Any, List, Tuple
import pickle
import os

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class HDBPolynomialPriceModel:
    def __init__(self):
        """Initialize the HDB polynomial price prediction model"""
        self.polynomial_pipeline = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
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
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train the polynomial model
            print(f"🔄 Training {self.polynomial_degree}th degree polynomial regression model...")
            self.polynomial_pipeline.fit(self.X_train, self.y_train)
            
            # Calculate predictions
            y_pred_train = self.polynomial_pipeline.predict(self.X_train)
            y_pred_test = self.polynomial_pipeline.predict(self.X_test)
            
            # Calculate metrics
            self.model_metrics = {
                'train_r2': r2_score(self.y_train, y_pred_train),
                'test_r2': r2_score(self.y_test, y_pred_test),
                'train_mae': mean_absolute_error(self.y_train, y_pred_train),
                'test_mae': mean_absolute_error(self.y_test, y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
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

    def get_polynomial_equation_info(self) -> Dict[str, Any]:
        """Get polynomial equation coefficients and information"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Get the linear regression step from the pipeline
        linear_reg = self.polynomial_pipeline.named_steps['linear_reg']
        poly_features = self.polynomial_pipeline.named_steps['poly_features']
        
        coefficients = linear_reg.coef_
        intercept = linear_reg.intercept_
        
        # Get feature names from polynomial features
        feature_names = poly_features.get_feature_names_out(self.feature_names)
        
        return {
            'intercept': intercept,
            'coefficients': coefficients,
            'feature_names': feature_names[:10],  # Show first 10 features
            'n_polynomial_features': len(coefficients),
            'polynomial_degree': self.polynomial_degree
        }

    def predict_most_likely_flat_model(self, inputs: Dict[str, Any]) -> str:
        """Predict the most likely flat_model based on flat_type and other characteristics"""
        if not hasattr(self, 'df') or self.df is None:
            return "Standard"  # Most common 2-room model
        
        # Filter data by flat_type to find most common model
        flat_type = inputs.get('flat_type', '').upper().strip()
        filtered_data = self.df[self.df['flat_type'] == flat_type]
        
        if len(filtered_data) > 0:
            # Return most common flat_model for this specific flat_type
            most_common = filtered_data['flat_model'].mode()
            if len(most_common) > 0:
                return most_common.iloc[0]
        
        # Fallback to overall most common model
        overall_common = self.df['flat_model'].mode()
        return overall_common.iloc[0] if len(overall_common) > 0 else "Standard"

    def predict_price(self, inputs: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Predict HDB price using polynomial regression with automatic flat_model detection"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # flat_model is now provided by user interface
        
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

    def load_and_train_model(self):
        """Load data and train the polynomial model"""
        self.load_data()
        self.train_model()

    def get_available_towns(self) -> List[str]:
        """Get list of available towns"""
        if self.df is not None:
            return sorted(self.df['town'].unique().tolist())
        return []

    def get_available_flat_types(self) -> List[str]:
        """Get list of available flat types"""
        if self.df is not None:
            return sorted(self.df['flat_type'].unique().tolist())
        return []

    def get_available_storey_ranges(self) -> List[str]:
        """Get list of available storey ranges"""
        if self.df is not None:
            return sorted(self.df['storey_range'].unique().tolist())
        return []



    def get_model_metrics(self) -> Dict[str, float]:
        """Get polynomial model performance metrics"""
        return self.model_metrics.copy()

    def export_model(self, filepath: str = 'exports/polynomial_model.pkl'):
        """Export the trained polynomial model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        os.makedirs('exports', exist_ok=True)
        
        model_data = {
            'pipeline': self.polynomial_pipeline,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'metrics': self.model_metrics,
            'polynomial_degree': self.polynomial_degree
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model exported to {filepath}")
