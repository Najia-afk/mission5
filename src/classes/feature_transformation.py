from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import pandas as pd

class FeatureTransformation:
    _instance = None
    _is_initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FeatureTransformation, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._is_initialized:
            # Initialize scalers with appropriate ranges
            self.rfm_scalers = {
                'recency': MinMaxScaler(feature_range=(0, 1)),  # Lower is better
                'frequency': MinMaxScaler(feature_range=(0, 1)),  # Simple weight
                'monetary': MinMaxScaler(feature_range=(0, 1))   # Simple weight
            }
            self.review_scalers = {
                'avg_review_score': MinMaxScaler(feature_range=(0, 1)),  # higher is better
                'review_count': MinMaxScaler(feature_range=(0, 1))  # Simple weight
            }
            self.time_scaler = RobustScaler()    # Keep robust for time
            self.imputer = SimpleImputer(strategy='median')
            
            # Feature groups
            self.rfm_features = ['recency_days', 'frequency', 'monetary']
            self.review_features = ['avg_review_score', 'review_count']
            self.time_features = ['avg_delivery_time']
            
            self.fitted_scalers = {}
            self._is_initialized = True
    
    def transform_features(self, df):
        """Transform features with weighted RFM scaling."""
        date_col = df['order_purchase_timestamp'].copy() if 'order_purchase_timestamp' in df.columns else None
        df_transformed = df.copy()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        df_transformed[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        # Transform RFM features individually with weights
        df_transformed['recency_days'] = -1 * df_transformed['recency_days']  # Invert recency
        self.rfm_scalers['recency'].fit(df_transformed[['recency_days']])
        self.rfm_scalers['frequency'].fit(df_transformed[['frequency']])
        self.rfm_scalers['monetary'].fit(df_transformed[['monetary']])
        
        df_transformed['recency_days'] = self.rfm_scalers['recency'].transform(df_transformed[['recency_days']])
        df_transformed['frequency'] = self.rfm_scalers['frequency'].transform(df_transformed[['frequency']])
        df_transformed['monetary'] = self.rfm_scalers['monetary'].transform(df_transformed[['monetary']])
        
        # Transform review features individually
        self.review_scalers['avg_review_score'].fit(df_transformed[['avg_review_score']])
        self.review_scalers['review_count'].fit(df_transformed[['review_count']])
        
        df_transformed['avg_review_score'] = self.review_scalers['avg_review_score'].transform(df_transformed[['avg_review_score']])
        df_transformed['review_count'] = self.review_scalers['review_count'].transform(df_transformed[['review_count']])
        
        # Transform time features
        self.time_scaler.fit(df_transformed[self.time_features])
        df_transformed[self.time_features] = self.time_scaler.transform(df_transformed[self.time_features])
        
        # Store fitted scalers
        self.fitted_scalers = {
            'rfm_recency': self.rfm_scalers['recency'],
            'rfm_frequency': self.rfm_scalers['frequency'],
            'rfm_monetary': self.rfm_scalers['monetary'],
            'review_score': self.review_scalers['avg_review_score'],
            'review_count': self.review_scalers['review_count'],
            'time': self.time_scaler
        }
        
        if date_col is not None:
            df_transformed['order_purchase_timestamp'] = date_col
        return df_transformed
    
    def inverse_transform_features(self, df):
        """Inverse transform scaled features."""
        if not self.fitted_scalers:
            raise ValueError("Scalers not fitted. Call transform_features first.")
            
        df_original = df.copy()
        
        # Inverse transform RFM features
        if 'recency_days' in df.columns:
            df_original['recency_days'] = self.fitted_scalers['rfm_recency'].inverse_transform(df[['recency_days']])
            df_original['recency_days'] = -1 * df_original['recency_days']  # Revert recency inversion
            
        if 'frequency' in df.columns:
            df_original['frequency'] = self.fitted_scalers['rfm_frequency'].inverse_transform(df[['frequency']])
            df_original['frequency'] = df_original['frequency'].round().astype(int)
            
        if 'monetary' in df.columns:
            df_original['monetary'] = self.fitted_scalers['rfm_monetary'].inverse_transform(df[['monetary']])
            df_original['monetary'] = df_original['monetary'].round(2)
            
        # Inverse transform review features
        if 'avg_review_score' in df.columns:
            df_original['avg_review_score'] = self.fitted_scalers['review_score'].inverse_transform(df[['avg_review_score']])
            df_original['avg_review_score'] = df_original['avg_review_score'].round(1)
            
        if 'review_count' in df.columns:
            df_original['review_count'] = self.fitted_scalers['review_count'].inverse_transform(df[['review_count']])
            df_original['review_count'] = df_original['review_count'].round().astype(int)
            
        # Inverse transform time features
        if set(self.time_features).issubset(df.columns):
            df_original[self.time_features] = self.fitted_scalers['time'].inverse_transform(df[self.time_features])
            df_original[self.time_features] = df_original[self.time_features].round(1)
        
        return df_original