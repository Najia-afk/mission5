from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

class FeatureTransformation:
    def __init__(self):
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def transform_features(self, df):
        # Preserve order_purchase_timestamp
        date_col = df['order_purchase_timestamp'].copy()
        
        # List of numerical columns to scale
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Handle missing values and scale
        df[num_cols] = self.imputer.fit_transform(df[num_cols])
        df[num_cols] = self.scaler.fit_transform(df[num_cols])
        
        # Restore date column
        df['order_purchase_timestamp'] = date_col
        return df