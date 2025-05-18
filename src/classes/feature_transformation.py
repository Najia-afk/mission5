from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

class GenericFeatureTransformer:
    def __init__(self):
        self.column_types = None
        self.transformer = None
        self.categorical_encoders = {}
        self.timestamp_transformer = None
        self.original_columns = None
    
    def fit_transform(self, df, categorical_cols=None, count_cols=None, timestamp_col=None):
        """
        Generic transformation pipeline that handles all column types.
        
        Parameters:
        -----------
        df : DataFrame
            Input data
        categorical_cols : list, optional
            Columns to treat as categorical (will be one-hot encoded)
        count_cols : list, optional
            Count columns (will be log-transformed)
        timestamp_col : str, optional
            Timestamp column (will be transformed to features)
        """
        # Store original column information
        self.original_columns = df.columns.tolist()
        
        # Auto-detect column types if not specified
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if count_cols is None:
            # Heuristic: integer columns with only positive values and limited unique values
            possible_count_cols = df.select_dtypes(include=['int']).columns
            count_cols = [col for col in possible_count_cols if 
                         (df[col] >= 0).all() and 
                         df[col].nunique() < len(df) * 0.1]
        
        # Store column type mapping
        self.column_types = {
            'categorical': categorical_cols,
            'count': count_cols,
            'timestamp': [timestamp_col] if timestamp_col else [],
            'numerical': [col for col in df.columns if col not in 
                         categorical_cols + count_cols + ([timestamp_col] if timestamp_col else [])]
        }
        
        # Build transformers for each column type
        transformers = []
        
        # Categorical columns -> OneHotEncoding
        if categorical_cols:
            cat_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ])
            transformers.append(('categorical', cat_transformer, categorical_cols))
        
        # Count columns -> Log transform + scaling
        if count_cols:
            # Log transform with handling of zeros
            def log_transform(X):
                return np.log1p(X)
                
            count_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('log', FunctionTransformer(log_transform)),
                ('scaler', RobustScaler())
            ])
            transformers.append(('count', count_transformer, count_cols))
        
        # Timestamp column -> Extract features
        if timestamp_col:
            def extract_time_features(X):
                # Convert to datetime if string
                if isinstance(X, pd.DataFrame):
                    # Get the Series from the DataFrame
                    X = X.iloc[:, 0]  # Extract the first (and only) column
                    
                # Now X is a Series and we can check its dtype
                if X.dtype == 'object':
                    X = pd.to_datetime(X)
                
                # Extract useful features
                day_of_week = X.dt.dayofweek.values.reshape(-1, 1)
                month = X.dt.month.values.reshape(-1, 1)
                quarter = X.dt.quarter.values.reshape(-1, 1)
                
                return np.hstack([day_of_week, month, quarter])
                
            time_transformer = FunctionTransformer(extract_time_features)
            transformers.append(('timestamp', time_transformer, [timestamp_col]))
            
        # Numerical columns -> Standard scaling
        numerical_cols = self.column_types['numerical']
        if numerical_cols:
            num_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])
            transformers.append(('numerical', num_transformer, numerical_cols))
        
        # Create and fit the column transformer
        self.transformer = ColumnTransformer(transformers, remainder='drop')
        result = self.transformer.fit_transform(df)
        
        # Create result DataFrame with proper column names
        feature_names = self._get_feature_names()
        result_df = pd.DataFrame(result, columns=feature_names)
        
        return result_df
    
    def transform(self, df):
        """Transform new data using the fitted transformer"""
        if self.transformer is None:
            raise ValueError("Transformer not fitted. Call fit_transform first.")
        
        result = self.transformer.transform(df)
        feature_names = self._get_feature_names()
        return pd.DataFrame(result, columns=feature_names)
    
    def inverse_transform(self, transformed_df):
        """Generic inverse transform that works with any column type"""
        if self.transformer is None:
            raise ValueError("Transformer not fitted. Call fit_transform first.")
        
        # Create a result dataframe
        result_df = pd.DataFrame(index=range(len(transformed_df)))
        
        # Inverse transform numerical columns
        num_cols = self.column_types['numerical']
        if num_cols:
            num_transformer = self.transformer.named_transformers_['numerical']
            # Get the indices of the numerical columns in the transformed data
            num_start_idx = 0
            for name, _, _ in self.transformer.transformers_:
                if name == 'numerical':
                    break
                num_start_idx += self._get_feature_count(name)
            
            num_end_idx = num_start_idx + len(num_cols)
            num_data = transformed_df.iloc[:, num_start_idx:num_end_idx].values
            
            # Apply inverse scaling
            num_scaler = num_transformer.named_steps['scaler']
            num_data_unscaled = num_scaler.inverse_transform(num_data)
            
            # Add to result
            for i, col in enumerate(num_cols):
                result_df[col] = num_data_unscaled[:, i]
        
        # Inverse transform count columns (reverse log)
        count_cols = self.column_types['count']
        if count_cols:
            count_start_idx = 0
            for name, _, _ in self.transformer.transformers_:
                if name == 'count':
                    break
                count_start_idx += self._get_feature_count(name)
            
            count_end_idx = count_start_idx + len(count_cols)
            count_data = transformed_df.iloc[:, count_start_idx:count_end_idx].values
            
            # Reverse scaling and log transform
            count_transformer = self.transformer.named_transformers_['count']
            count_scaler = count_transformer.named_steps['scaler']
            count_data_unscaled = count_scaler.inverse_transform(count_data)
            
            # Reverse log transform (exp(x) - 1)
            count_data_original = np.expm1(count_data_unscaled)
            
            # Add to result and round to integers
            for i, col in enumerate(count_cols):
                result_df[col] = np.round(count_data_original[:, i]).astype(int)
        
        # Categorical columns require more complex handling for OHE inversion
        # This is a simplified approach
        cat_cols = self.column_types['categorical']
        if cat_cols:
            # For each categorical column, find the one-hot encoded columns and convert back
            # This would need to be implemented based on your specific requirements
            pass
        
        return result_df
    
    def _get_feature_count(self, transformer_name):
        """Count how many output features a transformer creates"""
        for name, transformer, columns in self.transformer.transformers_:
            if name == transformer_name:
                if name == 'categorical':
                    # For categorical, count how many categories in total
                    return sum(len(self.transformer.named_transformers_[name]
                                  .named_steps['encoder']
                                  .categories_[i]) 
                              for i in range(len(columns)))
                elif name == 'timestamp':
                    # For timestamp we extract 3 features
                    return 3 * len(columns)
                else:
                    # For numerical and count, it's 1:1
                    return len(columns)
        return 0
    
    def _get_feature_names(self):
        """Generate feature names for the transformed data"""
        feature_names = []
        
        for name, transformer, columns in self.transformer.transformers_:
            if name == 'categorical':
                # Get encoded feature names from OneHotEncoder
                encoder = transformer.named_steps['encoder']
                for i, col in enumerate(columns):
                    for category in encoder.categories_[i]:
                        feature_names.append(f"{col}_{category}")
            
            elif name == 'timestamp':
                # Add timestamp derived features
                for col in columns:
                    feature_names.extend([f"{col}_day_of_week", f"{col}_month", f"{col}_quarter"])
            
            else:
                # Numerical and count columns keep their names
                feature_names.extend(columns)
        
        return feature_names