class FeatureEngineering:
    def __init__(self, df_rfm, df_satisfaction, df_behavior):
        self.df_rfm = df_rfm
        self.df_satisfaction = df_satisfaction
        self.df_behavior = df_behavior
        
    def combine_features(self):
        return (self.df_rfm
                .merge(self.df_satisfaction, on='customer_id', how='left')
                .merge(self.df_behavior, on='customer_id', how='left'))