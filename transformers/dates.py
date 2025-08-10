import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from scipy import stats
from scipy.sparse import csr_matrix, hstack
import warnings
warnings.filterwarnings('ignore')

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts comprehensive date features from datetime columns.
    
    This transformer converts datetime columns into multiple engineered features
    that capture temporal patterns useful for machine learning models.
    
    Parameters
    ----------
    date_columns : list of str
        The names of the columns in X to parse as dates.
    drop_original : bool, default=True
        If True, the original date columns are dropped after feature extraction.
    include_cyclical : bool, default=False
        If True, creates cyclical encodings (sin/cos) for periodic features.
    include_relative : bool, default=False
        If True, creates relative time features (days since epoch, etc.).
    include_business : bool, default=False
        If True, creates business-specific features (quarter, fiscal year, etc.).
    verbose : bool, default=True
        If True, prints feature extraction summary.
    """
    
    def __init__(
        self, 
        date_columns, 
        drop_original=True,
        include_cyclical=False,
        include_relative=False,
        include_business=False,
        verbose=True
    ):
        self.date_columns = date_columns
        self.drop_original = drop_original
        self.include_cyclical = include_cyclical
        self.include_relative = include_relative
        self.include_business = include_business
        self.verbose = verbose
        self.feature_names_out_ = []

    def fit(self, X, y=None):
        """
        Learn the structure of date columns (nothing to actually fit).
        
        Parameters
        ----------
        X : pandas DataFrame
            Input data containing date columns.
        y : array-like, optional
            Target values (ignored).
            
        Returns
        -------
        self : DateFeatureExtractor
            Returns self.
        """
        if self.verbose:
            print("📅 DATE FEATURE EXTRACTION ANALYSIS")
            print("="*40)
            
            X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
            
            for col in self.date_columns:
                if col in X_df.columns:
                    # Analyze date range and patterns
                    date_series = pd.to_datetime(X_df[col])
                    min_date = date_series.min()
                    max_date = date_series.max()
                    date_range = max_date - min_date
                    
                    print(f"\n📊 {col}:")
                    print(f"   Range: {min_date.date()} to {max_date.date()}")
                    print(f"   Span: {date_range.days} days")
                    print(f"   Null values: {date_series.isnull().sum()}")
                    
                    # Detect patterns
                    unique_times = date_series.dt.time.nunique()
                    unique_dates = date_series.dt.date.nunique()
                    
                    if unique_times == 1:
                        print(f"   Pattern: Date only (consistent time)")
                    elif unique_dates == len(date_series):
                        print(f"   Pattern: Timestamp (unique timestamps)")
                    else:
                        print(f"   Pattern: Mixed ({unique_dates} unique dates, {unique_times} unique times)")
        
        return self

    def transform(self, X):
        """
        Transform datetime columns into engineered features.
        
        Parameters
        ----------
        X : pandas DataFrame
            Input data containing date columns.
            
        Returns
        -------
        X_transformed : numpy array
            Transformed data with date features.
        """
        Xt = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        # Track output feature names
        output_features = []
        
        # Convert to datetime and extract features
        for col in self.date_columns:
            if col not in Xt.columns:
                continue
                
            # Ensure datetime dtype
            Xt[col] = pd.to_datetime(Xt[col])
            dt_series = Xt[col]
            
            # Basic date components
            basic_features = self._extract_basic_features(dt_series, col)
            for feature_name, feature_values in basic_features.items():
                Xt[feature_name] = feature_values
                output_features.append(feature_name)
            
            # Cyclical features (optional)
            if self.include_cyclical:
                cyclical_features = self._extract_cyclical_features(dt_series, col)
                for feature_name, feature_values in cyclical_features.items():
                    Xt[feature_name] = feature_values
                    output_features.append(feature_name)
            
            # Relative time features (optional)
            if self.include_relative:
                relative_features = self._extract_relative_features(dt_series, col)
                for feature_name, feature_values in relative_features.items():
                    Xt[feature_name] = feature_values
                    output_features.append(feature_name)
            
            # Business features (optional)
            if self.include_business:
                business_features = self._extract_business_features(dt_series, col)
                for feature_name, feature_values in business_features.items():
                    Xt[feature_name] = feature_values
                    output_features.append(feature_name)
        
        # Add remaining columns (non-date columns)
        for col in Xt.columns:
            if col not in self.date_columns and col not in output_features:
                output_features.append(col)
        
        # Drop original date columns if requested
        if self.drop_original:
            Xt = Xt.drop(columns=self.date_columns)
            # Remove dropped columns from output_features
            output_features = [col for col in output_features if col not in self.date_columns]
        
        # Store feature names for get_feature_names_out
        self.feature_names_out_ = output_features
        
        if self.verbose:
            self._print_transformation_summary(len(self.date_columns), len(output_features))
        
        return Xt.values  # Return numpy array for sklearn compatibility
    
    def _extract_basic_features(self, dt_series, col_name):
        """Extract basic date/time components"""
        features = {}
        
        # Core date components
        features[f"{col_name}_year"] = dt_series.dt.year
        features[f"{col_name}_month"] = dt_series.dt.month
        features[f"{col_name}_day"] = dt_series.dt.day
        features[f"{col_name}_weekday"] = dt_series.dt.weekday  # Monday=0, Sunday=6
        features[f"{col_name}_hour"] = dt_series.dt.hour
        
        # Derived features
        features[f"{col_name}_is_weekend"] = (dt_series.dt.weekday >= 5).astype(int)
        features[f"{col_name}_is_month_start"] = dt_series.dt.is_month_start.astype(int)
        features[f"{col_name}_is_month_end"] = dt_series.dt.is_month_end.astype(int)
        features[f"{col_name}_is_quarter_start"] = dt_series.dt.is_quarter_start.astype(int)
        features[f"{col_name}_is_quarter_end"] = dt_series.dt.is_quarter_end.astype(int)
        features[f"{col_name}_is_year_start"] = dt_series.dt.is_year_start.astype(int)
        features[f"{col_name}_is_year_end"] = dt_series.dt.is_year_end.astype(int)
        
        # Day of year and week of year
        features[f"{col_name}_day_of_year"] = dt_series.dt.dayofyear
        features[f"{col_name}_week_of_year"] = dt_series.dt.isocalendar().week
        
        return features
    
    def _extract_cyclical_features(self, dt_series, col_name):
        """Extract cyclical encodings for periodic patterns"""
        features = {}
        
        # Month cyclical (captures seasonality)
        month_cycle = 2 * np.pi * dt_series.dt.month / 12
        features[f"{col_name}_month_sin"] = np.sin(month_cycle)
        features[f"{col_name}_month_cos"] = np.cos(month_cycle)
        
        # Day of week cyclical
        dow_cycle = 2 * np.pi * dt_series.dt.weekday / 7
        features[f"{col_name}_weekday_sin"] = np.sin(dow_cycle)
        features[f"{col_name}_weekday_cos"] = np.cos(dow_cycle)
        
        # Hour cyclical (for timestamps)
        hour_cycle = 2 * np.pi * dt_series.dt.hour / 24
        features[f"{col_name}_hour_sin"] = np.sin(hour_cycle)
        features[f"{col_name}_hour_cos"] = np.cos(hour_cycle)
        
        # Day of year cyclical (annual patterns)
        doy_cycle = 2 * np.pi * dt_series.dt.dayofyear / 365.25
        features[f"{col_name}_dayofyear_sin"] = np.sin(doy_cycle)
        features[f"{col_name}_dayofyear_cos"] = np.cos(doy_cycle)
        
        return features
    
    def _extract_relative_features(self, dt_series, col_name):
        """Extract relative time features"""
        features = {}
        
        # Days since Unix epoch
        epoch = pd.Timestamp('1970-01-01')
        features[f"{col_name}_days_since_epoch"] = (dt_series - epoch).dt.days
        
        # Days since minimum date in series (relative to dataset)
        min_date = dt_series.min()
        features[f"{col_name}_days_since_min"] = (dt_series - min_date).dt.days
        
        # Days until maximum date in series
        max_date = dt_series.max()
        features[f"{col_name}_days_until_max"] = (max_date - dt_series).dt.days
        
        # Time since midnight (fractional day)
        features[f"{col_name}_time_of_day"] = (
            dt_series.dt.hour * 3600 + 
            dt_series.dt.minute * 60 + 
            dt_series.dt.second
        ) / 86400  # Normalize to [0, 1]
        
        return features
    
    def _extract_business_features(self, dt_series, col_name):
        """Extract business-relevant features"""
        features = {}
        
        # Quarter
        features[f"{col_name}_quarter"] = dt_series.dt.quarter
        
        # Business day indicators
        # Note: This is a simplified version - real business days would consider holidays
        features[f"{col_name}_is_business_day"] = (
            (dt_series.dt.weekday < 5) &  # Monday-Friday
            (~dt_series.dt.is_month_start) &  # Exclude typical holidays
            (~dt_series.dt.is_year_start)
        ).astype(int)
        
        # Fiscal year (assuming fiscal year starts in April)
        fiscal_year = dt_series.dt.year.copy()
        fiscal_year[dt_series.dt.month < 4] -= 1
        features[f"{col_name}_fiscal_year"] = fiscal_year
        
        # Season (meteorological seasons)
        month = dt_series.dt.month
        season = month.copy()
        season[(month >= 3) & (month <= 5)] = 1  # Spring
        season[(month >= 6) & (month <= 8)] = 2  # Summer
        season[(month >= 9) & (month <= 11)] = 3  # Fall
        season[(month == 12) | (month <= 2)] = 4  # Winter
        features[f"{col_name}_season"] = season
        
        return features
    
    def _print_transformation_summary(self, original_date_cols, total_features):
        """Print summary of transformation"""
        features_per_date = (total_features - (len(self.date_columns) if not self.drop_original else 0)) / len(self.date_columns)
        
        print(f"\n🔧 TRANSFORMATION SUMMARY:")
        print(f"   Date columns processed: {original_date_cols}")
        print(f"   Features per date column: ~{features_per_date:.0f}")
        print(f"   Total output features: {total_features}")
        print(f"   Original columns dropped: {self.drop_original}")
        
        feature_types = []
        if True:  # Basic features always included
            feature_types.append("Basic")
        if self.include_cyclical:
            feature_types.append("Cyclical")
        if self.include_relative:
            feature_types.append("Relative")
        if self.include_business:
            feature_types.append("Business")
        
        print(f"   Feature types: {', '.join(feature_types)}")
    
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.
        
        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features. If None, uses generic feature names.
            
        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        if hasattr(self, 'feature_names_out_') and self.feature_names_out_:
            return np.array(self.feature_names_out_, dtype=object)
        
        # Fallback if transform hasn't been called yet
        if input_features is not None:
            output_features = []
            
            # Generate expected feature names for each date column
            for col in self.date_columns:
                output_features.extend(self._get_expected_feature_names(col))
            
            # Add non-date columns
            if isinstance(input_features, (list, np.ndarray)):
                for feature in input_features:
                    if feature not in self.date_columns:
                        output_features.append(feature)
            
            # Remove original date columns if drop_original=True
            if self.drop_original:
                output_features = [f for f in output_features if f not in self.date_columns]
            
            return np.array(output_features, dtype=object)
        
        # Final fallback
        return np.array([], dtype=object)
    
    def _get_expected_feature_names(self, col_name):
        """Get expected feature names for a single date column"""
        features = []
        
        # Basic features (always included)
        features.extend([
            f"{col_name}_year",
            f"{col_name}_month",
            f"{col_name}_day",
            f"{col_name}_weekday",
            f"{col_name}_hour",
            f"{col_name}_is_weekend",
            f"{col_name}_is_month_start",
            f"{col_name}_is_month_end",
            f"{col_name}_is_quarter_start",
            f"{col_name}_is_quarter_end",
            f"{col_name}_is_year_start",
            f"{col_name}_is_year_end",
            f"{col_name}_day_of_year",
            f"{col_name}_week_of_year"
        ])
        
        # Cyclical features
        if self.include_cyclical:
            features.extend([
                f"{col_name}_month_sin",
                f"{col_name}_month_cos",
                f"{col_name}_weekday_sin",
                f"{col_name}_weekday_cos",
                f"{col_name}_hour_sin",
                f"{col_name}_hour_cos",
                f"{col_name}_dayofyear_sin",
                f"{col_name}_dayofyear_cos"
            ])
        
        # Relative features
        if self.include_relative:
            features.extend([
                f"{col_name}_days_since_epoch",
                f"{col_name}_days_since_min",
                f"{col_name}_days_until_max",
                f"{col_name}_time_of_day"
            ])
        
        # Business features
        if self.include_business:
            features.extend([
                f"{col_name}_quarter",
                f"{col_name}_is_business_day",
                f"{col_name}_fiscal_year",
                f"{col_name}_season"
            ])
        
        return features