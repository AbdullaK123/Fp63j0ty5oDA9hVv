import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from scipy import stats
from scipy.sparse import csr_matrix, hstack
import warnings
warnings.filterwarnings('ignore')


class SparseMaster(BaseEstimator, TransformerMixin):
    """
    The ULTIMATE sparse numerical feature transformer.
    Handles the chaos of sparse data like a boss.
    """
    
    def __init__(
        self, 
        sparsity_threshold=0.90,        # 90%+ zeros = sparse
        zero_encoding='separate',        # 'separate', 'median', 'mode', 'drop'
        scaling_method='robust',         # 'standard', 'robust', 'power'
        outlier_treatment='clip',        # 'clip', 'remove', 'transform'
        create_log_features=True,        # Log transform for skewed data
        create_binary_indicators=True,   # "Has this feature" binary flags
        create_binned_features=True,     # Discretize continuous sparse features
        create_interaction_features=False, # Sparse × sparse interactions
        preserve_sparsity=True,          # Keep memory efficient
        verbose=True
    ):
        
        self.sparsity_threshold = sparsity_threshold
        self.zero_encoding = zero_encoding
        self.scaling_method = scaling_method
        self.outlier_treatment = outlier_treatment
        self.create_log_features = create_log_features
        self.create_binary_indicators = create_binary_indicators
        self.create_binned_features = create_binned_features
        self.create_interaction_features = create_interaction_features
        self.preserve_sparsity = preserve_sparsity
        self.verbose = verbose
        
        # Storage for fitted parameters
        self.sparse_columns_ = []
        self.dense_columns_ = []
        self.sparsity_stats_ = {}
        self.scalers_ = {}
        self.outlier_bounds_ = {}
        self.bin_edges_ = {}
        self.feature_names_ = []
        
    def fit(self, X, y=None):
        """
        Analyze the sparse chaos and learn how to handle it
        """
        if self.verbose:
            print("🕳️  SPARSE FEATURE ANALYSIS")
            print("="*40)
        
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Analyze sparsity patterns
        self._analyze_sparsity(X_df)
        
        # Separate sparse and dense features
        self._categorize_features(X_df)
        
        # Fit transformations for each category
        self._fit_transformations(X_df)
        
        if self.verbose:
            self._print_analysis_report()
            
        return self
    
    def _analyze_sparsity(self, X):
        """Deep dive into sparsity patterns"""
        for col in X.columns:
            total_count = len(X[col])
            zero_count = (X[col] == 0).sum()
            null_count = X[col].isnull().sum()
            
            sparsity_ratio = (zero_count + null_count) / total_count
            
            # Additional stats for non-zero values
            non_zero_values = X[col][(X[col] != 0) & (X[col].notnull())]
            
            stats_dict = {
                'sparsity_ratio': sparsity_ratio,
                'zero_count': zero_count,
                'null_count': null_count,
                'non_zero_count': len(non_zero_values),
                'is_sparse': sparsity_ratio >= self.sparsity_threshold
            }
            
            if len(non_zero_values) > 0:
                stats_dict.update({
                    'non_zero_mean': non_zero_values.mean(),
                    'non_zero_std': non_zero_values.std(),
                    'non_zero_median': non_zero_values.median(),
                    'non_zero_skew': stats.skew(non_zero_values),
                    'non_zero_min': non_zero_values.min(),
                    'non_zero_max': non_zero_values.max(),
                    'outlier_ratio': self._calculate_outlier_ratio(non_zero_values)
                })
            
            self.sparsity_stats_[col] = stats_dict
    
    def _calculate_outlier_ratio(self, values):
        """Calculate percentage of outliers using IQR method"""
        if len(values) < 4:
            return 0.0
            
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((values < lower_bound) | (values > upper_bound)).sum()
        return outliers / len(values)
    
    def _categorize_features(self, X):
        """Separate sparse and dense features"""
        self.sparse_columns_ = []
        self.dense_columns_ = []
        
        for col in X.columns:
            if self.sparsity_stats_[col]['is_sparse']:
                self.sparse_columns_.append(col)
            else:
                self.dense_columns_.append(col)
    
    def _fit_transformations(self, X):
        """Fit appropriate transformations for each feature type"""
        
        # Fit scalers for non-zero values of sparse features
        for col in self.sparse_columns_:
            non_zero_mask = (X[col] != 0) & (X[col].notnull())
            if non_zero_mask.sum() > 1:  # Need at least 2 values to fit scaler
                non_zero_values = X[col][non_zero_mask].values.reshape(-1, 1)
                
                if self.scaling_method == 'standard':
                    scaler = StandardScaler()
                elif self.scaling_method == 'robust':
                    scaler = RobustScaler()
                elif self.scaling_method == 'power':
                    scaler = PowerTransformer(method='yeo-johnson')
                else:
                    scaler = RobustScaler()  # Default fallback
                
                scaler.fit(non_zero_values)
                self.scalers_[col] = scaler
                
                # Calculate outlier bounds for clipping
                if self.outlier_treatment == 'clip':
                    Q1 = np.percentile(non_zero_values, 25)
                    Q3 = np.percentile(non_zero_values, 75)
                    IQR = Q3 - Q1
                    self.outlier_bounds_[col] = {
                        'lower': Q1 - 3 * IQR,  # More aggressive clipping for sparse data
                        'upper': Q3 + 3 * IQR
                    }
                
                # Create bin edges for discretization
                if self.create_binned_features:
                    # Use quantiles for binning, but handle sparse data carefully
                    non_zero_sorted = np.sort(non_zero_values.flatten())
                    n_bins = min(5, len(np.unique(non_zero_sorted)))  # Adaptive bin count
                    
                    if n_bins > 1:
                        # Create quantile-based bins
                        bin_edges = np.unique(np.percentile(non_zero_sorted, 
                                                          np.linspace(0, 100, n_bins + 1)))
                        self.bin_edges_[col] = bin_edges
        
        # Fit scalers for dense features (normal approach)
        for col in self.dense_columns_:
            values = X[col].dropna().values.reshape(-1, 1)
            if len(values) > 1:
                if self.scaling_method == 'standard':
                    scaler = StandardScaler()
                elif self.scaling_method == 'robust':
                    scaler = RobustScaler()
                elif self.scaling_method == 'power':
                    scaler = PowerTransformer(method='yeo-johnson')
                else:
                    scaler = RobustScaler()
                
                scaler.fit(values)
                self.scalers_[col] = scaler
    
    def transform(self, X):
        """Transform the sparse chaos into ML-ready features"""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        feature_matrices = []
        feature_names = []
        
        # Transform sparse features
        for col in self.sparse_columns_:
            transformed_features, names = self._transform_sparse_feature(X_df[col], col)
            feature_matrices.append(transformed_features)
            feature_names.extend(names)
        
        # Transform dense features (standard approach)
        for col in self.dense_columns_:
            transformed_features, names = self._transform_dense_feature(X_df[col], col)
            feature_matrices.append(transformed_features)
            feature_names.extend(names)
        
        # Combine all features
        if self.preserve_sparsity and any(isinstance(m, csr_matrix) for m in feature_matrices):
            # Convert all to sparse and combine
            sparse_matrices = []
            for m in feature_matrices:
                if isinstance(m, csr_matrix):
                    sparse_matrices.append(m)
                else:
                    sparse_matrices.append(csr_matrix(m))
            result = hstack(sparse_matrices)
        else:
            # Dense combination
            result = np.hstack([m.toarray() if isinstance(m, csr_matrix) else m 
                              for m in feature_matrices])
        
        self.feature_names_ = feature_names
        return result
    
    def _transform_sparse_feature(self, series, col_name):
        """Transform a single sparse feature into multiple engineered features"""
        feature_matrices = []
        feature_names = []
        
        # Create binary indicator: "Has this feature"
        if self.create_binary_indicators:
            has_value = ((series != 0) & (series.notnull())).astype(int).values.reshape(-1, 1)
            feature_matrices.append(has_value)
            feature_names.append(f'{col_name}_has_value')
        
        # Handle zero encoding and scaling of non-zero values
        if col_name in self.scalers_:
            # Transform non-zero values
            non_zero_mask = (series != 0) & (series.notnull())
            transformed_values = np.zeros(len(series))
            
            if non_zero_mask.sum() > 0:
                non_zero_values = series[non_zero_mask].values.reshape(-1, 1)
                
                # Apply outlier treatment
                if self.outlier_treatment == 'clip' and col_name in self.outlier_bounds_:
                    bounds = self.outlier_bounds_[col_name]
                    non_zero_values = np.clip(non_zero_values, bounds['lower'], bounds['upper'])
                
                # Scale the non-zero values
                scaled_values = self.scalers_[col_name].transform(non_zero_values).flatten()
                transformed_values[non_zero_mask] = scaled_values
            
            # Handle zero encoding
            if self.zero_encoding == 'separate':
                # Keep zeros as a separate value (e.g., -999 or median of scaled values)
                if non_zero_mask.sum() > 0:
                    zero_fill_value = np.median(transformed_values[non_zero_mask])
                else:
                    zero_fill_value = 0
                transformed_values[~non_zero_mask] = zero_fill_value
            elif self.zero_encoding == 'median':
                if non_zero_mask.sum() > 0:
                    median_val = np.median(transformed_values[non_zero_mask])
                    transformed_values[~non_zero_mask] = median_val
            
            feature_matrices.append(transformed_values.reshape(-1, 1))
            feature_names.append(f'{col_name}_scaled')
        
        # Create log features for highly skewed sparse data
        if self.create_log_features and col_name in self.sparsity_stats_:
            stats = self.sparsity_stats_[col_name]
            if 'non_zero_skew' in stats and abs(stats['non_zero_skew']) > 2:
                # Create log(1 + x) transformation
                log_values = np.log1p(np.maximum(series.fillna(0), 0))
                feature_matrices.append(log_values.values.reshape(-1, 1))
                feature_names.append(f'{col_name}_log1p')
        
        # Create binned features
        if self.create_binned_features and col_name in self.bin_edges_:
            bin_edges = self.bin_edges_[col_name]
            non_zero_mask = (series != 0) & (series.notnull())
            
            # Create binned version for non-zero values
            binned_values = np.zeros(len(series))
            if non_zero_mask.sum() > 0:
                non_zero_values = series[non_zero_mask].values
                binned_non_zero = np.digitize(non_zero_values, bin_edges) - 1
                binned_values[non_zero_mask] = binned_non_zero
            
            feature_matrices.append(binned_values.reshape(-1, 1))
            feature_names.append(f'{col_name}_binned')
        
        # Combine all features for this column
        if feature_matrices:
            if self.preserve_sparsity:
                combined = hstack([csr_matrix(m) for m in feature_matrices])
            else:
                combined = np.hstack(feature_matrices)
        else:
            # Fallback: just return the original feature
            combined = series.values.reshape(-1, 1)
            feature_names = [col_name]
        
        return combined, feature_names
    
    def _transform_dense_feature(self, series, col_name):
        """Transform dense features with standard scaling"""
        if col_name in self.scalers_:
            values = series.fillna(series.median()).values.reshape(-1, 1)
            scaled_values = self.scalers_[col_name].transform(values)
            return scaled_values, [f'{col_name}_scaled']
        else:
            return series.values.reshape(-1, 1), [col_name]
    
    def _print_analysis_report(self):
        """Print comprehensive sparsity analysis"""
        print(f"\n📊 SPARSITY ANALYSIS REPORT")
        print("="*35)
        
        sparse_count = len(self.sparse_columns_)
        dense_count = len(self.dense_columns_)
        total_count = sparse_count + dense_count
        
        print(f"Total features: {total_count}")
        print(f"Sparse features: {sparse_count} ({sparse_count/total_count*100:.1f}%)")
        print(f"Dense features: {dense_count} ({dense_count/total_count*100:.1f}%)")
        
        if sparse_count > 0:
            print(f"\n🕳️  SPARSE FEATURE DETAILS:")
            for col in self.sparse_columns_[:10]:  # Show first 10
                stats = self.sparsity_stats_[col]
                print(f"   {col}:")
                print(f"     Sparsity: {stats['sparsity_ratio']:.1%}")
                print(f"     Non-zero count: {stats['non_zero_count']}")
                if 'non_zero_mean' in stats:
                    print(f"     Non-zero mean: {stats['non_zero_mean']:.3f}")
                    print(f"     Skewness: {stats.get('non_zero_skew', 0):.2f}")
            
            if len(self.sparse_columns_) > 10:
                print(f"   ... and {len(self.sparse_columns_) - 10} more sparse features")
        
        print(f"\n🔧 TRANSFORMATIONS APPLIED:")
        print(f"   Binary indicators: {self.create_binary_indicators}")
        print(f"   Log features: {self.create_log_features}")
        print(f"   Binned features: {self.create_binned_features}")
        print(f"   Scaling method: {self.scaling_method}")
        print(f"   Zero encoding: {self.zero_encoding}")
    
    def get_feature_names(self):
        """Get names of all engineered features (legacy method)"""
        return self.feature_names_ if hasattr(self, 'feature_names_') else []
    
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
        if hasattr(self, 'feature_names_') and self.feature_names_:
            return np.array(self.feature_names_, dtype=object)
        
        # Fallback if transform hasn't been called yet
        if input_features is not None:
            # Estimate feature names based on input features and configuration
            estimated_names = []
            
            for feature in input_features:
                # Assume all features will be processed as sparse for estimation
                if self.create_binary_indicators:
                    estimated_names.append(f'{feature}_has_value')
                estimated_names.append(f'{feature}_scaled')
                if self.create_log_features:
                    estimated_names.append(f'{feature}_log1p')
                if self.create_binned_features:
                    estimated_names.append(f'{feature}_binned')
            
            return np.array(estimated_names, dtype=object)
        
        # Final fallback
        return np.array([], dtype=object)
    
    def get_sparsity_report(self):
        """Get detailed sparsity statistics"""
        return pd.DataFrame.from_dict(self.sparsity_stats_, orient='index')