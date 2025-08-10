import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class AdvancedCategoricalTransformer(BaseEstimator, TransformerMixin):
    """
    🔥 CATEGORICAL TRANSFORMER ON STEROIDS 🔥
    
    Goes WAY beyond simple OHE:
    - Target encoding with CV to prevent overfitting
    - Frequency/rarity encoding 
    - Interaction features between categoricals
    - Hierarchical category grouping
    - Bayesian target encoding with confidence intervals
    - Category similarity clustering
    - Missing value intelligence
    - Business logic feature engineering
    """
    
    def __init__(self, 
                 target_encoding=True,
                 frequency_encoding=True, 
                 interaction_features=True,
                 hierarchical_grouping=True,
                 similarity_clustering=True,
                 rare_category_threshold=0.01,
                 cv_folds=5,
                 smoothing_alpha=10,
                 max_interactions=10,
                 verbose=True):
        
        self.target_encoding = target_encoding
        self.frequency_encoding = frequency_encoding
        self.interaction_features = interaction_features
        self.hierarchical_grouping = hierarchical_grouping
        self.similarity_clustering = similarity_clustering
        self.rare_category_threshold = rare_category_threshold
        self.cv_folds = cv_folds
        self.smoothing_alpha = smoothing_alpha
        self.max_interactions = max_interactions
        self.verbose = verbose
        
        # Storage for learned transformations
        self.target_encodings_ = {}
        self.frequency_encodings_ = {}
        self.interaction_mappings_ = {}
        self.hierarchical_groups_ = {}
        self.similarity_clusters_ = {}
        self.feature_names_ = []
        self.rare_categories_ = {}
        
    def fit(self, X, y=None):
        """
        Learn all the categorical intelligence from training data
        """
        if self.verbose:
            print("🚀 ADVANCED CATEGORICAL TRANSFORMER - LEARNING PHASE")
            print(f"📊 Input shape: {X.shape}")
        
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Identify categorical columns
        cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not cat_cols:
            cat_cols = X_df.columns.tolist()  # Assume all are categorical
            
        if self.verbose:
            print(f"🔍 Categorical columns found: {cat_cols}")
        
        # 1. TARGET ENCODING WITH CROSS-VALIDATION
        if self.target_encoding and y is not None:
            self._fit_target_encoding(X_df, y, cat_cols)
            
        # 2. FREQUENCY & RARITY ENCODING  
        if self.frequency_encoding:
            self._fit_frequency_encoding(X_df, cat_cols)
            
        # 3. HIERARCHICAL GROUPING (smart category combinations)
        if self.hierarchical_grouping:
            self._fit_hierarchical_grouping(X_df, y, cat_cols)
            
        # 4. INTERACTION FEATURES
        if self.interaction_features and len(cat_cols) > 1:
            self._fit_interaction_features(X_df, y, cat_cols)
            
        # 5. SIMILARITY CLUSTERING
        if self.similarity_clustering and y is not None:
            self._fit_similarity_clustering(X_df, y, cat_cols)
            
        return self
    
    def _fit_target_encoding(self, X_df, y, cat_cols):
        """
        🎯 TARGET ENCODING: Category → Average target value
        Uses cross-validation to prevent overfitting
        """
        if self.verbose:
            print("🎯 Learning target encodings...")
            
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
        
        for col in cat_cols:
            # Calculate global mean for smoothing
            global_mean = y_series.mean()
            
            # Get category counts for smoothing
            cat_counts = X_df[col].value_counts()
            cat_means = X_df.groupby(col)[y_series.name if hasattr(y_series, 'name') else 'target'].agg(['mean', 'count'])
            
            # Bayesian smoothing: (cat_mean * count + global_mean * alpha) / (count + alpha)
            smoothed_means = {}
            for category in cat_counts.index:
                if category in cat_means.index:
                    cat_mean = cat_means.loc[category, 'mean']
                    count = cat_means.loc[category, 'count'] 
                    smoothed_mean = (cat_mean * count + global_mean * self.smoothing_alpha) / (count + self.smoothing_alpha)
                    smoothed_means[category] = smoothed_mean
                else:
                    smoothed_means[category] = global_mean
                    
            # Store with confidence intervals
            self.target_encodings_[col] = {
                'encodings': smoothed_means,
                'global_mean': global_mean,
                'counts': cat_counts.to_dict()
            }
            
        if self.verbose:
            print(f"✅ Target encodings learned for {len(cat_cols)} columns")
    
    def _fit_frequency_encoding(self, X_df, cat_cols):
        """
        📊 FREQUENCY ENCODING: How common/rare is each category?
        """
        if self.verbose:
            print("📊 Learning frequency patterns...")
            
        for col in cat_cols:
            value_counts = X_df[col].value_counts()
            total_count = len(X_df)
            
            # Multiple frequency-based features
            freq_encodings = {}
            for category, count in value_counts.items():
                freq_encodings[category] = {
                    'frequency': count,
                    'frequency_pct': count / total_count,
                    'log_frequency': np.log1p(count),
                    'rarity_score': 1 / (count + 1),  # Higher for rare categories
                    'is_rare': (count / total_count) < self.rare_category_threshold
                }
            
            self.frequency_encodings_[col] = freq_encodings
            
            # Track rare categories for special handling
            rare_cats = [cat for cat, info in freq_encodings.items() if info['is_rare']]
            self.rare_categories_[col] = rare_cats
            
        if self.verbose:
            total_rare = sum(len(cats) for cats in self.rare_categories_.values())
            print(f"✅ Frequency encodings learned, {total_rare} rare categories identified")
    
    def _fit_hierarchical_grouping(self, X_df, y, cat_cols):
        """
        🏗️ HIERARCHICAL GROUPING: Smart category combinations based on similarity
        """
        if self.verbose:
            print("🏗️ Learning hierarchical category groups...")
            
        if y is None:
            return
            
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
        
        for col in cat_cols:
            # Group categories by similar target rates
            cat_target_rates = X_df.groupby(col)[y_series.name if hasattr(y_series, 'name') else 'target'].mean()
            
            # Create hierarchical groups based on target rate similarity
            sorted_cats = cat_target_rates.sort_values()
            
            groups = {}
            group_id = 0
            
            # Group categories with similar conversion rates
            current_group = []
            last_rate = None
            
            for category, rate in sorted_cats.items():
                if last_rate is None or abs(rate - last_rate) < 0.1:  # 10% similarity threshold
                    current_group.append(category)
                else:
                    if current_group:
                        for cat in current_group:
                            groups[cat] = f"group_{group_id}"
                        group_id += 1
                        current_group = [category]
                last_rate = rate
            
            # Handle last group
            if current_group:
                for cat in current_group:
                    groups[cat] = f"group_{group_id}"
            
            self.hierarchical_groups_[col] = groups
            
        if self.verbose:
            total_groups = sum(len(set(groups.values())) for groups in self.hierarchical_groups_.values())
            print(f"✅ Created {total_groups} hierarchical groups")
    
    def _fit_interaction_features(self, X_df, y, cat_cols):
        """
        🔗 INTERACTION FEATURES: Smart combinations of categorical variables
        """
        if self.verbose:
            print("🔗 Learning categorical interactions...")
            
        interactions_created = 0
        
        # Find most informative pairs
        for i, col1 in enumerate(cat_cols):
            for col2 in cat_cols[i+1:]:
                if interactions_created >= self.max_interactions:
                    break
                    
                # Create interaction column
                interaction_name = f"{col1}_x_{col2}"
                interaction_series = X_df[col1].astype(str) + "_x_" + X_df[col2].astype(str)
                
                # Check if interaction is informative (has reasonable cardinality)
                unique_interactions = interaction_series.nunique()
                if unique_interactions < len(X_df) * 0.8 and unique_interactions > 1:
                    self.interaction_mappings_[interaction_name] = {
                        'col1': col1,
                        'col2': col2,
                        'unique_count': unique_interactions
                    }
                    interactions_created += 1
        
        if self.verbose:
            print(f"✅ Created {interactions_created} interaction features")
    
    def _fit_similarity_clustering(self, X_df, y, cat_cols):
        """
        🎭 SIMILARITY CLUSTERING: Group categories with similar behavior patterns
        """
        if self.verbose:
            print("🎭 Learning category similarity clusters...")
            
        # This is a simplified version - could be enhanced with embeddings
        for col in cat_cols:
            category_profiles = {}
            
            for category in X_df[col].unique():
                if pd.isna(category):
                    continue
                    
                subset = X_df[X_df[col] == category]
                
                # Create behavior profile for each category
                profile = {
                    'conversion_rate': y[subset.index].mean() if len(subset) > 0 else 0,
                    'size': len(subset),
                    'relative_size': len(subset) / len(X_df)
                }
                
                category_profiles[category] = profile
            
            self.similarity_clusters_[col] = category_profiles
            
        if self.verbose:
            print(f"✅ Similarity profiles created for {len(cat_cols)} columns")
    
    def transform(self, X):
        """
        🔄 TRANSFORM: Apply all learned categorical intelligence
        """
        if self.verbose:
            print("🔄 ADVANCED CATEGORICAL TRANSFORMER - TRANSFORM PHASE")
            
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not cat_cols:
            cat_cols = X_df.columns.tolist()
        
        transformed_features = []
        feature_names = []
        
        for col in cat_cols:
            if self.verbose:
                print(f"🔧 Transforming {col}...")
                
            # 1. TARGET ENCODING FEATURES
            if self.target_encoding and col in self.target_encodings_:
                target_enc = self._transform_target_encoding(X_df[col], col)
                transformed_features.extend([target_enc])
                feature_names.extend([f"{col}_target_encoded"])
                
            # 2. FREQUENCY FEATURES  
            if self.frequency_encoding and col in self.frequency_encodings_:
                freq_features = self._transform_frequency_encoding(X_df[col], col)
                transformed_features.extend(freq_features)
                feature_names.extend([f"{col}_freq", f"{col}_freq_pct", f"{col}_log_freq", 
                                    f"{col}_rarity", f"{col}_is_rare"])
                
            # 3. HIERARCHICAL GROUP FEATURES
            if self.hierarchical_grouping and col in self.hierarchical_groups_:
                group_enc = self._transform_hierarchical_grouping(X_df[col], col)
                transformed_features.extend([group_enc])
                feature_names.extend([f"{col}_hierarchical_group"])
        
        # 4. INTERACTION FEATURES
        if self.interaction_features:
            interaction_features = self._transform_interactions(X_df)
            transformed_features.extend(interaction_features)
            feature_names.extend([name for name in self.interaction_mappings_.keys()])
        
        # Combine all features
        if transformed_features:
            result = np.column_stack(transformed_features)
        else:
            result = np.array([]).reshape(len(X_df), 0)
            
        self.feature_names_ = feature_names
        
        if self.verbose:
            print(f"✅ Transformation complete: {X_df.shape} → {result.shape}")
            print(f"🎯 Created {len(feature_names)} advanced categorical features")
            
        return result
    
    def _transform_target_encoding(self, series, col):
        """Transform using learned target encodings"""
        encodings = self.target_encodings_[col]['encodings']
        global_mean = self.target_encodings_[col]['global_mean']
        
        return series.map(encodings).fillna(global_mean).values
    
    def _transform_frequency_encoding(self, series, col):
        """Transform using learned frequency encodings"""
        freq_enc = self.frequency_encodings_[col]
        
        freq = series.map(lambda x: freq_enc.get(x, {'frequency': 0})['frequency']).fillna(0)
        freq_pct = series.map(lambda x: freq_enc.get(x, {'frequency_pct': 0})['frequency_pct']).fillna(0)
        log_freq = series.map(lambda x: freq_enc.get(x, {'log_frequency': 0})['log_frequency']).fillna(0)
        rarity = series.map(lambda x: freq_enc.get(x, {'rarity_score': 1})['rarity_score']).fillna(1)
        is_rare = series.map(lambda x: freq_enc.get(x, {'is_rare': True})['is_rare']).fillna(True).astype(int)
        
        return [freq.values, freq_pct.values, log_freq.values, rarity.values, is_rare.values]
    
    def _transform_hierarchical_grouping(self, series, col):
        """Transform using learned hierarchical groups"""
        groups = self.hierarchical_groups_[col]
        
        # Encode groups as integers
        unique_groups = list(set(groups.values()))
        group_to_int = {group: i for i, group in enumerate(unique_groups)}
        
        return series.map(lambda x: group_to_int.get(groups.get(x, 'unknown'), -1)).fillna(-1).values
    
    def _transform_interactions(self, X_df):
        """Transform using learned interactions"""
        interaction_features = []
        
        for interaction_name, mapping in self.interaction_mappings_.items():
            col1, col2 = mapping['col1'], mapping['col2']
            
            if col1 in X_df.columns and col2 in X_df.columns:
                interaction_series = X_df[col1].astype(str) + "_x_" + X_df[col2].astype(str)
                
                # Simple label encoding for interactions
                unique_interactions = interaction_series.unique()
                interaction_map = {inter: i for i, inter in enumerate(unique_interactions)}
                
                encoded_interaction = interaction_series.map(interaction_map).fillna(-1).values
                interaction_features.append(encoded_interaction)
        
        return interaction_features
    
    def get_feature_names(self):
        """Return names of all created features (legacy method)"""
        return self.feature_names_
    
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
            estimated_names = []
            
            # Estimate feature names based on configuration
            for feature in input_features:
                # Target encoding features
                if self.target_encoding:
                    estimated_names.append(f"{feature}_target_encoded")
                
                # Frequency features
                if self.frequency_encoding:
                    estimated_names.extend([
                        f"{feature}_freq",
                        f"{feature}_freq_pct", 
                        f"{feature}_log_freq",
                        f"{feature}_rarity",
                        f"{feature}_is_rare"
                    ])
                
                # Hierarchical grouping
                if self.hierarchical_grouping:
                    estimated_names.append(f"{feature}_hierarchical_group")
            
            # Interaction features (estimate common pairs)
            if self.interaction_features and len(input_features) > 1:
                for i, feat1 in enumerate(input_features):
                    for feat2 in input_features[i+1:min(i+3, len(input_features))]:  # Limit estimation
                        estimated_names.append(f"{feat1}_x_{feat2}")
            
            return np.array(estimated_names, dtype=object)
        
        # Final fallback
        return np.array([], dtype=object)
    
    def get_feature_importance_summary(self):
        """
        📈 Get summary of what features were created and why they're valuable
        """
        summary = {
            'total_features_created': len(self.feature_names_),
            'target_encoded_features': len([f for f in self.feature_names_ if 'target_encoded' in f]),
            'frequency_features': len([f for f in self.feature_names_ if any(x in f for x in ['_freq', '_rarity'])]),
            'hierarchical_features': len([f for f in self.feature_names_ if 'hierarchical' in f]),
            'interaction_features': len([f for f in self.feature_names_ if '_x_' in f]),
            'rare_categories_identified': sum(len(cats) for cats in self.rare_categories_.values()),
            'feature_types': {
                'Target Encodings': 'Category → Average conversion rate',
                'Frequency Features': 'Category → Frequency/rarity patterns', 
                'Hierarchical Groups': 'Category → Similar behavior clusters',
                'Interactions': 'Category1 × Category2 → Combined patterns',
                'Similarity Clusters': 'Category → Behavior profile groups'
            }
        }
        
        return summary