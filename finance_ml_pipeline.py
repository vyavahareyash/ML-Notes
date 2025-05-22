import pandas as pd
import numpy as np
import json
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FieldMetadata:
    """Data class to store field metadata from data.json"""
    field_name: str
    data_type: str
    category: str
    depends_on: str
    derivation_logic: str

class DataQualityAnalyzer:
    """Comprehensive data quality analysis component"""
    
    def __init__(self):
        self.dq_issues = {}
        self.field_metadata = {}
        
    def load_metadata(self, metadata_path: str) -> Dict[str, FieldMetadata]:
        """Load field metadata from data.json"""
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.field_metadata = {
                field_name: FieldMetadata(
                    field_name=field_name,
                    data_type=details.get('data_type', ''),
                    category=details.get('category', ''),
                    depends_on=details.get('depends_on', ''),
                    derivation_logic=details.get('derivation_logic', '')
                )
                for field_name, details in metadata.items()
            }
            logger.info(f"Loaded metadata for {len(self.field_metadata)} fields")
            return self.field_metadata
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {}
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality analysis"""
        dq_report = {
            'basic_stats': self._get_basic_stats(df),
            'missing_values': self._analyze_missing_values(df),
            'duplicates': self._analyze_duplicates(df),
            'outliers': self._detect_outliers(df),
            'data_types': self._analyze_data_types(df),
            'value_distributions': self._analyze_distributions(df),
            'business_rules': self._validate_business_rules(df),
            'dependency_validation': self._validate_dependencies(df)
        }
        
        return dq_report
    
    def _get_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset statistics"""
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'dtypes_summary': df.dtypes.value_counts().to_dict()
        }
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values patterns"""
        missing_stats = df.isnull().sum()
        missing_pct = (missing_stats / len(df)) * 100
        
        return {
            'missing_counts': missing_stats.to_dict(),
            'missing_percentages': missing_pct.to_dict(),
            'columns_with_missing': missing_stats[missing_stats > 0].index.tolist(),
            'high_missing_columns': missing_pct[missing_pct > 50].index.tolist()
        }
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate records"""
        total_duplicates = df.duplicated().sum()
        duplicate_pct = (total_duplicates / len(df)) * 100
        
        return {
            'total_duplicates': int(total_duplicates),
            'duplicate_percentage': float(duplicate_pct),
            'duplicate_indices': df[df.duplicated()].index.tolist()
        }
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using multiple methods"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_report = {}
        
        for col in numeric_cols:
            if df[col].nunique() > 10:  # Skip categorical numeric columns
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                iqr_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                
                # Z-score method
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                zscore_outliers = (z_scores > 3).sum()
                
                outliers_report[col] = {
                    'iqr_outliers': int(iqr_outliers),
                    'zscore_outliers': int(zscore_outliers),
                    'outlier_percentage_iqr': float((iqr_outliers / len(df)) * 100),
                    'outlier_percentage_zscore': float((zscore_outliers / len(df)) * 100)
                }
        
        return outliers_report
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types and suggest corrections"""
        type_analysis = {}
        
        for col in df.columns:
            col_data = df[col].dropna()
            
            if col_data.empty:
                type_analysis[col] = {'current_type': str(df[col].dtype), 'suggested_type': 'unknown', 'issues': ['All values are missing']}
                continue
            
            current_type = str(df[col].dtype)
            issues = []
            suggested_type = current_type
            
            # Check for numeric columns stored as strings
            if df[col].dtype == 'object':
                try:
                    pd.to_numeric(col_data)
                    suggested_type = 'numeric'
                    issues.append('Numeric data stored as string')
                except:
                    # Check for date columns
                    try:
                        pd.to_datetime(col_data)
                        suggested_type = 'datetime'
                        issues.append('Date data stored as string')
                    except:
                        pass
            
            # Check for high cardinality categorical columns
            if df[col].dtype == 'object' and df[col].nunique() > 0.5 * len(df):
                issues.append('High cardinality categorical column')
            
            type_analysis[col] = {
                'current_type': current_type,
                'suggested_type': suggested_type,
                'unique_values': int(df[col].nunique()),
                'cardinality_ratio': float(df[col].nunique() / len(df)),
                'issues': issues
            }
        
        return type_analysis
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze value distributions"""
        distribution_analysis = {}
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    distribution_analysis[col] = {
                        'mean': float(col_data.mean()),
                        'median': float(col_data.median()),
                        'std': float(col_data.std()),
                        'skewness': float(col_data.skew()),
                        'kurtosis': float(col_data.kurtosis()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'distribution_type': self._classify_distribution(col_data)
                    }
            else:
                # Categorical analysis
                value_counts = df[col].value_counts()
                distribution_analysis[col] = {
                    'unique_count': int(df[col].nunique()),
                    'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'least_frequent': str(value_counts.index[-1]) if len(value_counts) > 0 else None,
                    'least_frequent_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0
                }
        
        return distribution_analysis
    
    def _classify_distribution(self, data: pd.Series) -> str:
        """Classify the distribution of numeric data"""
        skewness = data.skew()
        kurtosis = data.kurtosis()
        
        if abs(skewness) < 0.5:
            if abs(kurtosis) < 0.5:
                return 'normal'
            elif kurtosis > 0.5:
                return 'leptokurtic'
            else:
                return 'platykurtic'
        elif skewness > 0.5:
            return 'right_skewed'
        else:
            return 'left_skewed'
    
    def _validate_business_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate business rules based on metadata"""
        business_rules_validation = {}
        
        # Example business rules for finance data
        rules = {
            'non_negative_amounts': ['amount', 'balance', 'value'],
            'valid_dates': ['date', 'timestamp', 'created_at'],
            'required_fields': ['id', 'account_id', 'transaction_id']
        }
        
        for rule_name, pattern_fields in rules.items():
            matching_columns = [col for col in df.columns if any(pattern in col.lower() for pattern in pattern_fields)]
            
            if matching_columns:
                violations = {}
                for col in matching_columns:
                    if rule_name == 'non_negative_amounts' and df[col].dtype in ['int64', 'float64']:
                        violations[col] = int((df[col] < 0).sum())
                    elif rule_name == 'required_fields':
                        violations[col] = int(df[col].isnull().sum())
                
                business_rules_validation[rule_name] = violations
        
        return business_rules_validation
    
    def _validate_dependencies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate field dependencies based on metadata"""
        dependency_validation = {}
        
        for field_name, metadata in self.field_metadata.items():
            if field_name in df.columns and metadata.depends_on:
                dependencies = [dep.strip() for dep in metadata.depends_on.split(',')]
                missing_deps = [dep for dep in dependencies if dep not in df.columns]
                
                if missing_deps:
                    dependency_validation[field_name] = {
                        'missing_dependencies': missing_deps,
                        'available_dependencies': [dep for dep in dependencies if dep in df.columns]
                    }
        
        return dependency_validation

class AdvancedEDA:
    """Advanced Exploratory Data Analysis component"""
    
    def __init__(self):
        self.insights = {}
        
    def perform_eda(self, df: pd.DataFrame, target_columns: List[str] = None) -> Dict[str, Any]:
        """Perform comprehensive EDA"""
        eda_results = {
            'univariate_analysis': self._univariate_analysis(df),
            'bivariate_analysis': self._bivariate_analysis(df, target_columns),
            'correlation_analysis': self._correlation_analysis(df),
            'feature_importance': self._feature_importance_analysis(df, target_columns),
            'insights_summary': self._generate_insights(df, target_columns)
        }
        
        return eda_results
    
    def _univariate_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detailed univariate analysis"""
        univariate_results = {}
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                col_data = df[col].dropna()
                univariate_results[col] = {
                    'type': 'numeric',
                    'statistics': {
                        'count': len(col_data),
                        'mean': float(col_data.mean()) if len(col_data) > 0 else None,
                        'median': float(col_data.median()) if len(col_data) > 0 else None,
                        'mode': float(col_data.mode().iloc[0]) if len(col_data.mode()) > 0 else None,
                        'std': float(col_data.std()) if len(col_data) > 0 else None,
                        'variance': float(col_data.var()) if len(col_data) > 0 else None,
                        'range': float(col_data.max() - col_data.min()) if len(col_data) > 0 else None,
                        'iqr': float(col_data.quantile(0.75) - col_data.quantile(0.25)) if len(col_data) > 0 else None,
                        'cv': float(col_data.std() / col_data.mean()) if len(col_data) > 0 and col_data.mean() != 0 else None
                    }
                }
            else:
                value_counts = df[col].value_counts()
                univariate_results[col] = {
                    'type': 'categorical',
                    'statistics': {
                        'unique_count': int(df[col].nunique()),
                        'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                        'most_frequent_freq': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                        'entropy': float(-sum((value_counts / len(df)) * np.log2(value_counts / len(df)))),
                        'concentration_ratio': float(value_counts.iloc[0] / len(df)) if len(value_counts) > 0 else 0
                    }
                }
        
        return univariate_results
    
    def _bivariate_analysis(self, df: pd.DataFrame, target_columns: List[str] = None) -> Dict[str, Any]:
        """Bivariate analysis focusing on target relationships"""
        if not target_columns:
            return {}
        
        bivariate_results = {}
        
        for target_col in target_columns:
            if target_col not in df.columns:
                continue
                
            bivariate_results[target_col] = {}
            
            for col in df.columns:
                if col == target_col:
                    continue
                
                # Numeric vs Numeric
                if df[col].dtype in ['int64', 'float64'] and df[target_col].dtype in ['int64', 'float64']:
                    correlation = self._calculate_correlation(df[col], df[target_col])
                    bivariate_results[target_col][col] = {
                        'type': 'numeric_vs_numeric',
                        'pearson_correlation': correlation['pearson'],
                        'spearman_correlation': correlation['spearman'],
                        'mutual_information': correlation['mutual_info']
                    }
                
                # Categorical vs Numeric
                elif df[col].dtype == 'object' and df[target_col].dtype in ['int64', 'float64']:
                    bivariate_results[target_col][col] = {
                        'type': 'categorical_vs_numeric',
                        'anova_f_stat': self._anova_test(df[col], df[target_col]),
                        'group_means': df.groupby(col)[target_col].mean().to_dict()
                    }
                
                # Categorical vs Categorical
                elif df[col].dtype == 'object' and df[target_col].dtype == 'object':
                    contingency_table = pd.crosstab(df[col], df[target_col])
                    chi2, p_value = chi2_contingency(contingency_table)[:2]
                    bivariate_results[target_col][col] = {
                        'type': 'categorical_vs_categorical',
                        'chi2_statistic': float(chi2),
                        'chi2_p_value': float(p_value),
                        'cramers_v': self._cramers_v(contingency_table)
                    }
        
        return bivariate_results
    
    def _correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive correlation analysis"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {}
        
        pearson_corr = numeric_df.corr(method='pearson')
        spearman_corr = numeric_df.corr(method='spearman')
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(pearson_corr.columns)):
            for j in range(i+1, len(pearson_corr.columns)):
                col1, col2 = pearson_corr.columns[i], pearson_corr.columns[j]
                corr_value = pearson_corr.iloc[i, j]
                if abs(corr_value) > 0.7:  # Threshold for high correlation
                    high_corr_pairs.append({
                        'feature1': col1,
                        'feature2': col2,
                        'correlation': float(corr_value)
                    })
        
        return {
            'pearson_correlation_matrix': pearson_corr.to_dict(),
            'spearman_correlation_matrix': spearman_corr.to_dict(),
            'highly_correlated_pairs': high_corr_pairs
        }
    
    def _feature_importance_analysis(self, df: pd.DataFrame, target_columns: List[str] = None) -> Dict[str, Any]:
        """Analyze feature importance for target variables"""
        if not target_columns:
            return {}
        
        feature_importance = {}
        
        for target_col in target_columns:
            if target_col not in df.columns:
                continue
            
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Remove non-numeric columns for now (can be enhanced with encoding)
            X_numeric = X.select_dtypes(include=[np.number])
            
            if X_numeric.empty:
                continue
            
            # Handle missing values
            X_clean = X_numeric.fillna(X_numeric.median())
            y_clean = y.dropna()
            X_clean = X_clean.loc[y_clean.index]
            
            if len(X_clean) == 0:
                continue
            
            # Determine problem type
            if y_clean.dtype == 'object' or y_clean.nunique() < 10:
                # Classification
                if y_clean.dtype == 'object':
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y_clean)
                else:
                    y_encoded = y_clean
                
                importance_scores = mutual_info_classif(X_clean, y_encoded, random_state=42)
            else:
                # Regression
                importance_scores = mutual_info_regression(X_clean, y_clean, random_state=42)
            
            feature_importance[target_col] = {
                col: float(score) for col, score in zip(X_clean.columns, importance_scores)
            }
        
        return feature_importance
    
    def _generate_insights(self, df: pd.DataFrame, target_columns: List[str] = None) -> List[str]:
        """Generate actionable insights from the analysis"""
        insights = []
        
        # Data quality insights
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > 30]
        if not high_missing.empty:
            insights.append(f"High missing data detected in columns: {list(high_missing.index)}. Consider imputation or removal.")
        
        # Imbalance insights
        if target_columns:
            for target_col in target_columns:
                if target_col in df.columns:
                    if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
                        value_counts = df[target_col].value_counts()
                        imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[-1] if len(value_counts) > 1 else 1
                        if imbalance_ratio > 5:
                            insights.append(f"Target variable '{target_col}' is highly imbalanced (ratio: {imbalance_ratio:.2f}). Consider resampling techniques.")
        
        # Outlier insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].nunique() > 10:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
                if outlier_count > len(df) * 0.1:  # More than 10% outliers
                    insights.append(f"Column '{col}' has {outlier_count} outliers ({outlier_count/len(df)*100:.1f}%). Consider outlier treatment.")
        
        return insights
    
    def _calculate_correlation(self, x: pd.Series, y: pd.Series) -> Dict[str, float]:
        """Calculate various correlation measures"""
        # Remove missing values
        valid_mask = ~(x.isnull() | y.isnull())
        x_clean, y_clean = x[valid_mask], y[valid_mask]
        
        if len(x_clean) < 2:
            return {'pearson': 0.0, 'spearman': 0.0, 'mutual_info': 0.0}
        
        pearson_corr, _ = pearsonr(x_clean, y_clean)
        spearman_corr, _ = spearmanr(x_clean, y_clean)
        
        # Mutual information
        try:
            if y_clean.nunique() < 10:  # Categorical target
                y_encoded = LabelEncoder().fit_transform(y_clean.astype(str))
                mi_score = mutual_info_classif(x_clean.values.reshape(-1, 1), y_encoded, random_state=42)[0]
            else:  # Continuous target
                mi_score = mutual_info_regression(x_clean.values.reshape(-1, 1), y_clean, random_state=42)[0]
        except:
            mi_score = 0.0
        
        return {
            'pearson': float(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
            'spearman': float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
            'mutual_info': float(mi_score)
        }
    
    def _anova_test(self, categorical: pd.Series, numeric: pd.Series) -> float:
        """Perform ANOVA test for categorical vs numeric relationship"""
        try:
            groups = [group for name, group in numeric.groupby(categorical) if len(group) > 1]
            if len(groups) < 2:
                return 0.0
            f_stat, _ = stats.f_oneway(*groups)
            return float(f_stat) if not np.isnan(f_stat) else 0.0
        except:
            return 0.0
    
    def _cramers_v(self, contingency_table: pd.DataFrame) -> float:
        """Calculate Cramer's V for categorical association"""
        try:
            chi2 = chi2_contingency(contingency_table)[0]
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape) - 1
            return float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0
        except:
            return 0.0

class AdaptivePreprocessor:
    """Intelligent preprocessing component that adapts to data characteristics"""
    
    def __init__(self, metadata: Dict[str, FieldMetadata] = None):
        self.metadata = metadata or {}
        self.transformers = {}
        self.feature_names = []
        self.target_encoders = {}
        self.is_fitted = False
        
    def fit_transform(self, df: pd.DataFrame, target_columns: List[str] = None) -> pd.DataFrame:
        """Fit transformers and transform the data"""
        return self.fit(df, target_columns).transform(df)
    
    def fit(self, df: pd.DataFrame, target_columns: List[str] = None):
        """Fit preprocessing transformers based on data characteristics"""
        target_columns = target_columns or []
        
        # Identify feature columns (exclude targets)
        feature_columns = [col for col in df.columns if col not in target_columns]
        
        for col in feature_columns:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            # Determine preprocessing strategy based on data characteristics
            if df[col].dtype in ['int64', 'float64']:
                self._fit_numeric_transformer(col, df[col])
            else:
                self._fit_categorical_transformer(col, df[col])
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using fitted transformers"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        transformed_data = {}
        
        for col in df.columns:
            if col in self.transformers:
                transformer_type, transformer = self.transformers[col]
                
                if transformer_type == 'numeric':
                    # Handle missing values first
                    col_data = df[[col]].copy()
                    if col_data[col].isnull().any():
                        col_data[col] = transformer['imputer'].transform(col_data[[col]]).flatten()
                    
                    # Apply scaling
                    scaled_data = transformer['scaler'].transform(col_data[[col]]).flatten()
                    transformed_data[col] = scaled_data
                
                elif transformer_type == 'categorical':
                    col_data = df[col].fillna('missing')  # Handle missing values
                    
                    if 'encoder' in transformer:
                        try:
                            encoded_data = transformer['encoder'].transform(col_data)
                            if hasattr(transformer['encoder'], 'get_feature_names_out'):
                                feature_names = transformer['encoder'].get_feature_names_out([col])
                                for i, fname in enumerate(feature_names):
                                    transformed_data[fname] = encoded_data[:, i]
                            else:
                                transformed_data[col] = encoded_data
                        except:
                            # Fallback for unknown categories
                            transformed_data[col] = [0] * len(df)
                    else:
                        transformed_data[col] = col_data
            else:
                # Column not in transformers, keep as is
                transformed_data[col] = df[col]
        
        return pd.DataFrame(transformed_data, index=df.index)
    
    def _fit_numeric_transformer(self, col: str, data: pd.Series):
        """Fit transformers for numeric columns"""
        # Missing value imputation strategy
        missing_pct = data.isnull().sum() / len(data)
        
        if missing_pct > 0:
            if missing_pct < 0.3:
                imputer = SimpleImputer(strategy='median')
            else:
                imputer = KNNImputer(n_neighbors=5)
        else:
            imputer = SimpleImputer(strategy='median')  # Keep for consistency
        
        # Fit imputer
        imputer.fit(data.values.reshape(-1, 1))
        
        # Determine scaling strategy based on distribution
        clean_data = data.dropna()
        if len(clean_data) > 0:
            skewness = abs(clean_data.skew())
            has_outliers = self._detect_outliers_in_series(clean_data)
            
            if has_outliers or skewness > 2:
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
        else:
            scaler = StandardScaler()
        
        # Fit scaler on imputed data
        imputed_data = imputer.transform(data.values.reshape(-1, 1))
        scaler.fit(imputed_data)
        
        self.transformers[col] = ('numeric', {
            'imputer': imputer,
            'scaler': scaler
        })
    
    def _fit_categorical_transformer(self, col: str, data: pd.Series):
        """Fit transformers for categorical columns"""
        unique_count = data.nunique()
        
        # Handle missing values by treating as separate category
        data_filled = data.fillna('missing')
        
        if unique_count <= 2:
            # Binary encoding for binary categories
            encoder = LabelEncoder()
            encoder.fit(data_filled)
            self.transformers[col] = ('categorical', {'encoder': encoder})
        
        elif unique_count <= 10:
            # One-hot encoding for low cardinality
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoder.fit(data_filled.values.reshape(-1, 1))
            self.transformers[col] = ('categorical', {'encoder': encoder})
        
        else:
            # High cardinality - use frequency encoding or target encoding
            freq_map = data_filled.value_counts().to_dict()
            self.transformers[col] = ('categorical', {'freq_map': freq_map})
    
    def _detect_outliers_in_series(self, data: pd.Series) -> bool:
        """Detect if series has significant outliers"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = ((data < lower_bound) | (data > upper_bound)).sum()
        return outlier_count > len(data) * 0.1  # More than 10% outliers

class ModelBuilder:
    """Intelligent model building component for multiple target variables"""
    
    def __init__(self, max_models: int = 4):
        self.max_models = max_models
        self.models = {}
        self.model_performances = {}
        self.feature_importance = {}
        
    def build_models(self, X: pd.DataFrame, targets: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Build models for multiple target variables"""
        results = {}
        
        for target_name, y in targets.items():
            logger.info(f"Building models for target: {target_name}")
            
            # Determine problem type
            problem_type = self._determine_problem_type(y)
            
            # Prepare data
            X_clean, y_clean = self._prepare_data(X, y)
            
            if len(X_clean) == 0:
                logger.warning(f"No clean data available for target {target_name}")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=0.2, random_state=42, 
                stratify=y_clean if problem_type != 'regression' else None
            )
            
            # Get candidate models
            candidate_models = self._get_candidate_models(problem_type)
            
            # Train and evaluate models
            model_results = self._train_and_evaluate_models(
                candidate_models, X_train, X_test, y_train, y_test, problem_type
            )
            
            # Select best models (up to max_models)
            best_models = sorted(model_results.items(), 
                               key=lambda x: x[1]['cv_score'], reverse=True)[:self.max_models]
            
            # Train final models on all data
            final_models = {}
            for model_name, model_info in best_models:
                final_model = model_info['model']
                final_model.fit(X_clean, y_clean)
                final_models[model_name] = {
                    'model': final_model,
                    'performance': model_info,
                    'feature_importance': self._get_feature_importance(final_model, X_clean.columns)
                }
            
            results[target_name] = {
                'problem_type': problem_type,
                'models': final_models,
                'data_shape': X_clean.shape,
                'class_distribution': y_clean.value_counts().to_dict() if problem_type != 'regression' else None
            }
            
        return results
    
    def _determine_problem_type(self, y: pd.Series) -> str:
        """Determine if the problem is classification or regression"""
        if y.dtype == 'object':
            return 'classification'
        elif y.nunique() <= 10:
            return 'classification'
        else:
            return 'regression'
    
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Clean and prepare data for modeling"""
        # Remove rows where target is missing
        valid_indices = ~y.isnull()
        X_clean = X.loc[valid_indices].copy()
        y_clean = y.loc[valid_indices].copy()
        
        # Remove columns with too many missing values
        missing_threshold = 0.5
        X_clean = X_clean.loc[:, X_clean.isnull().mean() < missing_threshold]
        
        # Handle remaining missing values in features
        for col in X_clean.columns:
            if X_clean[col].dtype in ['int64', 'float64']:
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
            else:
                X_clean[col] = X_clean[col].fillna(X_clean[col].mode().iloc[0] if not X_clean[col].mode().empty else 'unknown')
        
        return X_clean, y_clean
    
    def _get_candidate_models(self, problem_type: str) -> Dict[str, Any]:
        """Get candidate models based on problem type"""
        if problem_type == 'classification':
            return {
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
                'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
                'gradient_boosting': GradientBoostingClassifier(random_state=42),
                'svm': SVC(random_state=42, probability=True)
            }
        else:  # regression
            return {
                'linear_regression': Ridge(random_state=42),
                'random_forest': RandomForestRegressor(random_state=42, n_estimators=100),
                'gradient_boosting': GradientBoostingRegressor(random_state=42),
                'svr': SVR()
            }
    
    def _train_and_evaluate_models(self, models: Dict[str, Any], X_train: pd.DataFrame, 
                                 X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                                 problem_type: str) -> Dict[str, Any]:
        """Train and evaluate all candidate models"""
        results = {}
        
        # Determine cross-validation strategy
        if problem_type == 'classification':
            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'roc_auc' if len(y_train.unique()) == 2 else 'accuracy'
        else:
            cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'r2'
        
        for model_name, model in models.items():
            try:
                logger.info(f"Training {model_name}...")
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring=scoring)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Train on training set and evaluate on test set
                model.fit(X_train, y_train)
                
                if problem_type == 'classification':
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                    
                    test_metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'classification_report': classification_report(y_test, y_pred, output_dict=True)
                    }
                    
                    if y_pred_proba is not None and len(y_test.unique()) == 2:
                        test_metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                
                else:  # regression
                    y_pred = model.predict(X_test)
                    test_metrics = {
                        'r2': r2_score(y_test, y_pred),
                        'mse': mean_squared_error(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'mae': mean_absolute_error(y_test, y_pred)
                    }
                
                results[model_name] = {
                    'model': model,
                    'cv_score': cv_mean,
                    'cv_std': cv_std,
                    'test_metrics': test_metrics
                }
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        return results
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from trained model"""
        importance_dict = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_).flatten()
            else:
                # No importance available
                return {}
            
            # Normalize importances
            if importances.sum() > 0:
                importances = importances / importances.sum()
            
            importance_dict = {
                feature: float(importance) 
                for feature, importance in zip(feature_names, importances)
            }
            
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), 
                                       key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error extracting feature importance: {e}")
        
        return importance_dict

class FinanceMLPipeline:
    """Main pipeline orchestrator for finance ML tasks"""
    
    def __init__(self, metadata_path: str = None):
        self.metadata_path = metadata_path
        self.dq_analyzer = DataQualityAnalyzer()
        self.eda_analyzer = AdvancedEDA()
        self.preprocessor = AdaptivePreprocessor()
        self.model_builder = ModelBuilder()
        
        # Pipeline results storage
        self.results = {
            'metadata': {},
            'data_quality': {},
            'eda': {},
            'preprocessing': {},
            'modeling': {}
        }
        
    def run_pipeline(self, df: pd.DataFrame, target_columns: List[str], 
                    metadata_path: str = None) -> Dict[str, Any]:
        """Execute the complete ML pipeline"""
        
        logger.info("Starting Finance ML Pipeline...")
        
        # Step 1: Load metadata if provided
        if metadata_path or self.metadata_path:
            metadata = self.dq_analyzer.load_metadata(metadata_path or self.metadata_path)
            self.results['metadata'] = {k: v.__dict__ for k, v in metadata.items()}
            self.preprocessor.metadata = metadata
        
        # Step 2: Data Quality Analysis
        logger.info("Performing Data Quality Analysis...")
        dq_results = self.dq_analyzer.analyze_data_quality(df)
        self.results['data_quality'] = dq_results
        
        # Step 3: Exploratory Data Analysis
        logger.info("Performing Exploratory Data Analysis...")
        eda_results = self.eda_analyzer.perform_eda(df, target_columns)
        self.results['eda'] = eda_results
        
        # Step 4: Data Preprocessing
        logger.info("Preprocessing data...")
        df_processed = self.preprocessor.fit_transform(df, target_columns)
        self.results['preprocessing'] = {
            'original_shape': df.shape,
            'processed_shape': df_processed.shape,
            'transformers': list(self.preprocessor.transformers.keys())
        }
        
        # Step 5: Model Building
        logger.info("Building predictive models...")
        X = df_processed.drop(columns=target_columns, errors='ignore')
        targets = {col: df[col] for col in target_columns if col in df.columns}
        
        modeling_results = self.model_builder.build_models(X, targets)
        self.results['modeling'] = modeling_results
        
        # Step 6: Generate Summary Report
        summary_report = self._generate_summary_report()
        self.results['summary'] = summary_report
        
        logger.info("Pipeline execution completed!")
        return self.results
    
    def _generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        summary = {
            'data_overview': {
                'total_records': self.results['data_quality']['basic_stats']['shape'][0],
                'total_features': self.results['data_quality']['basic_stats']['shape'][1],
                'memory_usage_mb': self.results['data_quality']['basic_stats']['memory_usage'] / (1024*1024)
            },
            'data_quality_summary': {
                'columns_with_missing_data': len(self.results['data_quality']['missing_values']['columns_with_missing']),
                'high_missing_columns': self.results['data_quality']['missing_values']['high_missing_columns'],
                'duplicate_records': self.results['data_quality']['duplicates']['total_duplicates'],
                'outlier_columns': list(self.results['data_quality']['outliers'].keys())
            },
            'model_performance': {},
            'key_insights': self.results['eda'].get('insights_summary', []),
            'recommendations': self._generate_recommendations()
        }
        
        # Add model performance summary
        for target, target_results in self.results['modeling'].items():
            best_model = None
            best_score = -float('inf')
            
            for model_name, model_info in target_results['models'].items():
                cv_score = model_info['performance']['cv_score']
                if cv_score > best_score:
                    best_score = cv_score
                    best_model = model_name
            
            summary['model_performance'][target] = {
                'best_model': best_model,
                'best_cv_score': best_score,
                'problem_type': target_results['problem_type'],
                'total_models_trained': len(target_results['models'])
            }
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Data quality recommendations
        dq_results = self.results['data_quality']
        
        if dq_results['missing_values']['high_missing_columns']:
            recommendations.append(
                f"Consider removing or imputing columns with high missing data: "
                f"{', '.join(dq_results['missing_values']['high_missing_columns'])}"
            )
        
        if dq_results['duplicates']['total_duplicates'] > 0:
            recommendations.append(
                f"Remove {dq_results['duplicates']['total_duplicates']} duplicate records to improve data quality"
            )
        
        # Model performance recommendations
        for target, perf in self.results['summary']['model_performance'].items():
            if perf['best_cv_score'] < 0.7:  # Threshold for good performance
                recommendations.append(
                    f"Model performance for {target} is suboptimal ({perf['best_cv_score']:.3f}). "
                    f"Consider feature engineering or alternative algorithms."
                )
        
        # Feature importance recommendations
        for target, target_results in self.results['modeling'].items():
            for model_name, model_info in target_results['models'].items():
                feature_importance = model_info.get('feature_importance', {})
                if feature_importance:
                    top_features = list(feature_importance.keys())[:3]
                    recommendations.append(
                        f"For {target}, focus on top features: {', '.join(top_features)}"
                    )
                break  # Only need one model's feature importance per target
        
        return recommendations
    
    def predict(self, new_data: pd.DataFrame, target: str, model_name: str = None) -> np.ndarray:
        """Make predictions on new data"""
        if target not in self.results['modeling']:
            raise ValueError(f"No trained models available for target: {target}")
        
        # Preprocess new data
        new_data_processed = self.preprocessor.transform(new_data)
        X_new = new_data_processed.drop(columns=[target], errors='ignore')
        
        # Select model
        target_models = self.results['modeling'][target]['models']
        if model_name and model_name in target_models:
            model = target_models[model_name]['model']
        else:
            # Use best performing model
            best_model_name = max(target_models.keys(), 
                                key=lambda x: target_models[x]['performance']['cv_score'])
            model = target_models[best_model_name]['model']
        
        # Make predictions
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_new)
        else:
            return model.predict(X_new)
    
    def get_model_explanation(self, target: str, model_name: str = None) -> Dict[str, Any]:
        """Get detailed explanation of a specific model"""
        if target not in self.results['modeling']:
            raise ValueError(f"No trained models available for target: {target}")
        
        target_models = self.results['modeling'][target]['models']
        if model_name and model_name in target_models:
            model_info = target_models[model_name]
        else:
            # Use best performing model
            best_model_name = max(target_models.keys(), 
                                key=lambda x: target_models[x]['performance']['cv_score'])
            model_info = target_models[best_model_name]
        
        return {
            'model_type': type(model_info['model']).__name__,
            'performance_metrics': model_info['performance']['test_metrics'],
            'cross_validation_score': model_info['performance']['cv_score'],
            'feature_importance': model_info['feature_importance'],
            'problem_type': self.results['modeling'][target]['problem_type']
        }

# Example usage and testing
def main():
    """Example usage of the Finance ML Pipeline"""
    
    # Initialize pipeline
    pipeline = FinanceMLPipeline()
    
    # Example with synthetic finance data (replace with your actual data loading)
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic finance dataset
    synthetic_data = {
        'account_id': np.random.randint(1000, 9999, n_samples),
        'balance_amount': np.random.normal(10000, 5000, n_samples),
        'transaction_count': np.random.poisson(5, n_samples),
        'pnl_attribution': np.random.normal(100, 500, n_samples),
        'cash_position': np.random.normal(50000, 20000, n_samples),
        'mark_to_market': np.random.normal(10000, 3000, n_samples),
        'risk_score': np.random.uniform(0, 1, n_samples),
        'sector': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Energy'], n_samples),
        'region': np.random.choice(['US', 'EU', 'APAC'], n_samples),
    }
    
    # Add some missing values and outliers to simulate real data issues
    synthetic_data['balance_amount'][np.random.choice(n_samples, 50, replace=False)] = np.nan
    synthetic_data['pnl_attribution'][np.random.choice(n_samples, 20, replace=False)] = np.random.normal(0, 10000, 20)  # Outliers
    
    # Create target variables (simulating the ones mentioned in the problem)
    synthetic_data['tv_mark_diff_variance_break_flag'] = (np.random.random(n_samples) > 0.8).astype(int)
    synthetic_data['sum_of_scallops_check_break_flag'] = (np.random.random(n_samples) > 0.85).astype(int)
    synthetic_data['overall_flag'] = ((synthetic_data['tv_mark_diff_variance_break_flag'] == 1) | 
                                     (synthetic_data['sum_of_scallops_check_break_flag'] == 1)).astype(int)
    synthetic_data['ggl_cash_tc_adj_flag'] = (np.random.random(n_samples) > 0.9).astype(int)
    
    df = pd.DataFrame(synthetic_data)
    
    # Define target columns as specified in the problem
    target_columns = [
        'tv_mark_diff_variance_break_flag',
        'sum_of_scallops_check_break_flag', 
        'overall_flag',
        'ggl_cash_tc_adj_flag'
    ]
    
    # Run the pipeline
    try:
        results = pipeline.run_pipeline(df, target_columns)
        
        # Print summary results
        print("="*80)
        print("FINANCE ML PIPELINE RESULTS SUMMARY")
        print("="*80)
        
        print(f"\nDataset Overview:")
        print(f"- Total Records: {results['summary']['data_overview']['total_records']}")
        print(f"- Total Features: {results['summary']['data_overview']['total_features']}")
        print(f"- Memory Usage: {results['summary']['data_overview']['memory_usage_mb']:.2f} MB")
        
        print(f"\nData Quality Issues:")
        print(f"- Columns with Missing Data: {results['summary']['data_quality_summary']['columns_with_missing_data']}")
        print(f"- High Missing Columns: {results['summary']['data_quality_summary']['high_missing_columns']}")
        print(f"- Duplicate Records: {results['summary']['data_quality_summary']['duplicate_records']}")
        
        print(f"\nModel Performance:")
        for target, perf in results['summary']['model_performance'].items():
            print(f"- {target}: {perf['best_model']} (CV Score: {perf['best_cv_score']:.3f})")
        
        print(f"\nKey Recommendations:")
        for i, rec in enumerate(results['summary']['recommendations'], 1):
            print(f"{i}. {rec}")
        
        # Example prediction
        sample_data = df.iloc[:5].copy()  # Take first 5 rows for prediction
        predictions = pipeline.predict(sample_data, 'overall_flag')
        print(f"\nSample Predictions for 'overall_flag': {predictions}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

if __name__ == "__main__":
    main()