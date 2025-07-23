


# Data Preprocessing & Feature Engineering in Python (with Pandas & Scikit-learn Tips)

## 1. Preprocessing

### 1.1 Data Selection
- Ensure the target dataset is large enough to contain relevant patterns but concise for performance.
- Avoid variables that are *too perfectly* correlated with the target.
- Set aside a **hold-out test set** early (e.g., 20%) using:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 1.2 Data Types
- **Master data:** Core entities (customers, products, etc.)
- **Transactional data:** Time-stamped, quantitative events (POS data, transactions, etc.)
- **External data:** Social media, macroeconomic, weather, open data, etc.

### 1.3 Exploratory Data Analysis (EDA)

#### 1.3.1 Visual Analytics
- Boxplots, histograms, scatter plots, correlation plots.
- Libraries: `pandas-profiling`, `ydata-profiling`, `Sweetviz`, `AutoViz`
```python
import pandas_profiling
profile = df.profile_report(title='Pandas Profiling Report')
profile.to_file("output.html")

import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(df_weeks["Week"], df_weeks["Weekly_Sales"])
plt.plot(df_weeks["Week"], df_weeks["MarkDown1"], label='Mkd1', color='red')
plt.plot(df_weeks["Week"], df_weeks["MarkDown2"], label='Mkd2', color = 'yellow')
plt.plot(df_weeks["Week"], df_weeks["MarkDown3"], label='Mkd3', color='green')
plt.plot(df_weeks["Week"], df_weeks["MarkDown4"], label='Mkd4', color='orange')
plt.plot(df_weeks["Week"], df_weeks["MarkDown5"], label='Mkd5', color='purple')
plt.xlabel("Week")
plt.ylabel("Weekly Sales")

###
def scatter(dataset, column):
    plt.figure()
    plt.scatter(data[column] , data['Weekly_Sales'], color = 'turquoise')
    plt.ylabel('Weekly Sales')
    plt.xlabel(column)
scatter(data, 'Fuel_Price')
scatter(data, 'Size')
scatter(data, 'CPI')
scatter(data, 'Type')
scatter(data, 'IsHoliday')
scatter(data, 'Unemployment')
scatter(data, 'Temperature')
scatter(data, 'Store')

#Histogram

#Correlation Plots

#2D All Variable Plots

```

### 1.4 Data Cleaning
- Normalize inconsistent values (e.g., Male/Man/M):
```python
df['gender'] = df['gender'].str.lower().replace({'man': 'male', 'm': 'male'})
```
- Remove future variables (target leakage).
- Drop duplicates:
```python
df.drop_duplicates(inplace=True)
```

### 1.5 Missing Values

#### 1.5.1 Delete
```python
df.dropna(thresh=len(df.columns) - 2, inplace=True)  # Keep rows with at most 2 missing values
```

#### 1.5.2 Replace (Impute)
```python
df.fillna(df.median(numeric_only=True), inplace=True)
```
Advanced methods:
```python
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df)
```

#### 1.5.3 Keep
```python
df['age_missing'] = df['age'].isna().astype(int)

#Fill with NA
data['MarkDown1'].fillna(-500, inplace=True)
```

### 1.6 Outliers
Boxplots,  z-scores
```python
from scipy.stats import zscore
df_outliers = df[(np.abs(zscore(df['age'])) > 3)]
```
- Treatment: cap, impute, categorize, or leave unchanged based on context.

### 1.7 Transformations
#### 1.7.1 Standardization
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### 1.7.2 Normalization
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)
```

#### 1.7.3 Categorization, Binning, Grouping
 Equal-width vs equal-frequency binning
##### 1.7.3.1 Dummyfication
```python
pd.get_dummies(df['category'], drop_first=True)
```

##### 1.7.3.2 Odds-based Grouping
Compute outcome odds per category.
```python
odds = df.groupby('category')['target'].mean()
```

##### 1.7.3.3 Weight of Evidence Encoding
```python
import category_encoders as ce
encoder = ce.WOEEncoder(cols=['category'])
df = encoder.fit_transform(df, y)
```

##### 1.7.3.4 Decision Tree Binning
Fit decision tree with single variable, extract splits.
```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_leaf_nodes=4)
tree.fit(df[['var']], y)
```

##### 1.7.3.5 Embeddings
- For high-cardinality categoricals using `word2vec`-style embeddings.

#### 1.7.4 Mathematical Transformations
4 Mathematical Transformations
- Log, sqrt, Box-Cox, Yeo-Johnson
```python
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X)
```

### 1.8 Interaction Variables
```python
df['interaction'] = df['x1'] * df['x2']
```

---

## 2. Feature Engineering

### 2.1 RFM Features
Recency, Frequency, Monetary
```python
df['recency'] = np.exp(-gamma * df['days_since_last_purchase'])
```

### 2.2 Date Features
Day, month, quarter, is_weekend, is_holiday
```python
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
```

### 2.3 Time Features
- Use **Von Mises distribution** or cyclic encodings:
```python
df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
```

### 2.4 Trends and Deltas
```python
df['delta'] = df['value'] - df['value'].shift(1)
df['trend'] = df['delta'] / df['value'].shift(1)
```

### 2.5 Ordinal Features (Thermometer/Percentile)
```python
df['percentile'] = pd.qcut(df['score'], q=4, labels=False)
```

### 2.6 Relational Data
Join normalized tables using keys (denormalization)
```python
df = df.merge(other_df, on='key_column', how='left')
```

### 2.7 Featurization from Unstructured Data
- Libraries: `featuretools`, `tsfresh`, `autofeat`, `stumpy`, `nixtla`

### 2.8 Feature Selection
Chi-squared, information gain, variance threshold
```python
from sklearn.feature_selection import SelectKBest, chi2
X_new = SelectKBest(chi2, k=10).fit_transform(X, y)
```

Other methods:
```python
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.01)
X_reduced = sel.fit_transform(X)
```

### 2.9 Principal Component Analysis (PCA)
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
```
- Always standardize before PCA.
