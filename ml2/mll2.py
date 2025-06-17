import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error

data = pd.read_csv('AmesHousing.csv')
print(data.head())
print(data.info())
print(data.describe())

data.fillna(0, inplace=True)

numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
data_numeric = data[numeric_cols]

cols_to_drop = ['Order', 'PID', 'Mo Sold', 'Yr Sold']
data_numeric = data_numeric.drop(cols_to_drop, axis=1, errors='ignore')

corr_matrix = data_numeric.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
data_numeric = data_numeric.drop(to_drop, axis=1)

X = data_numeric.drop('SalePrice', axis=1)
y = data_numeric['SalePrice']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], y, c=y, cmap='viridis', alpha=0.6)
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('SalePrice')
plt.title('3D Visualization of Housing Data')
plt.colorbar(scatter, label='SalePrice')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE (Linear Regression): {rmse:.2f}')

alphas = np.logspace(-4, 4, 100)
rmse_values = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_values.append(rmse)

plt.figure(figsize=(10, 6))
plt.semilogx(alphas, rmse_values)
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('RMSE')
plt.title('RMSE vs. Regularization Strength (Lasso)')
plt.grid(True)
plt.show()

best_alpha = alphas[np.argmin(rmse_values)]
print(f'Best alpha: {best_alpha:.4f}')

lasso = Lasso(alpha=best_alpha, max_iter=10000)
lasso.fit(X_train, y_train)
coef = pd.Series(lasso.coef_, index=X.columns)
important_features = coef[coef != 0].sort_values(ascending=False)

print("\nMost important features:")
print(important_features.head(10))

most_important_feature = important_features.index[0]
print(f"\nThe most important feature is: {most_important_feature}")
