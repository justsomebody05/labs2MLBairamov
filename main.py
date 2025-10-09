import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('future.no_silent_downcasting', True)

df = pd.read_csv("spaceship-titanic/train.csv")
print(df.shape)
print(df.head())

print("\nпропущенные значения:")
print(df.isnull().sum())
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for feature in numeric_features:
    df[feature] = df[feature].fillna(df[feature].median())
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
for feature in categorical_features:
    df[feature] = df[feature].fillna(df[feature].mode()[0] if len(df[feature].mode()) > 0 else 'Unknown')
    
print()
print(df.isnull().sum())

label_encoders = {}
for feature in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature].astype(str))
    label_encoders[feature] = le

features = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP', 
            'Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

X_reg = df[features]
y_reg = df['RoomService']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print(f"размеры выборок для регрессии:")
print(f"обучающая: {X_train_reg.shape}, тестовая: {X_test_reg.shape}")

print()

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

model_reg = LinearRegression()
model_reg.fit(X_train_reg_scaled, y_train_reg)
y_pred_reg = model_reg.predict(X_test_reg_scaled)

mse = mean_squared_error(y_test_reg, y_pred_reg)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)

print(f"\nрезультаты линейной регрессии:")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")

print()

plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_test_reg, y=y_pred_reg, alpha=0.6, color='royalblue', edgecolor='white')
plt.plot([y_test_reg.min(), y_test_reg.max()],
         [y_test_reg.min(), y_test_reg.max()],
         color='red', linestyle='--', lw=2)
plt.title("Сравнение реальных и предсказанных значений (RoomService)")
plt.xlabel("Истинные значения")
plt.ylabel("Предсказанные значения")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

residuals = y_test_reg - y_pred_reg
plt.figure(figsize=(6, 4))
sns.histplot(residuals, kde=True, color='darkorange', bins=30)
plt.title("Распределение ошибок предсказания (residuals)")
plt.xlabel("Ошибка (y_true - y_pred)")
plt.ylabel("Количество наблюдений")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_reg_scaled)
X_test_poly = poly.transform(X_test_reg_scaled)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_reg)
y_pred_poly = poly_model.predict(X_test_poly)

mse_poly = mean_squared_error(y_test_reg, y_pred_poly)
mae_poly = mean_absolute_error(y_test_reg, y_pred_poly)

print(f"\nполиномиальная регрессия (degree=2):")
print(f"MSE: {mse_poly:.2f}")
print(f"MAE: {mae_poly:.2f}")

print()

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_reg_scaled, y_train_reg)
ridge_pred = ridge_model.predict(X_test_reg_scaled)
ridge_mae = mean_absolute_error(y_test_reg, ridge_pred)
ridge_mse = mean_squared_error(y_test_reg, ridge_pred)
ridge_rmse = np.sqrt(ridge_mse)

print(f"\nрезультаты ridge регрессии:")
print(f"ridge MAE:  {ridge_mae:.2f}")
print(f"ridge MSE:  {ridge_mse:.2f}")
print(f"ridge RMSE: {ridge_rmse:.2f}")

print()


lasso = Lasso(alpha=0.1)
lasso.fit(X_train_reg_scaled, y_train_reg)
lasso_pred = lasso.predict(X_test_reg_scaled)
lasso_mae = mean_absolute_error(y_test_reg, lasso_pred)
lasso_mse = mean_squared_error(y_test_reg, lasso_pred)

print(f"\nрезультаты lasso регрессии:")
print(f"lasso MAE:  {lasso_mae:.2f}")
print(f"lasso MSE:  {lasso_mse:.2f}")

print()

X_clf = df[features]
y_clf = df['Transported']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

print(f"размеры выборок для классификации:")
print(f"обучающая: {X_train_clf.shape}, тестовая: {X_test_clf.shape}")

print()

scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

model_clf = LogisticRegression(max_iter=1000, random_state=42)
model_clf.fit(X_train_clf_scaled, y_train_clf)
y_pred_clf = model_clf.predict(X_test_clf_scaled)

accuracy = accuracy_score(y_test_clf, y_pred_clf)

print(f"\nрезультаты классификации:")
print(f"accuracy: {accuracy:.3f}")

print()

print("\nотчёт по классификации:")
print(classification_report(y_test_clf, y_pred_clf))

cm = confusion_matrix(y_test_clf, y_pred_clf)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['False', 'True'], 
            yticklabels=['False', 'True'])
plt.title('Confusion Matrix - Transported Prediction')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()


