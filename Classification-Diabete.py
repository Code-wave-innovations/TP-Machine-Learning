#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd

df = pd.read_csv("./dataset/diabetes.csv")

# Aperçu des premières lignes
print("Aperçu du dataset :")
print(df.head())

# Dimensions
print(f"\nDimensions : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# Noms des colonnes
print("\nColonnes :")
print(df.columns.tolist())


# In[14]:


import numpy as np

cols_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].replace(0, np.nan)

print("Valeurs manquantes après remplacement des zéros :")
print(df.isnull().sum())

df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].fillna(df[cols_with_invalid_zeros].median())

print("\nValeurs manquantes après imputation :")
print(df.isnull().sum())

from sklearn.preprocessing import StandardScaler

features = df.drop(columns=['Outcome'])
target = df['Outcome']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
df_scaled['Outcome'] = target.reset_index(drop=True)

# Aperçu
print("\nAperçu du dataset standardisé :")
print(df_scaled.head())


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns

# Statistiques descriptives globales
print("Statistiques descriptives globales :")
print(df_scaled.describe())

plt.figure(figsize=(6,4))
sns.countplot(x='Outcome', data=df_scaled)
plt.title("Distribution of Diabetes Outcome (0 = No, 1 = Yes)")
plt.xlabel("Diabetes Outcome")
plt.ylabel("Count")
plt.show()

# Statistiques descriptives par classe Outcome
print("\nStatistiques descriptives par classe Outcome :")
print(df_scaled.groupby('Outcome').describe().T)

features = df_scaled.columns.drop('Outcome')

plt.figure(figsize=(15, 12))
for i, col in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data=df_scaled, x=col, hue='Outcome', kde=True, element="step", stat="density")
    plt.title(f'Distribution of {col} by Outcome')
plt.tight_layout()
plt.show()

# Matrice de corrélation des features
plt.figure(figsize=(12, 10))
corr = df_scaled.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation matrix")
plt.show()


# In[16]:


from sklearn.model_selection import train_test_split

# Séparation des features et de la cible
X = df_scaled.drop(columns=['Outcome'])
y = df_scaled['Outcome']

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

# Vérification des tailles
print(f"Total samples: {len(df_scaled)}")
print(f"Train size: {len(X_train)} ({len(X_train)/len(df_scaled)*100:.1f}%)")
print(f"Validation size: {len(X_val)} ({len(X_val)/len(df_scaled)*100:.1f}%)")
print(f"Test size: {len(X_test)} ({len(X_test)/len(df_scaled)*100:.1f}%)")

# Vérification de la répartition des classes (stratification)
print("\nClass distribution in Train set:")
print(y_train.value_counts(normalize=True))

print("\nClass distribution in Validation set:")
print(y_val.value_counts(normalize=True))

print("\nClass distribution in Test set:")
print(y_test.value_counts(normalize=True))


# In[17]:


from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

scaler = StandardScaler()

# Fit uniquement sur train pour éviter la fuite de données
X_train_scaled = scaler.fit_transform(X_train)

# Transformer validation et test avec les mêmes paramètres
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Résultat en DataFrame (optionnel)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("Feature scaling done.")
print("Train features sample:")
print(X_train_scaled.head())


# In[18]:


import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Reconstituer train set en DataFrame pour manipuler facilement
df_train = X_train.copy()
df_train['Outcome'] = y_train

df_majority = df_train[df_train['Outcome'] == 0]
df_minority = df_train[df_train['Outcome'] == 1]

df_majority_downsampled = df_majority.sample(len(df_minority), random_state=42)

# Recomposer un dataset équilibré
df_balanced = pd.concat([df_majority_downsampled, df_minority])

# Séparer features et cible
X_train_balanced = df_balanced.drop('Outcome', axis=1)
y_train_balanced = df_balanced['Outcome']

print("Distribution après undersampling :")
print(Counter(y_train_balanced))

# Visualisation
plt.bar(['No Diabetes (0)', 'Diabetes (1)'], [Counter(y_train_balanced)[0], Counter(y_train_balanced)[1]])
plt.title("Class distribution in Training Set After Undersampling")
plt.show()


# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = LogisticRegression(random_state=42, max_iter=1000)

model.fit(X_train_balanced, y_train_balanced)

# Prédictions sur validation
val_preds = model.predict(X_val_scaled)

# Évaluation sur validation
accuracy = accuracy_score(y_val, val_preds)
precision = precision_score(y_val, val_preds)
recall = recall_score(y_val, val_preds)
f1 = f1_score(y_val, val_preds)

print("Validation metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")


# In[20]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

# Prédictions sur le test set
test_preds = model.predict(X_test_scaled)

# Calcul des métriques
accuracy = accuracy_score(y_test, test_preds)
precision = precision_score(y_test, test_preds)
recall = recall_score(y_test, test_preds)
f1 = f1_score(y_test, test_preds)

print("Test set evaluation:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# Matrice de confusion
cm = confusion_matrix(y_test, test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix on Test Set")
plt.show()


# In[21]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# Importance des features avec coefficients du modèle (Logistic Regression)
coefficients = model.coef_[0]
features = X_train.columns if hasattr(X_train, 'columns') else [f'Feature {i}' for i in range(len(coefficients))]

coef_df = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})
coef_df['AbsCoefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='AbsCoefficient', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='coolwarm')
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.show()

test_preds = model.predict(X_test_scaled)

print("Classification Report on Test Set:")
print(classification_report(y_test, test_preds, target_names=['No Diabetes', 'Diabetes']))

fp_indices = np.where((y_test == 0) & (test_preds == 1))[0]
fn_indices = np.where((y_test == 1) & (test_preds == 0))[0]

print(f"Number of False Positives: {len(fp_indices)}")
print(f"Number of False Negatives: {len(fn_indices)}")

print("\nExamples of False Positives (first 5 rows):")
print(X_test.iloc[fp_indices[:5]])

print("\nExamples of False Negatives (first 5 rows):")
print(X_test.iloc[fn_indices[:5]])


# In[22]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

test_preds = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, test_preds)
precision = precision_score(y_test, test_preds)
recall = recall_score(y_test, test_preds)
f1 = f1_score(y_test, test_preds)

print("Test set evaluation metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

cm = confusion_matrix(y_test, test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix on Test Set")
plt.show()


# ## Conclusion
# 
# - **Model performance:**  
#   The logistic regression model showed satisfactory results on the test set, balancing precision, recall, and F1-score well. This indicates that the model can correctly detect the presence or absence of diabetes in most cases.
# 
# - **Handling imbalance:**  
#   The chosen method (undersampling or class weighting) helped improve minority class detection.
# 
# - **Feature importance:**  
#   Variables like glucose levels and BMI were most influential, aligning with medical knowledge.
# 
# - **Error analysis:**  
#   False positives and negatives analysis points to potential data collection improvements.
# 
# - **Future improvements:**  
#   - Try more complex models (random forests, boosting)  
#   - Use advanced imbalance techniques (SMOTE, ADASYN)  
#   - Collect more data or features  
#   - Use explainability tools like SHAP or LIME
# 
# - **Deployment recommendations:**  
#   - Continuous monitoring and model retraining  
#   - Collaboration with medical experts for validation
