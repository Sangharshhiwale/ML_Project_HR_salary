# ML_Project_HR_salary
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv('HR_comma_sep.csv')
data
# 1. Perform Data Quality Checks
print("Checking for missing values...")
print(data.isnull().sum())

print("\nBasic Statistics:")
print(data.describe())
sns.pairplot(data= data)
data.info()
# 2. Understand Factors Contributing to Employee Turnover
# 2.1 Heatmap for the correlation matrix

plt.figure(figsize=(12,8))
c=data.corr(numeric_only=True)
sns.heatmap(c,annot=True,fmt=".1f")
plt.title("Correlation between dataset",fontsize=17,c="k")
plt.show()

# 2.2 Draw the distribution plot
# a) Distribution of the satisfaction level variable
sns.countplot(x="satisfaction_level",data=data)
plt.title("Distribution of Satisfaction Level",fontsize=17,c="k")
plt.xlabel("Satisfaction Level",fontsize=12)
plt.ylabel("Count of Satisfaction Level",fontsize=12)
plt.xticks([0, 100], ['Benign', 'Malignant'], fontsize=12)
plt.show()
# b) Distribution of the Last Evaluation variable
sns.countplot(x="last_evaluation",data=data)
plt.title("Distribution of last evaluation",fontsize=17,c="k")
plt.xlabel("Last Evaluation",fontsize=12)
plt.ylabel("Count of Last Evaluation",fontsize=12)
plt.xticks([0, 100], ['Benign', 'Malignant'], fontsize=12)
plt.show()
# c) Distribution of the Average Montly Hours variable
sns.countplot(x="average_montly_hours",data=data)
plt.title("Distribution of Average Montly Hours",fontsize=17,c="k")
plt.xlabel("Average Montly Hours",fontsize=12)
plt.ylabel("Count of Average Montly Hours",fontsize=12)
plt.xticks([0, 100], ['Benign', 'Malignant'], fontsize=12)
plt.show()
# 2.3 Bar plot for project count
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='number_project', hue='left', palette='Set2')
plt.title('Project Count by Employee Turnover')
plt.xlabel('Number of Projects')
plt.ylabel('Count')
plt.legend(title='Left', labels=['Stayed', 'Left'])
plt.show()
# 4. Handle the left Class Imbalance using the SMOTE technique.

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#4.1 Pre-process the data by converting categorical columns to numerical 
# Filter data for employees who left
left_employees = data[data['left'] == 1][['satisfaction_level', 'last_evaluation']]
X=left_employees.values

model = KMeans(n_clusters = 5, n_init = 10, init = 'k-means++', random_state = 42)
y_kmeans = model.fit_predict(X)
y_kmeans
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clustering of Employees Who Left (Satisfaction vs Evaluation)')
plt.xlabel('Satisfaction Level')
plt.ylabel('Last Evaluation')
plt.legend(title='Cluster')
plt.show()
data.columns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

categorical_columns = ['sales', 'salary']
numerical_columns = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years']
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Separate features and target variable
X = data_encoded.drop('left', axis=1)
y = data_encoded['left']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# # Standardize the feature variables
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
smote = SMOTE(random_state=123)
X_train_smote, y_train_smote = smote.fit_resample(X_train_sc, y_train)
plt.scatter(X_train_smote[:, 0], X_train_smote[:, 1], c=y_train_smote, alpha=0.5, cmap="viridis", marker='o')
plt.title('Balanced Data after SMOTE')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
# Train the Decision Tree Classifier with pruning
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)  # Limiting depth to 3
dt_classifier.fit(X_train_smote, y_train_smote)

# Predict on training set
y_train_pred = dt_classifier.predict(X_train_smote)
train_accuracy = accuracy_score(y_train_smote, y_train_pred)

# Predict on testing set
y_test_pred = dt_classifier.predict(X_test_sc)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Train Accuracy: {train_accuracy}')
print(f'Test Accuracy: {test_accuracy}')
# 5. Perform 5-fold cross-validation model training and evaluate performance.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Initialize models
log_reg = LogisticRegression(max_iter=1000, random_state=123)
random_forest = RandomForestClassifier(random_state=123)
gradient_boosting = GradientBoostingClassifier(random_state=123)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
def evaluate_model(model, model_name):
    print(f"\nEvaluating {model_name}...")
    
    # Cross-Validation
    y_pred = cross_val_predict(model, X_train_smote, y_train_smote, cv=skf)
    print(f"\nClassification Report for {model_name}:\n")
    print(classification_report(y_train_smote, y_pred))
    
    ## 6.Find the ROC/AUC for each model and plot the ROC curve.
    # ROC Curve
    y_pred_prob = cross_val_predict(model, X_train_smote, y_train_smote, cv=skf, method='predict_proba')[:, 1]
    fpr, tpr, _ = roc_curve(y_train_smote, y_pred_prob)
    auc_score = roc_auc_score(y_train_smote, y_pred_prob)
    
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")
    return auc_score
# Evaluate all models
plt.figure(figsize=(10, 7))
plt.title("ROC Curves")

log_reg_auc = evaluate_model(log_reg, "Logistic Regression")
random_forest_auc = evaluate_model(random_forest, "Random Forest")
gradient_boosting_auc = evaluate_model(gradient_boosting, "Gradient Boosting")
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
# Compare AUC scores
print("\nModel Comparison Based on AUC Scores:")
print(f"Logistic Regression AUC: {log_reg_auc:.2f}")
print(f"Random Forest AUC: {random_forest_auc:.2f}")
print(f"Gradient Boosting AUC: {gradient_boosting_auc:.2f}")
# 7. Suggest various retention strategies for targeted employees.
best_model = GradientBoostingClassifier(random_state=123)
best_model.fit(X_train_smote, y_train_smote)
# Predict probabilities on the test dataset
y_test_prob = best_model.predict_proba(X_test)[:, 1]

risk_zones = pd.DataFrame({'Employee_ID': X_test.index, 'Turnover_Probability': y_test_prob})
risk_zones['Risk_Zone'] = pd.cut(
    risk_zones['Turnover_Probability'],
    bins=[0, 0.2, 0.6, 0.9, 1],
    labels=['Safe Zone (Green)', 'Low-Risk Zone (Yellow)', 'Medium-Risk Zone (Orange)', 'High-Risk Zone (Red)']
)
# Display risk zones
print("\nSample of Risk Zone Categorization:")
print(risk_zones.head())
# Count employees in each risk zone
risk_zone_counts = risk_zones['Risk_Zone'].value_counts()
print("\nNumber of Employees in Each Risk Zone:")
print(risk_zone_counts)
# Visualize the risk zone distribution
plt.figure(figsize=(12, 7))
sns.barplot(x=risk_zone_counts.index, y=risk_zone_counts.values, palette=['green', 'yellow', 'orange', 'red'])
plt.title('Employee Distribution by Risk Zone')
plt.xlabel('Risk Zone')
plt.ylabel('Number of Employees')
plt.show()


