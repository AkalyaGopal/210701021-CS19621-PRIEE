from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import pickle
  
# load the data
dataset = pd.read_csv(r'C:\Users\akalamri\Downloads\MINI PROJECT\CODE\INTELLIGENT-CAREER-GUIDANCE-SYSTEM-main\INTELLIGENT-CAREER-GUIDANCE-SYSTEM-main\dataset9000.data', header=None)
print(dataset.head())

# Prepare the data
X = np.array(dataset.iloc[:, 0:17])
print(X)
Y = np.array(dataset.iloc[:, 17])
print(Y)

dataset.columns = [
    "Database Fundamentals", "Computer Architecture", "Distributed Computing Systems",
    "Cyber-Security", "Networking", "Development", "Programming Skills", "Project Management",
    "Computer Forensics Fundamentals", "Technical Communication", "AI ML", "Software Engineering", "Business Analysis",
    "Communication skills", "Data Science", "Troubleshooting-skills", "Graphics Designing", "Roles"
]
dataset.dropna(how='all', inplace=True)

seed = 5

# Initialize KFold with shuffle=True
kfold = model_selection.KFold(n_splits=15, shuffle=True, random_state=seed)

# Initialize the base classifier
base_cls = DecisionTreeClassifier()

# Number of base classifiers
num_trees = 50

# Bagging classifier
model = BaggingClassifier(estimator=base_cls, n_estimators=num_trees, random_state=seed)

# Evaluate the model
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("Accuracy:", results.mean() * 100)
