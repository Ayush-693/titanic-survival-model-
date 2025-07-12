# titanic-survival-model-
import numpy as np
import pandas as pd


df = pd.read_csv("titanic.csv")
df.head()

# Get the list of columns to drop
columns_to_drop = ["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"]

# Get the columns currently in the DataFrame
existing_columns = df.columns.tolist()

# Filter the list of columns to drop to only include existing columns
columns_to_drop_existing = [col for col in columns_to_drop if col in existing_columns]

# Drop the existing columns
df.drop(columns_to_drop_existing, axis=1, inplace=True)

df.head()

df.info()


# drop duplicated data
df.drop_duplicates(inplace=True)

df.duplicated().sum()

# Encode Sex column 
df["Sex"] = df["Sex"].map({"male":0, "female":1})

df.head()


# seperate both features and target variable 
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train and test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

len(X_train)

len(X_test)

# Import the Decision Tree model and train it
from sklearn import tree

model= tree.DecisionTreeClassifier()
model.fit(X_train, y_train)


# Model score
model.score(X_test, y_test)

# Confusion Matrix
y_predict = model.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predict)
cm

# Create Heatmap
import seaborn as sns

ax = sns.heatmap(cm, annot = True)
# Add x and y axis labels
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')


ax.set_title('Confusion Matrix')


# Import Random Forest and train it
from sklearn.ensemble import RandomForestClassifier 
model2 = RandomForestClassifier()
model2.fit(X_train, y_train)

model2.score(X_test, y_test)

y_predict = model2.predict(X_test)

# Confusion matrix
cm2 = confusion_matrix(y_test, y_predict)
cm2

# heatmap
sns.heatmap(cm2, annot = True)


# prompt: output 0f 5 person from data and print it with name

import pandas as pd
# Since the "Name" column was dropped, we need to reload the data to access the names.
# Assuming the file "titanic.csv" used before contains the "Name" column.
try:
    df_original = pd.read_csv("titanic.csv")

    # Combine the original 'Name' column with the processed dataframe
    # We'll use the original dataframe to get the names based on the index of the processed dataframe.
    # Ensure that the indices align after dropping duplicates
    df_with_names = pd.concat([df_original['Name'], df], axis=1).dropna(subset=['Survived'])


    # Get the first 5 rows of the combined dataframe
    first_5_people = df_with_names.head(5)

    # Print the output of the first 5 people with their names
    print("First 5 people from the data with their names and processed features:")
    display(first_5_people)
except FileNotFoundError:
    print("Error: 'titanic.csv' not found. Please ensure the file is uploaded or in the correct directory.")
except KeyError:
    print("Error: The 'Name' column was not found in 'titanic.csv'. Please check the file content.")

