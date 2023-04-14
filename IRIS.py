import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Load the iris dataset
iris = pd.read_csv('C:/Users/hsri2/OneDrive/Desktop/Data sheet/Iris.csv')

# Print the first five rows of the dataset
print(iris.head())

# Print the information about the dataset
print(iris.info())

# Check for missing values
print(iris.isnull().sum())

# Remove the Id column
del iris['Id']

# Encode the species column using LabelEncoder
LB = LabelEncoder()
iris['Species'] = LB.fit_transform(iris['Species'])

# Scale the numerical columns using MinMaxScaler
MMS = MinMaxScaler(feature_range=(0, 1))
num_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
iris[num_cols] = MMS.fit_transform(iris[num_cols])

# Split the dataset into training and testing sets
x = iris.drop(['Species'], axis=1)
y = iris['Species']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Initialize the K-Nearest Neighbors classifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

# Train the classifier on the training set
classifier.fit(x_train, y_train)

# Make predictions on the testing set
y_pred = classifier.predict(x_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print('Accuracy:', accuracy)
print('Confusion Matrix:', cm)

# Visualize the confusion matrix using heatmap
sns.heatmap(cm, annot=True, cmap=sns.color_palette('Blues'), cbar=False,
            xticklabels=LB.classes_, yticklabels=LB.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Visualize the distribution of each numerical feature by species using histograms
sns.set_palette('husl')
sns.displot(iris, x='SepalLengthCm', hue='Species', kde=True)
sns.displot(iris, x='SepalWidthCm', hue='Species', kde=True)
sns.displot(iris, x='PetalLengthCm', hue='Species', kde=True)
sns.displot(iris, x='PetalWidthCm', hue='Species', kde=True)
plt.show()

# Visualize the distribution of each numerical feature individually using histograms
sns.displot(iris, x='SepalLengthCm', hue='Species', kde=True)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Count')
plt.show()

sns.displot(iris, x='SepalWidthCm', hue='Species', kde=True)
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Count')
plt.show()

sns.displot(iris, x='PetalLengthCm', hue='Species', kde=True)
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Count')
plt.show()


