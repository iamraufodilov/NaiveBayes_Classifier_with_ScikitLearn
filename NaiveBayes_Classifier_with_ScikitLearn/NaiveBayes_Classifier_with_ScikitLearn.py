# importing 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# loading data
from sklearn.datasets import load_breast_cancer
my_data = load_breast_cancer()
label_names = my_data['target_names']
labels = my_data['target']
feature_names = my_data['feature_names']
feature = my_data['data']
print(label_names)
print(feature_names)
print(feature[0])
print(labels[0])

#data splitting
train, test, train_labels, test_labels = train_test_split(feature, labels, test_size=0.4, random_state=42)

#create model
GNBClf = GaussianNB()

# train model
my_model = GNBClf.fit(train, train_labels)
my_preds = GNBClf.predict(test)
print(my_preds)