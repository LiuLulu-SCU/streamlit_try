from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pickle

# 加载iris数据集，并进行最简单的训练过程，可以自行扩展
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# 将模型保存为pkl格式，准备应用层调用
pickle.dump(classifier, open("model.pkl", "wb"))
