from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics


from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_openml
mnist = fetch_openml('Fashion-MNIST')

X_train, X_test, y_train, y_test = train_test_split(
    mnist.data, mnist.target, test_size=0.2)

rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train, y_train)


y_pred = rf_clf.predict(X_test)

acc = accuracy_score(y_pred, y_test)

print(classification_report(y_test, y_pred))
