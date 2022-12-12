from sklearn import metrics
from sklearn.model_selection import train_test_split

def gridSearch(X, y, parameters, get_regressor, estimator = metrics.accuracy_score) -> int:
    max_acc = -1
    curr = -1
    for parameter in parameters:
        clf = get_regressor(n_neighbors=parameter)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)
        clf.fit(X_train, y_train)
        acc = estimator(clf.predict(X_test), y_test)
        if acc > max_acc:
            curr = parameter
            max_acc = acc
    return curr
