from sklearn.tree import DecisionTreeRegressor
import pandas as pd

class Model:
    def __init__(self):
        self.model = DecisionTreeRegressor()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")

    id_col = "id"
    target_col = "cost"

    feature_cols = [c for c in train.columns if c not in [id_col, target_col]]

    X_train = train[feature_cols]
    y_train = train[target_col]

    X_test = test[feature_cols]

    model = Model().fit(X_train, y_train)
    preds = model.predict(X_test)

    submission = sample_sub.copy()
    submission[target_col] = preds
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")


if __name__ == "__main__":
    main()
