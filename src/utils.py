from sklearn.model_selection import train_test_split
from src.config import CFG


def split_dataset(dataframe):
    X = dataframe.drop(columns=CFG.target_label)
    y = dataframe[CFG.target_label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6, test_size=0.2)
    return X_train, X_test, y_train, y_test
