import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def svm_model(train_points, train_bits, test_points, test_bits):
    train_points = np.asarray(train_points)
    train_bits = np.asarray(train_bits)
    test_points = np.asarray(test_points)
    test_bits = np.asarray(test_bits)

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", gamma="scale", C=1.0)
    )

    model.fit(train_points, train_bits)

    acc = model.score(test_points, test_bits)
    return acc, model
