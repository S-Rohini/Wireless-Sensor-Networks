import numpy as np
from sklearn.svm import SVR  # "Support Vector Regressor"
from Evaluate_Error import evaluat_error


def Model_SVM(train_data, train_target, test_data, test_target, Epoch):
    svr_model = SVR(kernel='linear', epsilon=Epoch)
    svr_model.fit(train_data, train_target[:, 0])
    pred = svr_model.predict(test_data)

    Eval = evaluat_error(pred, test_target[:, 0])
    return Eval, pred


