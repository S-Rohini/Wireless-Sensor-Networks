import numpy as np
from sklearn.ensemble import AdaBoostRegressor

from Evaluate_Error import evaluat_error


def Model_AdaBoost(Train_Data, Train_Target, Test_Data, Test_target, Epoch):
    abc =AdaBoostRegressor(n_estimators=Epoch, learning_rate=1, random_state=0)

    model2 = abc.fit(Train_Data, Train_Target[:, 0])
    pred = model2.predict(Test_Data)

    Eval = evaluat_error(pred, Test_target[:, 0])
    return Eval, pred


