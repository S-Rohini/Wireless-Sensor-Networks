import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from Evaluate_Error import evaluat_error


def Model_Random_Forest(Train_Data, Train_Target, Test_Data, Test_Target, Epoch):
    regressor = RandomForestRegressor(n_estimators=Epoch, random_state=0, oob_score=True)
    regressor.fit(Train_Data, Train_Target)
    pred = regressor.predict(Test_Data)

    Eval = evaluat_error(pred, Test_Target)
    return Eval, pred

