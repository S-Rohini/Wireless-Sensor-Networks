from keras.models import Sequential
from keras.layers import Dense
from keras.src.optimizers.adam import Adam

from Evaluate_Error import evaluat_error


def Model_MLP(Train_Data, Train_Target, Test_Data, Test_Target, Epoch, sol=None):
    if sol is None:
        sol = [5, 0.01, 1]
    act = ['linear', 'Tanh', 'relu', 'softmax', 'sigmoid']
    model = Sequential()
    model.add(Dense(Train_Target.shape[1], input_dim=Train_Data.shape[1], activation=act[sol[2]]))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=sol[1]), metrics=['accuracy'])
    model.fit(Train_Data, Train_Target, epochs=Epoch, batch_size=10, verbose=0)
    y_pred = model.predict(Test_Data)
    Eval = evaluat_error(y_pred, Test_Target)
    return Eval, y_pred



