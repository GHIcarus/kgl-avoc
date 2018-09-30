import numpy as np
from sklearn.model_selection import cross_val_score

def get_scores(model, data, labels):
    '''
    model.fit(data, labels)
    preds = model.predict(data)
    '''
    scores = cross_val_score(model, data, labels, scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)

    return rmse_scores