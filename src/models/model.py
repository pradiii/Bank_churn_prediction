from sklearn.linear_model import LogisticRegression

def logistic_regression_model():         
    return LogisticRegression(solver='saga',max_iter=1000)
