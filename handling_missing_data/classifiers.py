import numpy as np


def fit_classifier(classifier, complete_train_x, complete_train_y, hparams):
    fail_state = 0

    if classifier == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression

        logr = LogisticRegression(random_state=0, penalty="none", **hparams)
        logr.fit(complete_train_x, complete_train_y)
        return logr, fail_state

    if classifier == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier

        rfc = RandomForestClassifier(oob_score=True, random_state=0, **hparams)
        rfc.fit(complete_train_x, complete_train_y)
        return rfc, fail_state

    if classifier == "XGBoost":
        from xgboost import XGBClassifier

        xgb = XGBClassifier(learning_rate=0.02, verbosity=1, random_state=0, **hparams)
        xgb.fit(complete_train_x, complete_train_y)
        return xgb, fail_state

    if classifier == "NGBoost":
        from ngboost import NGBClassifier
        from ngboost.distns import Bernoulli

        ngb = NGBClassifier(Dist=Bernoulli, random_state=0, **hparams)
        try:
            ngb.fit(complete_train_x, complete_train_y.astype(int))
        except np.linalg.LinAlgError as err:
            if "Singular matrix" in str(err):
                fail_state = 1
            else:
                raise
        except ValueError as err:
            if "contains infinity" in str(err):
                fail_state = 1
            else:
                raise
        return ngb, fail_state

    if classifier == "NeuralNetwork":
        from sklearn.neural_network import MLPClassifier

        hparams2 = hparams.copy()
        hparams2["hidden_layer_sizes"] = np.ones(
            hparams["number_of_hidden_layers"], dtype=int
        ) * int(hparams["neurons_per_hidden_layer"])
        del hparams2["number_of_hidden_layers"]
        del hparams2["neurons_per_hidden_layer"]

        nn = MLPClassifier(random_state=0, **hparams2)
        nn.fit(complete_train_x, complete_train_y)
        return nn, fail_state

    raise ValueError(f"Unrecognised classifier: {classifier}")
