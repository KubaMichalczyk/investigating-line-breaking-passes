import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import brier_score_loss, roc_auc_score
from socceraction import vaep
from tqdm import tqdm
import matplotlib.pyplot as plt
from ordered_set import OrderedSet
import argparse


def plot_grid_search(cv_results):
    param1 = list(cv_results['params'][0].keys())[0]
    param2 = list(cv_results['params'][0].keys())[1]
    grid_param1 = list(OrderedSet([d[param1] for d in cv_results['params']]))
    grid_param2 = list(OrderedSet([d[param2] for d in cv_results['params']]))
    scores_mean = np.array(cv_results['mean_test_score']).reshape(len(grid_param1), len(grid_param2))
    scores_sd = np.array(cv_results['std_test_score']).reshape(len(grid_param1), len(grid_param2))

    fig, ax = plt.subplots(1, 1)

    for idx, val in enumerate(grid_param2):
        ax.errorbar(grid_param1, scores_mean[:, idx], yerr=scores_sd[:, idx], fmt='-o', label=param2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(param1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')

    return fig, ax


def brier_obj_sklearn(y_true, y_pred):
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    grad = 2 * y_pred * (y_true - y_pred) * (y_pred - 1)
    hess = 2 * y_pred ** (1 - y_pred) * (2 * y_pred * (y_true + 1) - y_true - 3 * y_pred ** 2)
    return grad, hess


def brier_score_sklearn(preds, dtrain):
    labels = dtrain.get_label()
    errors = (labels - preds)**2
    return 'brier-error', float(np.mean(errors))


if __name__ == '__main__':

    with open('all_actions_imputed.pkl', 'rb') as file:
        all_actions = pickle.load(file)
    actions = pd.concat(all_actions)

    X = pd.read_parquet('X_epv.parquet')
    Y = pd.read_parquet('Y_epv.parquet')
    Y_pred = pd.DataFrame()

    parser = argparse.ArgumentParser(description='Compute VAEP values.')
    parser.add_argument('--train', dest='train', action='store_const', const=True, default=False,
                        help='Train the models (default: False)')
    args = parser.parse_args()

    if args.train:

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        model_scores = XGBClassifier(obj=brier_obj_sklearn,
                                     learning_rate=0.01, n_estimators=2500, max_depth=6, min_child_weight=6,
                                     gamma=0, reg_lambda=1, reg_alpha=0, subsample=1, colsample_bytree=0.4,
                                     scale_pos_weight=1, verbosity=1, seed=42, disable_default_eval_metric=1)
        model_scores.fit(X_train, Y_train['scores'],
                         eval_metric=brier_score_sklearn,
                         eval_set=[(X_train, Y_train['scores']), (X_test, Y_test['scores'])],
                         verbose=True,
                         early_stopping_rounds=100)
        Y_pred['scores'] = model_scores.predict_proba(X_test)[:, 1]
        print(f"  Brier score: %.6f" % brier_score_loss(Y_test['scores'], Y_pred['scores']))
        print(f"  ROC AUC: %.4f" % roc_auc_score(Y_test['scores'], Y_pred['scores']))

        pickle.dump(model_scores, open('model_scores_jenks.pkl', 'wb'))

        model_concedes = XGBClassifier(obj=brier_obj_sklearn,
                                       learning_rate=0.1, n_estimators=100, max_depth=4, min_child_weight=6,
                                       gamma=0.1, reg_lambda=0.1, reg_alpha=0, subsample=1, colsample_bytree=0.4,
                                       scale_pos_weight=1, verbosity=1, seed=42, disable_default_eval_metric=1)
        model_concedes.fit(X_train, Y_train['concedes'],
                           eval_metric=brier_score_sklearn,
                           eval_set=[(X_train, Y_train['concedes']), (X_test, Y_test['concedes'])],
                           verbose=True,
                           early_stopping_rounds=50)
        Y_pred['concedes'] = model_concedes.predict_proba(X_test)[:, 1]
        print(f"  Brier score: %.6f" % brier_score_loss(Y_test['concedes'], Y_pred['concedes']))
        print(f"  ROC AUC: %.4f" % roc_auc_score(Y_test['concedes'], Y_pred['concedes']))

        pickle.dump(model_concedes, open('model_concedes_jenks.pkl', 'wb'))

    else:

        model_scores = pickle.load(open('model_scores_jenks.pkl', 'rb'))
        model_concedes = pickle.load(open('model_concedes_jenks.pkl', 'rb'))

    predictions = actions.loc[:, ['id_match', 'event_id']]
    predictions['scores'] = model_scores.predict_proba(X)[:, 1]
    predictions['concedes'] = model_concedes.predict_proba(X)[:, 1]

    # The following columns are created only to match the socceraction naming:
    actions['team_id'] = actions['id_team']
    actions['type_name'] = actions['event_type']
    actions['result_name'] = actions['outcome'].map({0: 'fail', 1: 'success'})

    actions_with_predictions = actions.merge(predictions, on=['id_match', 'event_id'], how='left')
    actions_with_vaep = []
    for _, group in tqdm(actions_with_predictions.groupby(['id_match', 'id_half'])):
        group = pd.concat([group, vaep.value(group, group.scores, group.concedes)], axis=1)
        actions_with_vaep.append(group)
    actions_with_vaep = pd.concat(actions_with_vaep)

    pickle.dump(actions_with_vaep, open('actions_with_vaep_jenks.pkl', 'wb'))
