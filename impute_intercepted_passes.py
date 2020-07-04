import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from janitor import clean_names
from lifelines import WeibullAFTFitter
from statsmodels.stats.outliers_influence import variance_inflation_factor
from read_data import standardise_direction_of_play, unstandardise_direction_of_play

if __name__ == '__main__':
    pass_events = ['Pass', 'Cross']
    
    with open('all_games_events.pkl', 'rb') as file:
        all_games_events = pickle.load(file)
    all_games_events = pd.concat(all_games_events).reset_index(drop=True)
    
    all_passes_and_crosses = all_games_events.loc[all_games_events['event_name'].isin(pass_events)].copy()
    all_passes_and_crosses = standardise_direction_of_play(all_passes_and_crosses)
    all_passes_and_crosses['dx'] = all_passes_and_crosses['end_x'] - all_passes_and_crosses['start_x']
    all_passes_and_crosses['dy'] = all_passes_and_crosses['end_y'] - all_passes_and_crosses['start_y']
    all_passes_and_crosses['pass_length'] = np.sqrt(all_passes_and_crosses['dx'] ** 2 +
                                                    all_passes_and_crosses['dy'] ** 2)
    all_passes_and_crosses.loc[all_passes_and_crosses['pass_length'] == 0, 'pass_length'] = 0.5
    all_passes_and_crosses['pass_angle'] = np.arctan2(all_passes_and_crosses['dy'], all_passes_and_crosses['dx'])
    all_passes_and_crosses['pass_angle_abs'] = np.abs(all_passes_and_crosses['pass_angle'])
    
    X = all_passes_and_crosses[['pass_length', 'intercepted', 'break_through_ball', 'start_x', 'start_y',
                    'pass_angle_abs', 'freekick_direct', 'freekick_indirect', 'throw_in', 'corner', 'goal_kick']].copy()
    X = pd.concat([X,
                   pd.get_dummies(all_passes_and_crosses['event_name']).clean_names().drop('pass', axis=1),
                   pd.get_dummies(all_passes_and_crosses['body_part']).clean_names().drop('foot', axis=1)], axis=1)
    X.dropna(inplace=True)
    X['not_intercepted'] = 1 - X['intercepted']
    X.drop('intercepted', axis=1, inplace=True)

    sns.heatmap(X.corr(), annot=True)
    
    X['hands_gk'] = np.where((X['hands_dropped_'] == 1) & (X['throw_in'] == 0), 1, 0)
    X.drop('hands_dropped_', axis=1, inplace=True)
    
    X.not_intercepted.value_counts(dropna=False)
    
    X_test = X.loc[X.not_intercepted == 1, :].sample(int(X.shape[0] * 0.3), random_state=42)
    X_train = X[~X.index.isin(X_test.index)]
    
    weib = WeibullAFTFitter()
    
    weib.fit(X_train, duration_col='pass_length', event_col='not_intercepted')
    weib.print_summary(3)
    weib.score(X_test)
    
    weib.fit(X_train.drop('start_y', axis=1), duration_col='pass_length', event_col='not_intercepted')
    weib.print_summary(3)
    weib.score(X_test)
    np.exp(-weib.params_.lambda_)
    
    X_with_intercept = X.copy()
    X_with_intercept['intercept'] = 1
    vif_df = pd.DataFrame.from_dict({'variable': X_with_intercept.columns,
                                     'VIF': [variance_inflation_factor(X_with_intercept.values, i)
                                             for i in range(X_with_intercept.shape[1])]})
    
    intercepted_passes = X.loc[~X.not_intercepted.astype(bool)]
    predicted_pass_length = intercepted_passes['pass_length'] + \
                            weib.predict_median(intercepted_passes, conditional_after=intercepted_passes['pass_length'])
    
    recovery_df = pd.concat([predicted_pass_length.to_frame('predicted_pass_length'),
                             all_passes_and_crosses.loc[all_passes_and_crosses['intercepted'].astype(bool),
                                            ['start_x', 'start_y', 'pass_angle', 'pass_length']]], axis=1) \
        .join(all_passes_and_crosses[['end_x', 'end_y', 'field_length', 'field_width']])
    recovery_df['predicted_end_x'] = recovery_df['start_x'] + \
                                     recovery_df['predicted_pass_length'] * np.cos(recovery_df['pass_angle'])
    recovery_df['predicted_end_y'] = recovery_df['start_y'] + \
                                     recovery_df['predicted_pass_length'] * np.sin(recovery_df['pass_angle'])
    
    
    def get_trimmed_y(x):
        slopes = (recovery_df['predicted_end_y'] - recovery_df['start_y']) / (recovery_df['predicted_end_x'] - recovery_df['start_x'])
        intercepts = recovery_df['start_y'] - slopes * recovery_df['start_x']
    
        return slopes * x + intercepts
    
    
    def get_trimmed_x(y):
        slopes = (recovery_df['predicted_end_y'] - recovery_df['start_y']) / (recovery_df['predicted_end_x'] - recovery_df['start_x'])
        intercepts = recovery_df['start_y'] - slopes * recovery_df['start_x']
    
        return np.where(recovery_df['predicted_end_x'] - recovery_df['start_x'] == 0,
                        recovery_df['start_x'],
                        (y - intercepts) / slopes)
    

    # Trim at x = 0
    recovery_df['predicted_end_y'] = np.where(recovery_df['predicted_end_x'] < 0, get_trimmed_y(x=0), recovery_df['predicted_end_y'])
    recovery_df['predicted_end_x'] = np.where(recovery_df['predicted_end_x'] < 0, 0, recovery_df['predicted_end_x'])
    
    # Trim at x = 105
    recovery_df['predicted_end_y'] = np.where(recovery_df['predicted_end_x'] > 105, get_trimmed_y(x=105), recovery_df['predicted_end_y'])
    recovery_df['predicted_end_x'] = np.where(recovery_df['predicted_end_x'] > 105, 105, recovery_df['predicted_end_x'])
    
    # Trim at y = 0
    recovery_df['predicted_end_x'] = np.where(recovery_df['predicted_end_y'] < 0, get_trimmed_x(y=0), recovery_df['predicted_end_x'])
    recovery_df['predicted_end_y'] = np.where(recovery_df['predicted_end_y'] < 0, 0, recovery_df['predicted_end_y'])

    # Trim at y = 68
    recovery_df['predicted_end_x'] = np.where(recovery_df['predicted_end_y'] > 68, get_trimmed_x(y=68), recovery_df['predicted_end_x'])
    recovery_df['predicted_end_y'] = np.where(recovery_df['predicted_end_y'] > 68, 68, recovery_df['predicted_end_y'])

    recovery_df['predicted_pass_length'] = np.sqrt((recovery_df['predicted_end_x'] - recovery_df['start_x']) ** 2 +
                                                   (recovery_df['predicted_end_y'] - recovery_df['start_y']) ** 2)

    all_passes_and_crosses.loc[all_passes_and_crosses['intercepted'] == 1, 'end_x'] = recovery_df['predicted_end_x']
    all_passes_and_crosses.loc[all_passes_and_crosses['intercepted'] == 1, 'end_y'] = recovery_df['predicted_end_y']
    all_passes_and_crosses.drop(['dx', 'dy', 'pass_length', 'pass_angle', 'pass_angle_abs'], axis=1, inplace=True)

    all_passes = all_passes_and_crosses.loc[all_passes_and_crosses['event_name'] == 'Pass', :]
    all_passes = unstandardise_direction_of_play(all_passes)
    all_passes = [df for _, df in all_passes.groupby('id_match')]

    with open('all_passes_imputed.pkl', 'wb') as file:
        pickle.dump(all_passes, file)

