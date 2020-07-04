import pandas as pd
import socceraction.classification.features as features
import glob
from janitor import clean_names
from read_data import *
from visualise import *
from tqdm import tqdm
from typing import List
from scipy.spatial import distance_matrix
from calculate_soft_voronoi import PITCH

_action_types = ['freekick_pass', 'corner_short', 'throw_in', 'pass', 'freekick_cross', 'corner_crossed', 'cross',
                 'running_with_ball', 'reception', 'clearance', 'block', 'shot_freekick', 'shot_penalty', 'shot',
                 'gk_catch', 'gk_pick_up', 'gk_save', 'gk_drop', 'gk_punch', 'gk_diving', 'foul']
_duel_types = ['Tackle', 'Ground 50 / 50', 'Air 50 / 50', 'One v One', '50 / 50']
_body_parts = ['foot', 'Header', 'Hands (dropped)', 'Chest', 'Hands (punched)']
_phase_types = ['OpenPlay', 'SetPlay']


def determine_custom_event_types(event_df):
    """
    This is a function that changes original STATS event types to more granular event types incorporating the action
    context, e.g. Free kick pass instead of original Pass.
    :param event_df:
    :return:
    """

    df = event_df
    df['event_type'] = np.select(
        [
            (df['event_name'] == 'Pass') & ((df['goal_kick'] == 1) | (df['freekick_direct'] == 1) |
                                            (df['freekick_indirect'] == 1) | (df['penalty'] == 1)),
            (df['event_name'] == 'Pass') & (df['corner'] == 1),
            ((df['event_name'] == 'Pass') | (df['event_name'] == 'Cross')) & (df['throw_in'] == 1),
            df['event_name'] == 'Pass',
            (df['event_name'] == 'Cross') & ((df['freekick_direct'] == 1) | (df['freekick_indirect'] == 1)),
            (df['event_name'] == 'Cross') & (df['corner'] == 1),
            df['event_name'] == 'Cross',
            df['event_name'] == 'Running with ball',
            df['event_name'] == 'Reception',
            df['event_name'].isin(['Clearance', 'Clearance uncontrolled', 'Neutral clearance']),
            df['event_name'] == 'Block',
            ((df['event_name'] == 'Shot on target') | (df['event_name'] == 'Shot not on target')) &
            ((df['freekick_direct'] == 1) | (df['freekick_indirect'] == 1)),
            ((df['event_name'] == 'Shot on target') | (df['event_name'] == 'Shot not on target')) &
            (df['penalty'] == 1),
            df['event_name'].isin(['Shot on target', 'Shot not on target']),
            df['event_name'].isin(['Catch', 'Catch drop']),
            df['event_name'] == 'Hold of ball',
            df['event_name'].isin(['Diving save', 'Catch save', 'Catch drop save', 'Punch save',
                                   'Neutral clearance save']),
            df['event_name'] == 'Drop of ball',
            df['event_name'] == 'Punch',
            df['event_name'] == 'Diving',
            df['event_name'].isin(['Foul - Direct free-kick', 'Foul - Indirect free-kick', 'Foul - Penalty',
                                   'Foul - Throw-in'])
        ],
        [

            'freekick_pass',
            'corner_short',
            'throw_in',
            'pass',
            'freekick_cross',
            'corner_crossed',
            'cross',
            'running_with_ball',
            'reception',
            'clearance',
            'block',
            'shot_freekick',
            'shot_penalty',
            'shot',
            'gk_catch',
            'gk_pick_up',
            'gk_save',
            'gk_drop',
            'gk_punch',
            'gk_diving',
            'foul'

        ]
    )

    return df


def determine_labels(actions_df, n_actions=10):
    """

    :param actions_df: Data frame of actions only, sorted by time of event.
    :param n_actions: Number of actions to look forward.
    :return: Data frame with team, scores and concedes columns.

    Warning: Should be applied to each half of game separately.
    """
    df = actions_df.loc[:, ['team']]

    df['goal'] = (actions_df['event_name'] == 'Shot on target') & (actions_df['outcome'] == 1)
    df['goal_propagated'] = df['goal'][::-1].rolling(n_actions, min_periods=0).max()[::-1]
    df['goal_team_propagated'] = df['team'].where(df['goal'].eq(1)).bfill(limit=n_actions - 1)

    df['own_goal'] = actions_df['outcome'] == 3
    df['own_goal_propagated'] = df['own_goal'][::-1].rolling(n_actions, min_periods=0).max()[::-1]
    df['own_goal_team_propagated'] = df['team'].where(df['own_goal'].eq(1)).bfill(limit=n_actions - 1)

    res_df = actions_df.loc[:, ['team']]
    res_df['scores'] = (((df['goal_propagated'] == 1) & (df['team'] == df['goal_team_propagated'])) |
                        ((df['own_goal_propagated'] == 1) & (df['team'] != df['own_goal_team_propagated']))).astype(int)
    res_df['concedes'] = (((df['goal_propagated'] == 1) & (df['team'] != df['goal_team_propagated'])) |
                          ((df['own_goal_propagated'] == 1) & (df['team'] == df['own_goal_team_propagated']))).astype(int)

    return res_df


def add_tracking_info(event_df, tracking_df):
    event_df.loc[:, 'frame'] = event_df.loc[:, 'time'].round(-2)
    event_df.loc[:, 'next_action_frame'] = event_df.loc[:, 'next_action_time'].round(-2)

    df = event_df \
        .merge(tracking_df.reset_index()[['id_half', 'frame', 'team', 'x', 'y', 'area']],
               how='left', on=['id_half', 'frame', 'team'])
    df['dist_to_start_coordinates'] = np.sqrt((df['start_x'] - df['x']) ** 2 + (df['start_y'] - df['y']) ** 2).fillna(9999)
    start_vor_area = df.loc[df.groupby('event_id')['dist_to_start_coordinates'].idxmin(), 'area']
    start_vor_area.index = event_df.index
    event_df['start_vor_area'] = start_vor_area

    df = event_df \
        .merge(tracking_df.reset_index()[['id_half', 'frame', 'team', 'x', 'y', 'area']] \
               .rename({'frame': 'next_action_frame', 'team': 'next_action_team', 'x': 'next_action_x',
                        'y': 'next_action_y', 'area': 'next_action_area'}, axis=1),
               how='left', on=['id_half', 'next_action_frame', 'next_action_team'])
    df['dist_to_end_coordinates'] = np.sqrt((df['end_x'] - df['next_action_x']) ** 2 +
                                            (df['end_y'] - df['next_action_y']) ** 2).fillna(9999)

    end_vor_area = df.loc[df.groupby('event_id')['dist_to_end_coordinates'].idxmin(), 'next_action_area']
    end_vor_area.index = event_df.index
    event_df['end_vor_area'] = end_vor_area

    return event_df


def add_soft_voronoi_values(event_df, soft_voronoi):
    """

    :param event_df: Pandas DataFrame with game events
    :param soft_voronoi: Pandas Series containing Numpy Array of size (7140, 3) witch soft voronoi values for each frame
    :return: event_df with columns 'start_soft_voronoi' and 'end_soft_voronoi' added

    """

    if 'frame' not in event_df.columns and 'frame' not in event_df.index.names:
        event_df.loc[:, 'frame'] = event_df.loc[:, 'time'].round(-2)
    if 'next_action_frame' not in event_df.columns and 'next_action_frame' not in event_df.index.names:
        event_df.loc[:, 'next_action_frame'] = event_df.loc[:, 'next_action_time'].round(-2)

    # Merging with event data - soft voronoi values only for event times are left
    if event_df.index.names != ['id_half', 'frame']:
        event_df = event_df.reset_index().set_index(['id_half', 'frame'])
    event_df['soft_voronoi'] = soft_voronoi

    # Filtering values for action start coordinates
    start_soft_voronoi = np.stack(event_df['soft_voronoi'])[np.arange(len(event_df['soft_voronoi'])),
                                                            np.argmin(distance_matrix(
                                                                event_df[['start_x', 'start_y']].values, PITCH),
                                                                axis=1)]
    event_df['start_soft_voronoi'] = np.select([event_df['team'] == 'home', event_df['team'] == 'away'],
                                               [start_soft_voronoi[:, 0], start_soft_voronoi[:, 1]],
                                               default=np.nan)

    event_df = event_df.reset_index().set_index(['id_half', 'next_action_frame'])
    event_df['soft_voronoi'] = soft_voronoi
    event_df = event_df.set_index(event_df['soft_voronoi'].groupby(event_df['soft_voronoi'].index).cumcount(), append=True)

    # Filtering values for action end coordinates
    mask = event_df['soft_voronoi'].notna()
    end_soft_voronoi = np.stack(event_df['soft_voronoi'][mask])[np.arange(mask.sum()),
                                                                  np.argmin(distance_matrix(
                                                                      event_df.loc[mask, ['end_x', 'end_y']].values,
                                                                      PITCH),
                                                                      axis=1)]
    event_df.loc[mask, 'end_soft_voronoi'] = np.select([event_df.loc[mask, 'team'] == 'home',
                                                        event_df.loc[mask, 'team'] == 'away'],
                                                        [end_soft_voronoi[:, 0], end_soft_voronoi[:, 1]],
                                                        default=np.nan)

    event_df.reset_index(2, drop=True, inplace=True)
    event_df.drop('soft_voronoi', axis=1, inplace=True)
    event_df.reset_index(inplace=True)

    return event_df


@features.simple
def onehot_event_type(actions):
    res_df = pd.DataFrame()
    for type in _action_types:
        res_df['type_' + type] = (actions['event_type'] == type).astype(float)
    return res_df


@features.simple
def onehot_body_part(actions):
    res_df = pd.DataFrame()
    for part in _body_parts:
        res_df[part] = np.where(actions['body_part'].isnull(), np.nan, (actions['body_part'] == part).astype(float))
    res_df.index = actions.index
    return res_df.clean_names(strip_underscores=True)


@features.simple
def onehot_duel_type(actions):
    res_df = pd.DataFrame()
    for type in _duel_types:
        res_df['duel_' + type] = (actions['duel_type'] == type).astype(float)
    return res_df.clean_names(strip_underscores=True)


@features.simple
def phase_type(actions):
    res_df = pd.DataFrame()
    for type in _phase_types:
        res_df[type] = np.where(pd.isnull(actions['phase_type']), np.nan, (actions['phase_type'] == type).astype(float))
    res_df.index = actions.index
    return res_df.clean_names(strip_underscores=True)


@features.simple
def technical_characteristics(actions):
    res_df = pd.DataFrame()
    res_df['flick_on'] = actions['technical_characteristics'] == 'Flick on'
    return res_df


@features.simple
def line_breaking(actions):
    return actions[['line_breaking']]


def time_delta(gamestates):
    a0 = gamestates[0]
    dt = pd.DataFrame()
    for i, a in enumerate(gamestates[1:]):
        dt["time_delta_" + (str(i + 1))] = a0.time - a.time
    return dt


def create_game_states(actions: pd.DataFrame, nb_prev_actions: int=3) -> List[pd.DataFrame]:
    states = [actions]
    for i in range(1, nb_prev_actions):
        prev_actions = actions.copy().shift(i)
        prev_actions.iloc[:i, :] = pd.concat([actions[:1]] * i).values
        states.append(prev_actions)
    return states


@features.simple
def pass_line_setup_info(actions):
    return actions[['a', 'd', 'line_integrity', 'line_compactness']]


@features.simple
def voronoi_info(actions):
    return actions[['start_vor_area', 'end_vor_area']]


@features.simple
def soft_voronoi_info(actions):
    return actions[['start_soft_voronoi', 'end_soft_voronoi']]


@features.simple
def pitch_control_info(actions):
    return actions[['start_pitch_control', 'end_pitch_control']]


if __name__ == '__main__':

    with open('all_games_events.pkl', 'rb') as file:
        all_games_events = pickle.load(file)

    with open('all_passes_with_line_breaking_info_jenks.pkl', 'rb') as file:
        all_passes = pickle.load(file)

    # Filter all_games_events and sort it to match all_passes order
    filtered_game_ids = [game.id_match.unique().item() for game in all_passes]
    all_games_events = [game for game in all_games_events if game.id_match[0] in filtered_game_ids]
    game_ids = [int(game.id_match.unique().item()) for game in all_games_events]
    order = {str(game_id): i for i, game_id in enumerate(sorted(game_ids))}
    all_games_events = sorted(all_games_events, key=lambda x: order[x['id_match'][0]])

    with open('all_games_events_matched_jenks.pkl', 'wb') as file:
        pickle.dump(all_games_events, file)

    all_games_events_imputed = []
    for game, passes in tqdm(zip(all_games_events, all_passes)):
        assert game.id_match.unique() == passes.id_match.unique()
        game = game.sort_values('event_id').set_index('event_id')
        passes = passes.sort_values('event_id').set_index('event_id')
        cols = game.columns.union(passes.columns, sort=False)
        game, passes = game.align(passes)
        game = game[cols]
        passes = passes[cols]
        game.update(passes)
        all_games_events_imputed.append(game.reset_index())

    xfns = [onehot_event_type, onehot_body_part, onehot_duel_type, phase_type, features.startlocation,
            features.startpolar, features.endlocation, features.endpolar, features.movement, features.space_delta,
            time_delta, line_breaking, pass_line_setup_info, pitch_control_info]

    with open('all_games_actions_pitch_control.pkl', 'rb') as file:
        all_games_actions_pitch_control = pickle.load(file)
    pitch_control = pd.concat(all_games_actions_pitch_control)

    X = []
    all_actions = []
    for game in tqdm(all_games_events_imputed):

        actions = filter_actions(game)
        actions = determine_custom_event_types(actions)

        game_id = actions.iloc[0]['id_match']
        tracking_data = pd.read_parquet(glob.glob('./jenks-clusters/*' + game_id + '.parquet')[0])

        actions = actions.merge(pitch_control, on=['id_match', 'event_id'], how='left')

        actions = standardise_direction_of_play(actions)
        all_actions.append(actions)

        X_game = []
        for _, actions_half in actions.groupby('id_half'):
            game_states = create_game_states(actions_half, 3)
            X_game.append(pd.concat([fn(game_states) for fn in xfns], axis=1))
        X.append(pd.concat(X_game))

    X = pd.concat(X)

    Y = []
    for game in tqdm(all_games_events_imputed):
        actions = filter_actions(game)
        Y_game = []
        for _, actions_half in actions.groupby('id_half'):
            Y_half = determine_labels(actions_half)
            Y_game.append(Y_half)
        Y.append(pd.concat(Y_game))
    Y = pd.concat(Y)

    X.to_parquet('X_epv_jenks.parquet')
    Y.to_parquet('Y_epv_jenks.parquet')
    with open('all_actions_imputed_jenks.pkl', 'wb') as file:
        pickle.dump(all_actions, file)
