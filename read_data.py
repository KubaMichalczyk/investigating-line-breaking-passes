import numpy as np
import pandas as pd
import xml.etree.ElementTree as et
import re
import os
from tqdm import tqdm
import pickle
import scipy.signal as signal


def clean_names(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def read_tracking_sportvu(file):
    """
    Parsing tracking data into wide format. The order of information within the data frame: time info,
    22 or more player coordinates and ball coordinates.

    Greater than 22 number of players is possible due to substitution events or reduntant white spaces at the end
    of the parsed message.
    """

    data = pd.read_csv(file, header=None, sep=':')
    data.columns = ['time_data', 'player_coordinates', 'ball_coordinates']

    time_data = data['time_data'].str.split(';', expand=True)
    time_data = pd.concat([time_data[0], time_data[1].str.split(',', expand=True)], axis=1)
    time_data.columns = pd.MultiIndex.from_product([['time_info'], ['timestamp', 'time', 'id_half', 'paused']])
    time_data = time_data.astype({('time_info', 'timestamp'): 'uint64', ('time_info', 'time'): 'uint32',
                                  ('time_info', 'id_half'): 'int8', ('time_info', 'paused'): 'int8'})

    player_data = data['player_coordinates'].str.split(';', expand=True)
    player_data = player_data.stack().str.split(',', expand=True)
    player_data.columns = ['team_id', 'player_id', 'jersey_number', 'x', 'y']
    player_data = player_data \
        .replace({'': np.nan})  \
        .astype({'x': 'float32', 'y': 'float32', 'player_id': 'object', 'team_id': 'object', 'jersey_number': 'object'})
    player_data = player_data.unstack().swaplevel(axis=1).sort_index(level=0, axis=1)
    player_data = player_data.reindex(['team_id', 'player_id', 'jersey_number', 'x', 'y'], level=1, axis=1)
    player_data.columns = pd.MultiIndex.from_tuples([(f'player{a + 1}', b) for a, b in player_data.columns])

    ball_data = data['ball_coordinates'].str.replace(';', '').str.split(',', expand=True)
    ball_data.columns = pd.MultiIndex.from_tuples([('ball', a) for a in ['x', 'y', 'z']])
    ball_data = ball_data.replace({'': np.nan}).astype({('ball', 'x'): 'float32', ('ball', 'y'): 'float32',
                                                        ('ball', 'z'): 'float32'})

    data = pd.concat([time_data, player_data, ball_data], axis=1, sort=False)

    idx = pd.IndexSlice
    x_columns = data.loc[:, idx[:, 'x']]
    data.loc[:, idx[:, 'x']] = x_columns.mask((x_columns < -10) | (x_columns > 115))
    y_columns = data.loc[:, idx[:, 'y']]
    data.loc[:, idx[:, 'y']] = y_columns.mask((y_columns < -10) | (y_columns > 78))
    # z coordinate is not registered in tracking data despite that is claimed in specification (z = 0 for all frames)

    data = data[data[('time_info', 'time')] % 100 == 0]

    return data


def tracking_to_long(data):
    data = data.set_index([('time_info', 'id_half'), ('time_info', 'time')]) \
        .drop('time_info', axis=1, level=0) \
        .stack(level=0)
    data.index.names = ['id_half', 'frame', 'object']
    data.loc[data.index.get_level_values('object') == 'ball', 'team_id'] = '99'
    return data


def add_movement(data):

    data[['dx', 'dy']] = (data[['x', 'y']] - data.groupby(['id_half', 'object'])[['x', 'y']].shift()).fillna(0)
    data['dx'] = data.groupby(['id_half', 'object'])['dx'] \
        .transform(lambda x: signal.savgol_filter(x, window_length=3, polyorder=1, mode='nearest'))
    data['dy'] = data.groupby(['id_half', 'object'])['dy'] \
        .transform(lambda x: signal.savgol_filter(x, window_length=3, polyorder=1, mode='nearest'))
    
    return data


def read_event_sportvu(file, match_sheet_info=['id_actor', 'id_team', 'name', 'nick_name', 'position', 'team']):
    """
    Coordinates are scaled into metres (usually 105m x 68m, but it depends on the real pitch size).
    """
    xroot = et.parse(file).getroot()
    game_info = pd.DataFrame.from_dict(xroot.attrib, orient='index').T

    halves = xroot.findall('./Events/EventsHalf')
    game_events = []
    for half in halves:
        half_info = pd.DataFrame.from_dict(half.attrib, orient='index').T.drop('DateAndTime', axis=1)
        events = []
        for event in list(half):
            events.append(pd.DataFrame.from_dict(event.attrib, orient='index').T)
        events = pd.concat(events, axis=0, ignore_index=True)
        half_events = pd.concat([half_info, events], axis=1)
        half_events[half_info.columns] = half_events[half_info.columns].ffill()
        game_events.append(half_events)
    game_events = pd.concat(game_events, axis=0, ignore_index=True)
    game_events = pd.concat([game_info, game_events], axis=1).replace('', np.nan)
    game_events[['HitsPost', 'Blocked', 'BreakThroughBall']] = \
        game_events[['HitsPost', 'Blocked', 'BreakThroughBall']].replace({'': np.nan, 'False': 0, 'True': 1})

    game_events[game_info.columns] = game_events[game_info.columns].ffill()
    game_events = game_events.astype({'IdHalf': 'int8', 'Time': 'float32', 'FieldLength': 'float32',
                                      'FieldWidth': 'float32', 'LocationX': 'float64', 'LocationY': 'float64',
                                      'LocationZ': 'float64', 'TargetX': 'float32', 'TargetY': 'float32',
                                      'TargetZ': 'float32', 'HitsPost': 'float32', 'Blocked': 'float32',
                                      'PhaseStartTime': 'float32', 'PhaseEndTime': 'float32',
                                      'BreakThroughBall': 'float32', 'ScoreHomeTeam': 'float32',
                                      'ScoreAwayTeam': 'float32', 'RedCardsHomeTeam': 'float32',
                                      'RedCardsAwayTeam': 'float32'})
    game_events.columns = [clean_names(c) for c in game_events.columns]

    game_events['assist'] = np.nan
    game_events.loc[game_events['event_name'].isin(['Pass assist', 'Cross assist']), 'assist'] = 1
    game_events.loc[game_events['event_name'].isin(['Pass', 'Cross']), 'assist'] = 0
    game_events['event_name'].replace({'Pass assist': 'Pass', 'Cross assist': 'Cross'}, inplace=True)

    match_sheet = []
    for team in list(xroot.find('MatchSheet')):
        team_info = pd.DataFrame.from_dict(team.attrib, orient='index').T
        players = []
        for player in list(team):
            players.append(pd.DataFrame.from_dict(player.attrib, orient='index').T)
        players = pd.concat(players, axis=0, sort=False, ignore_index=True)
        team_info = pd.concat([team_info, players], axis=1)
        match_sheet.append(team_info)
    match_sheet = pd.concat(match_sheet, axis=0, ignore_index=True, sort=False).replace('', np.nan).ffill()
    match_sheet.columns = [clean_names(c) for c in match_sheet.columns]
    match_sheet['team'] = match_sheet['type'].replace({'HomeTeam': 'home', 'AwayTeam': 'away'})
    match_sheet_excerpt = match_sheet[match_sheet_info]

    game_events = game_events.merge(match_sheet_excerpt, how='left', left_on='id_actor1', right_on='id_actor')

    game_events['field_length'] /= 100
    game_events['field_width'] /= 100
    game_events['location_x'] /= 100
    game_events['location_y'] /= 100
    game_events['target_x'] /= 100
    game_events['target_y'] /= 100
    game_events['location_x'] = (game_events['location_x'] + game_events['field_length'] / 2)
    game_events['location_y'] = (game_events['location_y'] + game_events['field_width'] / 2)
    game_events['target_x'] = (game_events['target_x'] + game_events['field_length'] / 2)
    game_events['target_y'] = (game_events['target_y'] + game_events['field_width'] / 2)
    game_events['body_part'] = np.where(game_events['body_part'] == 'Left foot', 'foot', game_events['body_part'])

    direction_of_play = game_events \
        .loc[game_events['position'] == 'Goalkeeper', ['id_half', 'team', 'location_x']] \
        .groupby(['id_half', 'team'])['location_x'] \
        .apply(lambda x: 'Left to right' if x.mean() < 52.5 else 'Right to left')
    direction_of_play.name = 'direction_of_play'
    game_events = game_events.merge(direction_of_play, how='left', on=['id_half', 'team'])

    game_events = game_events \
        .sort_values(['id_half', 'time'])
    game_events = game_events \
        .reset_index() \
        .rename({'index': 'event_id'}, axis=1)

    game_events['date_and_time'] = pd.to_datetime(game_events['date_and_time'])
    game_events['time_formatted'] = pd.to_timedelta(game_events['time'], unit='milliseconds')
    game_events['time_seconds'] = game_events['time'] / 1000

    return game_events


def read_playerinfo_sportvu(file):
    player_df = pd.read_csv(file, names=['jersey_number', 'team_id', 'first_name', 'last_name', 'position_id',
                                         'birth_date', 'inpitch', 'outpitch', 'player_id_event', 'player_id_tracking',
                                         'match_id_event', 'match_id_tracking'])
    player_df[['player_id_event', 'player_id_tracking']] = player_df[['player_id_event', 'player_id_tracking']].astype(str)
    return player_df


def query_with_context(df, expr, before=1, after=1, add_index_cols=False):
    """
    Takes a data frame, filters by condition and returns all rows that match the condition, with additional number
    of rows before and after.

    expr : str
        The query string to evaluate.  You can refer to variables
        in the environment by prefixing them with an '@' character like
        ``@a + b``.

    (!) check if the function works properly regarding two halves indexed separetely
    """

    indices = df \
        .reset_index() \
        .query(expr) \
        .index
    indices_prev = np.array(indices - before)
    indices_prev[indices_prev < 0] = 0
    indices_next = np.array(indices + after)
    indices_next[indices_next > df.shape[0]] = df.shape[0]
    if add_index_cols:
        for i, value in enumerate(list(range(-before, after + 1))):
            if value < 0:
                df['before' + str(-value)] = False
                df.loc[(indices + value)[indices + value >= 0], 'before' + str(-value)] = True
            elif value == 0:
                df['indices'] = False
                df.loc[indices + value, 'indices'] = True
            else:
                df['after' + str(value)] = False
                df.loc[(indices + value)[indices + value <= df.shape[0]], 'after' + str(value)] = True
    indices = np.unique(np.concatenate([np.arange(start, end + 1) for start, end in zip(indices_prev, indices_next)]))
    return df.iloc[indices]


_actions = ['Block', 'Catch', 'Catch drop', 'Catch drop save', 'Catch save', 'Clearance', 'Clearance uncontrolled',
            'Cross', 'Cross assist', 'Diving', 'Diving save', 'Drop of ball', 'Foul - Direct free-kick',
            'Foul - Indirect free-kick', 'Foul - Penalty', 'Foul - Throw-in', 'Hold of ball',
            'Neutral clearance', 'Neutral clearance save', 'Pass', 'Pass assist', 'Punch', 'Punch save',
            'Reception', 'Running with ball', 'Shot not on target', 'Shot on target']


def filter_actions(event_df):
    return event_df.loc[event_df['event_name'].isin(_actions), :]


def add_helpers(event_df):

    event_info = pd.DataFrame.from_dict(
        {'event_name': {0: 'Block', 1: 'Catch', 2: 'Catch drop', 3: 'Catch drop save', 4: 'Catch save', 5: 'Chance',
                        6: 'Clearance', 7: 'Clearance uncontrolled', 8: 'Cross', 9: 'Cross assist', 10: 'Crossbar',
                        11: 'Diving', 12: 'Diving save', 13: 'Drop of ball', 14: 'End of Half',
                        15: 'Foul - Direct free-kick',
                        16: 'Foul - Indirect free-kick', 17: 'Foul - Penalty', 18: 'Foul - Throw-in', 19: 'Goal',
                        20: 'HalfTime Start', 21: 'Hold of ball', 22: 'Interruption', 23: 'Left goal post',
                        24: 'Neutral clearance', 25: 'Neutral clearance save', 26: 'Off side', 27: 'Other obstacle',
                        28: 'Out for corner', 29: 'Out for goal kick', 30: 'Out for throw-in', 31: 'Own Goal',
                        32: 'Pass', 33: 'Pass assist', 34: 'Punch', 35: 'Punch save', 36: 'Reception', 37: 'Red card',
                        38: 'Right goal post', 39: 'Running with ball', 40: 'Shot not on target', 41: 'Shot on target',
                        42: 'Substitution', 43: 'Yellow card'},
         'is_action': {0: 'Action', 1: 'Action', 2: 'Action', 3: 'Action', 4: 'Action', 5: 'Technical', 6: 'Action',
                       7: 'Action', 8: 'Action', 9: 'Action', 10: 'Technical', 11: 'Action', 12: 'Action', 13: 'Action',
                       14: 'Technical', 15: 'Action', 16: 'Action', 17: 'Action', 18: 'Action', 19: 'Technical',
                       20: 'Technical', 21: 'Action', 22: 'Technical', 23: 'Technical', 24: 'Action', 25: 'Action',
                       26: 'Technical', 27: 'Technical', 28: 'Technical', 29: 'Technical', 30: 'Technical',
                       31: 'Technical', 32: 'Action', 33: 'Action', 34: 'Action', 35: 'Action', 36: 'Action',
                       37: 'Technical', 38: 'Technical', 39: 'Action', 40: 'Action', 41: 'Action', 42: 'Technical',
                       43: 'Technical'},
         'action_type': {0: 'Defensive', 1: 'Defensive', 2: 'Defensive', 3: 'Defensive', 4: 'Defensive', 5: pd.NA,
                         6: 'Defensive', 7: 'Defensive', 8: 'Offensive', 9: 'Offensive', 10: pd.NA, 11: 'Defensive',
                         12: 'Defensive', 13: 'Offensive', 14: pd.NA, 15: 'Defensive', 16: 'Defensive', 17: 'Defensive',
                         18: 'Offensive', 19: pd.NA, 20: pd.NA, 21: 'Offensive', 22: pd.NA, 23: pd.NA, 24: 'Defensive',
                         25: 'Defensive', 26: pd.NA, 27: pd.NA, 28: pd.NA, 29: pd.NA, 30: pd.NA, 31: pd.NA,
                         32: 'Offensive',
                         33: 'Offensive', 34: 'Defensive', 35: 'Defensive', 36: 'Offensive', 37: pd.NA, 38: pd.NA,
                         39: 'Offensive', 40: 'Offensive', 41: 'Offensive', 42: pd.NA, 43: pd.NA}}
    )

    event_df = event_df.merge(event_info[['event_name', 'action_type', 'is_action']], on='event_name', how='left')

    event_df_shifted = event_df.groupby(['id_match', 'id_half']).shift(-1)
    event_df[['next_action_time', 'next_action_event_name', 'next_action_event_id', 'next_action_team',
              'next_action_location_x', 'next_action_location_y']] = \
        event_df_shifted[['time', 'event_name', 'event_id', 'team', 'location_x', 'location_y']] \
            .where(event_df_shifted['is_action'] == 'Action')
    event_df.loc[:, ['next_action_time', 'next_action_event_name', 'next_action_event_id', 'next_action_team',
                     'next_action_location_x', 'next_action_location_y']] = event_df \
        .groupby(['id_match', 'id_half'])[['next_action_time', 'next_action_event_name', 'next_action_event_id',
                                           'next_action_team', 'next_action_location_x', 'next_action_location_y']] \
        .bfill()

    event_df_shifted = event_df.groupby(['id_match', 'id_half']).shift()
    event_df[['prev_action_event_name', 'prev_action_event_id']] = \
        event_df_shifted[['event_name', 'event_id']].where(event_df_shifted['is_action'] == 'Action')
    event_df.loc[:, ['prev_action_event_name', 'prev_action_event_id']] = event_df \
        .groupby(['id_match', 'id_half'])[['prev_action_event_name', 'prev_action_event_id']] \
        .ffill()
    event_df[['prev_offensive_action_event_name', 'prev_offensive_action_event_id']] = \
        event_df_shifted[['event_name', 'event_id']] \
            .where((event_df_shifted['is_action'] == 'Action') & (event_df_shifted['action_type'] == 'Offensive'))
    event_df.loc[:, ['prev_offensive_action_event_name', 'prev_offensive_action_event_id']] = event_df \
        .groupby(['id_match', 'id_half'])[['prev_offensive_action_event_name', 'prev_offensive_action_event_id']] \
        .ffill()

    event_df_shifted = event_df.groupby(['id_match', 'id_half', 'id_actor1']).shift()
    event_df[['prev_actor_action_event_name', 'prev_actor_action_event_id']] = \
        event_df_shifted[['event_name', 'event_id']].where(event_df_shifted['is_action'] == 'Action')
    event_df.loc[:, ['prev_actor_action_event_name', 'prev_actor_action_event_id']] = event_df \
        .groupby(['id_match', 'id_half', 'id_actor1'])[['prev_actor_action_event_name', 'prev_actor_action_event_id']] \
        .ffill()

    event_df[['next_event_time', 'next_event_name', 'next_event_team', 'next_event_location_x',
                'next_event_location_y']] = event_df \
        .sort_values(['id_half', 'time']) \
        .groupby(['id_match', 'id_half'])[['time', 'event_name', 'team', 'location_x', 'location_y']] \
        .shift(-1)
    event_df['next_actor_event_name'] = event_df \
        .groupby(['id_match', 'id_half', 'id_actor1'])['event_name'] \
        .shift(-1)
    event_df['next_2nd_actor_event_name'] = event_df \
        .groupby(['id_match', 'id_half', 'id_actor1'])['event_name'] \
        .shift(-2)
    event_df['prev_actor_event_name'] = event_df \
        .groupby(['id_match', 'id_half', 'id_actor1'])['event_name'] \
        .shift(1)

    return event_df


def add_start_coordinates(event_df):

    event_df['start_x'] = event_df['location_x']
    event_df['start_y'] = event_df['location_y']

    return event_df


def add_end_coordinates(event_df):

    if ('next_action_location_x' not in event_df.columns) | ('next_action_location_y' not in event_df.columns) | \
            ('prev_action_event_id' not in event_df.columns):
        event_df = add_helpers(event_df)

    ball_out_of_play = event_df \
        .loc[event_df['event_name'].isin(['Out for throw-in', 'Out for goal kick', 'Out for corner']),
             ['prev_action_event_id', 'location_x', 'location_y']] \
        .rename({'location_x': 'end_x', 'location_y': 'end_y'}, axis=1)
    event_df = event_df.merge(ball_out_of_play.rename({'prev_action_event_id': 'event_id'}, axis=1),
                              on='event_id', how='left')

    _shot_actions = ['Shot on target', 'Shot not on target']
    _moving_actions = ['Clearance', 'Clearance uncontrolled', 'Cross', 'Cross assist', 'Neutral clearance', 'Pass',
                       'Pass assist', 'Punch', 'Running with ball']
    _inplace_actions = list(set(_actions).difference(set(_moving_actions)))

    shot_actions_id = event_df['event_name'].isin(_shot_actions)
    inplay_moving_action_id = (event_df['event_name'].isin(_moving_actions)) & \
                              ((event_df['end_x'].isnull()) | (event_df['end_y'].isnull()))
    inplace_action_id = event_df['event_name'].isin(_inplace_actions)

    event_df['end_x'] = np.select([shot_actions_id, inplay_moving_action_id, inplace_action_id],
                                  [event_df['target_x'], event_df['next_action_location_x'], event_df['location_x']],
                                  default=event_df['end_x'])
    event_df['end_y'] = np.select([shot_actions_id, inplay_moving_action_id, inplace_action_id],
                                  [event_df['target_y'], event_df['next_action_location_y'], event_df['location_y']],
                                  default=event_df['end_y'])

    return event_df


def add_offside_flag(event_df):

    offsides = event_df.loc[event_df['event_name'] == 'Off side', ['prev_offensive_action_event_id']]
    offsides['offside'] = 1
    event_df = event_df.merge(offsides.rename({'prev_offensive_action_event_id': 'event_id'}, axis=1), on='event_id',
                              how='left')
    event_df['offside'] = event_df.loc[event_df['event_name'].isin(['Cross', 'Pass', 'Shot on target',
                                                                    'Shot not on target']), 'offside'].fillna(0)

    return event_df


def add_interception_flag(event_df):

    interceptions = event_df.loc[event_df['event_name'] == 'Block', ['prev_action_event_id']]
    interceptions['intercepted'] = 1
    event_df = event_df.merge(interceptions.rename({'prev_action_event_id': 'event_id'}, axis=1), on='event_id',
                              how='left')
    event_df['intercepted'] = event_df[['blocked', 'intercepted']].max(axis=1)
    event_df['intercepted'] = event_df.loc[event_df['event_name'].isin(['Cross', 'Pass']), 'intercepted'].fillna(0)

    return event_df


def add_freekick_direct_flag(event_df):

    event_df['freekick_direct'] = (event_df['start_of_play'] == 'Direct free-kick').astype(int)
    event_df['freekick_direct'] = event_df.loc[event_df['event_name'].isin(['Cross', 'Pass', 'Shot not on target',
                                                                            'Shot on target']), 'freekick_direct'] \
        .fillna(0)

    return event_df


def add_freekick_indirect_flag(event_df):

    event_df['freekick_indirect'] = (event_df['start_of_play'] == 'Indirect free-kick').astype(int)
    event_df['freekick_indirect'] = event_df.loc[event_df['event_name'].isin(['Cross', 'Pass', 'Shot not on target',
                                                                              'Shot on target']), 'freekick_indirect'] \
        .fillna(0)

    return event_df


def add_throw_in_flag(event_df):

    event_df['throw_in'] = (event_df['start_of_play'] == 'Throw in').astype(int)
    event_df['throw_in'] = event_df.loc[event_df['event_name'].isin(['Cross', 'Pass']), 'throw_in'].fillna(0)

    return event_df


def add_corner_flag(event_df):

    event_df['corner'] = (event_df['start_of_play'] == 'Corner').astype(int)
    event_df['corner'] = event_df.loc[event_df['event_name'].isin(['Cross', 'Pass']), 'corner'].fillna(0)

    return event_df


def add_penalty_flag(event_df):

    event_df['penalty'] = (event_df['start_of_play'] == 'Penalty').astype(int)
    event_df['penalty'] = event_df.loc[event_df['event_name'].isin(['Shot not on target', 'Shot on target']),
                                       'penalty'].fillna(0)

    return event_df


def add_goal_kick_flag(event_df):

    event_df['goal_kick'] = (event_df['start_of_play'] == 'Goal kick').astype(int)
    event_df['goal_kick'] = event_df.loc[event_df['event_name'].isin(['Pass', 'Cross']), 'goal_kick'].fillna(0)

    return event_df


def define_possession(event_df, return_actions_only=False):
    """
    A team is in possession until the opponent makes the action. Defensive action is treated as a start of possession
    if and only if it is followed by offensive action performed by the same team.

    (!) To consider if should be changed to: Defensive action is treated as a start of possession if and only if it is
    followed by a uninterrupted series of actions performed by the same team.
    This should resolve the issue with Catch drop - Catch series which classify Catch drop as failure.
    On the other hand, this is negligible from our perspective.
    """
    actions_df = filter_actions(event_df).copy()
    actions_df.loc[:, 'prev_team'] = actions_df.groupby(['id_match', 'id_half'])['team'].shift(1)
    actions_df.loc[:, 'id_possession'] = (actions_df['team'] != actions_df['prev_team'])
    actions_df.loc[:, 'next_action_type'] = actions_df.groupby(['id_match', 'id_half'])['action_type'].shift(-1)
    actions_df.loc[:, 'id_possession'] = np.where(
        (actions_df['action_type'] == 'Defensive') &
        ((actions_df['next_action_type'] != 'Offensive') | (actions_df['next_action_team'] != actions_df['team'])),
        np.nan,
        actions_df['id_possession']
    )
    actions_df.loc[:, 'id_possession'] = actions_df['id_possession'].cumsum()
    actions_df[['next_id_possession', 'next_action_type']] = actions_df \
        .groupby(['id_match', 'id_half'])[['id_possession', 'action_type']] \
        .shift(-1)
    event_df = event_df.merge(actions_df[['event_id', 'id_possession', 'next_id_possession', 'next_action_type']],
                              how='left', on='event_id')
    if return_actions_only:
        return actions_df
    else:
        return event_df


def determine_outcome(event):
    """
    (!) Offsides are currently not included as a possible outcome.
    """

    if event['next_actor_event_name'] == 'Own Goal':
        return 3
    elif event['event_name'] in ['Block', 'Catch drop save', 'Catch save', 'Clearance', 'Clearance uncontrolled',
                                 'Cross assist', 'Diving save', 'Drop of ball', 'Goal', 'Neutral clearance',
                                 'Neutral clearance save', 'Pass assist', 'Punch', 'Punch save']:
        return 1
    elif event['event_name'] in ['Foul - Direct free-kick', 'Foul - Indirect free-kick', 'Foul - Penalty',
                                 'Foul - Throw-in', 'Shot not on target']:
        return 0
    elif event['event_name'] == 'Shot on target':
        if (event['next_actor_event_name'] == 'Chance') & \
                (event['next_2nd_actor_event_name'] == 'Goal') | (event['next_actor_event_name'] == 'Goal'):
            return 1
        else:
            return 0
    elif event['event_name'] in ['Cross', 'Pass']:
        if event['next_actor_event_name'] == 'Foul - Throw-in':
            return 0
        elif event['id_possession'] == event['next_id_possession']:
            return 1
        else:
            return 0
    elif event['event_name'] in ['Catch', 'Catch drop', 'Diving', 'Hold of ball', 'Reception', 'Running with ball']:
        if event['id_possession'] == event['next_id_possession']:
            return 1
        else:
            return 0
    else:
        return np.nan


def scale_coordinates(df, xy_columns, field_length, field_width):
    """
    Scales the coordinates in xy_columns to 105x68 (metres) size.

    :param df: Pandas DataFrame that contains coordinates columns to be scaled.
    :param xy_columns: List of tuples with pairs of coordinates (x, y) column names.
    :param field_length: Either a scalar of Pandas Series with an original length of the pitch.
    :param field_width: Either a scalar or Pandas Series with an original width of the pitch.
    :return: Pandas DataFrame with coordinates in xy_columns scaled to 105x68 size (in metres).
    """
    df = df.copy()
    for coords in xy_columns:
        df[coords[0]] = df[coords[0]] * 105 / field_length
        df[coords[1]] = df[coords[1]] * 68 / field_width
    return df


def unscale_coordinates(df, xy_columns, field_length, field_width):
    """
    Unscales the coordinates in xy_columns to original coordinates (field_length x field_width).

    :param df: Pandas DataFrame that contains coordinates columns to be unscaled.
    :param xy_columns: List of tuples with pairs of coordinates (x, y) column names.
    :param field_length: Either a scalar of Pandas Series with a scaled length of the pitch.
    :param field_width: Either a scalar or Pandas Series with a scaled width of the pitch.
    :return: Pandas DataFrame with coordinates in xy_columns unscaled to an original size of the pitch.
    """
    df = df.copy()
    for coords in xy_columns:
        df[coords[0]] = df[coords[0]] * field_length / 105
        df[coords[1]] = df[coords[1]] * field_width / 68
    return df


def standardise_direction_of_play(df, coords_scaled=True):
    df = df.copy()

    if coords_scaled:
        df.loc[df['direction_of_play'] == 'Right to left', 'start_x'] = \
            105 - df.loc[df['direction_of_play'] == 'Right to left', 'start_x']
        df.loc[df['direction_of_play'] == 'Right to left', 'end_x'] = \
            105 - df.loc[df['direction_of_play'] == 'Right to left', 'end_x']
        df.loc[df['direction_of_play'] == 'Right to left', 'start_y'] = \
            68 - df.loc[df['direction_of_play'] == 'Right to left', 'start_y']
        df.loc[df['direction_of_play'] == 'Right to left', 'end_y'] = \
            68 - df.loc[df['direction_of_play'] == 'Right to left', 'end_y']
    else:
        df.loc[df['direction_of_play'] == 'Right to left', 'start_x'] = \
            df.loc[df['direction_of_play'] == 'Right to left', 'field_length'] - \
            df.loc[df['direction_of_play'] == 'Right to left', 'start_x']
        df.loc[df['direction_of_play'] == 'Right to left', 'end_x'] = \
            df.loc[df['direction_of_play'] == 'Right to left', 'field_length'] - \
            df.loc[df['direction_of_play'] == 'Right to left', 'end_x']
        df.loc[df['direction_of_play'] == 'Right to left', 'start_y'] = \
            df.loc[df['direction_of_play'] == 'Right to left', 'field_width'] - \
            df.loc[df['direction_of_play'] == 'Right to left', 'start_y']
        df.loc[df['direction_of_play'] == 'Right to left', 'end_y'] = \
            df.loc[df['direction_of_play'] == 'Right to left', 'field_width'] - \
            df.loc[df['direction_of_play'] == 'Right to left', 'end_y']

    return df


def unstandardise_direction_of_play(df, coords_scaled=True):
    df = df.copy()

    if coords_scaled:
        df.loc[df['direction_of_play'] == 'Right to left', 'start_x'] = \
            105 - df.loc[df['direction_of_play'] == 'Right to left', 'start_x']
        df.loc[df['direction_of_play'] == 'Right to left', 'end_x'] = \
            105 - df.loc[df['direction_of_play'] == 'Right to left', 'end_x']
        df.loc[df['direction_of_play'] == 'Right to left', 'start_y'] = \
            68 - df.loc[df['direction_of_play'] == 'Right to left', 'start_y']
        df.loc[df['direction_of_play'] == 'Right to left', 'end_y'] = \
            68 - df.loc[df['direction_of_play'] == 'Right to left', 'end_y']
    else:
        df.loc[df['direction_of_play'] == 'Right to left', 'start_x'] = \
            df.loc[df['direction_of_play'] == 'Right to left', 'field_length'] - \
            df.loc[df['direction_of_play'] == 'Right to left', 'start_x']
        df.loc[df['direction_of_play'] == 'Right to left', 'end_x'] = \
            df.loc[df['direction_of_play'] == 'Right to left', 'field_length'] - \
            df.loc[df['direction_of_play'] == 'Right to left', 'end_x']
        df.loc[df['direction_of_play'] == 'Right to left', 'start_y'] = \
            df.loc[df['direction_of_play'] == 'Right to left', 'field_width'] - \
            df.loc[df['direction_of_play'] == 'Right to left', 'start_y']
        df.loc[df['direction_of_play'] == 'Right to left', 'end_y'] = \
            df.loc[df['direction_of_play'] == 'Right to left', 'field_width'] - \
            df.loc[df['direction_of_play'] == 'Right to left', 'end_y']

    return df


def extract_pitch_dimensions(file):
    xroot = et.parse(file).getroot()
    field_length = float(xroot.attrib['FieldLength']) / 100
    field_width = float(xroot.attrib['FieldWidth']) / 100

    return field_length, field_width


if __name__ == '__main__':
    games = []
    for file in tqdm(os.listdir('../raw-data/SportVU/event-data')):
        df = read_event_sportvu('../raw-data/SportVU/event-data/' + file)
        df = scale_coordinates(df, [('location_x', 'location_y'), ('target_x', 'target_y')],
                               df['field_length'], df['field_width'])
        df = add_helpers(df)
        df = define_possession(df)
        df = add_start_coordinates(df)
        df = add_end_coordinates(df)
        df = add_offside_flag(df)
        df = add_interception_flag(df)
        df = add_freekick_direct_flag(df)
        df = add_freekick_indirect_flag(df)
        df = add_throw_in_flag(df)
        df = add_corner_flag(df)
        df = add_penalty_flag(df)
        df = add_goal_kick_flag(df)
        games.append(df)

    for game in tqdm(games):
        game['outcome'] = game\
            .groupby(['id_match', 'id_half'], group_keys=False) \
            .apply(lambda x: x.apply(determine_outcome, axis=1))

    with open('all_games_events.pkl', 'wb') as f:
        pickle.dump(games, f)
