from itertools import compress
import pandas as pd
import pickle
from visualise import *
import glob
from tqdm import tqdm
from joblib import Parallel, delayed


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def orientation(A, B, C):
    """
    https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    """
    value = (B[1] - A[1]) * (C[0] - B[0]) - (B[0] - A[0]) * (C[1] - B[1])
    if value > 0:
        return 1
    elif value == 0:
        return 0
    else:
        return -1


def intersect(pass_start, pass_end, segment_start, segment_end):
    return ccw(pass_start, segment_start, segment_end) != ccw(pass_end, segment_start, segment_end) and \
           ccw(pass_start, pass_end, segment_start) != ccw(pass_start, pass_end, segment_end)


def point_of_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def create_segments(points):
    """
    Takes a list of points (x, y) and returns segments between subsequent points.
    """
    points.sort(key=lambda p: p[1])
    segments = []
    if len(points) > 1:
        for i in range(len(points) - 1):
            segments.append((points[i], points[i + 1]))
    return segments


def df_to_points(df):
    points = []
    for index, row in df.iterrows():
        points.append(tuple(row))
    return points


def filter_segments(_pass, points):
    """
    This is to reduce the number of segments to check.
    """
    pass_y_min = min(_pass[0][1], _pass[1][1])
    pass_y_max = max(_pass[0][1], _pass[1][1])

    y_points = np.array([p[1] for p in points])
    p_min = y_points[0:-1]
    p_max = y_points[1:]
    to_check = ~((pass_y_max < p_min) | (p_max < pass_y_min))
    segments = list(compress(create_segments(points), to_check))

    return segments


def is_breaking(_pass, segments, direction_of_play):
    segments = np.array(segments)
    if direction_of_play == 'Right to left':
        if _pass[1][0] - _pass[0][0] >= -10:  # pass at least 10m forward
            return False
        elif _pass[1][0] > segments[:, :, 0].min() - 2:  # at least 2m past the last line member
            return False
    elif direction_of_play == 'Left to right':
        if _pass[1][0] - _pass[0][0] <= 10:
            return False
        elif _pass[1][0] < segments[:, :, 0].max() + 2:
            return False
    else:
        raise ValueError

    for segment in segments:
        if not intersect(_pass[0], _pass[1], segment[0], segment[1]):
            return False
        else:
            poi = point_of_intersection(_pass, segment)
            dist_to_poi = np.sqrt((poi[0] - _pass[0][0]) ** 2 + (poi[1] - _pass[0][0]) ** 2)
            if dist_to_poi > 5:  # pass start at least 5m away from the point of intersection with line
                return True

    return False


def is_line_breaking(_pass, lines, direction_of_play):
    """
    Takes list of lines where each line is list of points (x, y).

    Example input:
    _pass = ((60, 53), (37, 52))
    lines = [[(7.0, 36.0)],
     [(40.0, 14.0), (39.0, 44.0), (43.0, 57.0), (38.0, 31.0)],
     [(52.0, 26.0), (58.0, 60.0), (54.0, 51.0), (47.0, 41.0)],
     [(65.0, 52.0), (65.0, 32.0)]]
    """
    counter = 0
    broken_lines = []

    for i, line in enumerate(lines):
        if len(line) > 1:
            segments = create_segments(line)
            if is_breaking(_pass, segments, direction_of_play):
                counter += 1
                broken_lines.append(line)
    return counter, broken_lines


def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def angle_to_ball(player, ball):
    return np.arctan2(-(ball[1] - player[1]), np.abs(ball[0] - player[0]))


def aov(ball, segment):
    theta1 = angle_to_ball(segment[0], ball)
    theta2 = angle_to_ball(segment[1], ball)
    return theta2 - theta1


def add_tracking_features(_pass, lines, direction_of_play):
    line = find_first_line(_pass, lines, direction_of_play)
    if len(line) > 1:
        segments = create_segments(line)
        line_integrity = 0
        line_compactness = 0
        a = 0
        d = 0
        for segment in segments:
            line_integrity += max(0, np.nan_to_num(1 / aov(_pass[0], segment)))
            line_compactness += 1 / dist(segment[0], segment[1])
            a = max(a, aov(_pass[0], segment))
            d = max(d, dist(segment[0], segment[1]))
            # if intersect(_pass[0], _pass[1], segment[0], segment[1]):
            #     a = aov(_pass, segment)
            #     d = dist(segment[0], segment[1])
            # else:
            #     a = np.nan
            #     d = np.nan
    else:
        a = np.nan
        d = np.nan
        line_integrity = np.nan
        line_compactness = np.nan
    return a, d, line_integrity, line_compactness


def find_first_line(_pass, lines, direction_of_play):
    line_heights = np.array([np.mean(line[:, 0]) for line in np.array([np.array(line) for line in lines])])
    if direction_of_play == 'Left to right':
        diff = line_heights - _pass[0][0]
        lines = np.flip(lines)
        diff = np.flip(diff)
        diff = diff[diff > 0]
        try:
            return lines[np.argmin(diff)]
        except ValueError:
            return []
    elif direction_of_play == 'Right to left':
        diff = _pass[0][0] - line_heights
        diff = diff[diff > 0]
        try:
            return lines[np.argmin(diff)]
        except ValueError:
            return []
    else:
        raise ValueError
        
        
def add_breaking_line_info(passes_df, tracking_df):
    passes_df.loc[:, 'defending_team'] = passes_df.loc[:, 'team'].replace({'away': 'home', 'home': 'away'})
    passes_df.loc[:, 'frame'] = passes_df.loc[:, 'time'].round(-2)
    passes_df = passes_df.astype({'id_half': 'int64'})
    passes = passes_df \
        .merge(tracking_df.reset_index(), how='left', on=['id_half', 'frame']) \
        .groupby(['id_half', 'frame', 'time', 'id_actor1', 'defending_team', 'direction_of_play'], group_keys=False) \
        .apply(lambda x: ((x['start_x'].unique().item(), x['start_y'].unique().item()),
                          (x['end_x'].unique().item(), x['end_y'].unique().item()))) \
        .to_frame('_pass') \
        .reset_index(['time', 'id_actor1', 'defending_team', 'direction_of_play'])

    lines = tracking_df
    lines['coords'] = lines[['x', 'y']].values.tolist()
    lines = lines.pivot_table(index=['id_half', 'frame', 'team'], columns=['line'], values=['coords'], aggfunc=list)
    lines = lines.apply(lambda x: list(x[x.notna()]), axis=1)

    idx = pd.IndexSlice
    passes_grpd = passes \
        .join(lines.to_frame('line'), how='left') \
        .groupby(['id_half', 'frame', 'id_actor1', 'time'])
    passes_breaking_series = passes_grpd \
        .apply(lambda x: is_line_breaking(x['_pass'][0], x.loc[idx[:, :, x['defending_team']], 'line'][0],
                                          x['direction_of_play'][0]))
    passes_tracking_features_series = passes_grpd \
        .apply(lambda x: add_tracking_features(x['_pass'][0],
                                               x.loc[idx[:, :, x['defending_team']], 'line'][0],
                                               x['direction_of_play'][0]))

    passes_breaking = pd.DataFrame()
    passes_breaking['n_broken_lines'], passes_breaking['broken_lines'] = zip(*passes_breaking_series)
    passes_breaking.set_index(passes_breaking_series.index, inplace=True)

    passes_tracking_features = pd.DataFrame()
    passes_tracking_features['a'], passes_tracking_features['d'], passes_tracking_features['line_integrity'], \
        passes_tracking_features['line_compactness'] = zip(*passes_tracking_features_series)
    passes_tracking_features.set_index(passes_tracking_features_series.index, inplace=True)

    passes_df = passes_df \
        .merge(passes_breaking, how='left', on=['id_half', 'frame', 'id_actor1', 'time']) \
        .merge(passes_tracking_features, how='left', on=['id_half', 'frame', 'id_actor1', 'time'])

    passes_df['line_breaking'] = (passes_df['n_broken_lines'] > 0).astype('int')

    return passes_df


def draw_pass(passes_df, tracking_df, half_value, frame, defending_team):
    fig, ax = draw_lines(tracking_df,
                         half_id='id_half',
                         frame_id='frame',
                         object_id='object',
                         team_id='team_id',
                         half_value=half_value,
                         defending_team=defending_team,
                         t=frame / 1000,
                         fps=10,
                         display_num=False)[0:2]
    ax.add_line(
        plt.Line2D([float(passes_df.loc[(passes_df['id_half'] == half_value) & (passes_df['frame'] == frame), 'start_x']),
                    float(passes_df.loc[(passes_df['id_half'] == half_value) & (passes_df['frame'] == frame), 'end_x'])],
                   [float(passes_df.loc[(passes_df['id_half'] == half_value) & (passes_df['frame'] == frame), 'start_y']),
                    float(passes_df.loc[(passes_df['id_half'] == half_value) & (passes_df['frame'] == frame), 'end_y'])],
                   c='red', zorder=10))

    return fig, ax


def draw_action(tracking_df, half_value, frame, defending_team, before=2000, after=5000):
    df = tracking_df.loc[pd.IndexSlice[half_value, (frame - before):(frame + after)], :]
    plt.interactive(False)
    anim = VideoClip(lambda x: mplfig_to_npimage(draw_lines(df, 'id_half', 'frame', 'object', 'team_id',
                                                            half_value, x, defending_team=defending_team, fps=10,
                                                            display_num=False)[0]),
                     duration=df.index.get_level_values('frame').unique().shape[0] / 10)
    anim.write_videofile('./videos/frame_' + str(frame) + '_action.mp4', fps=10, threads=4)


if __name__ == '__main__':

    with open('all_passes_imputed.pkl', 'rb') as file:
        all_passes = pickle.load(file)

    def process_game_passes(game_passes):
        game_id = game_passes.iloc[0]['id_match']
        try:
            tracking_data = pd.read_parquet(glob.glob('./jenks-clusters/*' + game_id + '.parquet')[0])
            game_passes = add_breaking_line_info(game_passes, tracking_data)
            return game_passes
        except IndexError:
            return None

    all_passes = Parallel(n_jobs=12, verbose=10)(delayed(process_game_passes)(game_passes) for game_passes in all_passes)
    all_passes = [game_passes for game_passes in all_passes if game_passes is not None]

    with open('all_passes_with_line_breaking_info_jenks.pkl', 'wb') as file:
        pickle.dump(all_passes, file)

    # game_id = all_passes[0].iloc[0]['id_match']
    # tracking_data = pd.read_parquet(glob.glob('./processed-tracking-data/*' + game_id + '.parquet')[0])
    # game_passes = add_breaking_line_info(all_passes[0].iloc[:3], tracking_data)
