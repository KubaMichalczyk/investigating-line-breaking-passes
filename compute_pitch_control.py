import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob
from scipy.spatial.distance import euclidean
from visualise import draw_pitch
from read_data import *
from joblib import Parallel, delayed
import multiprocessing


def radius_calc(dist_to_ball):
    return np.minimum(4 + dist_to_ball ** 3 / (18 ** 3 / 6), 10)


class Controller:
    """This is the adjusted and corrected version of the original implementation from: https://www.kaggle.com/pednt9/vip-hint-coded"""

    def __init__(self, frame_df, team_in_possession):
        self.frame = frame_df
        self.team_in_possession = team_in_possession
        self.vec_influence = np.vectorize(self.compute_influence)
        self.vec_control = np.vectorize(self.pitch_control)

    def compute_influence(self, x_point, y_point, player_id):
        """Computes the influence of a certain player over a coordinate (x, y) of the pitch"""

        point = np.array([x_point, y_point])
        player_row = self.frame.loc[player_id]
        theta = np.nan_to_num(np.arccos(player_row['dx'] / np.linalg.norm(player_row[['dx', 'dy']])))
        speed = np.linalg.norm(player_row[['dx', 'dy']].values) / 0.1
        player_coords = player_row[['x', 'y']].values
        ball_coords = self.frame.loc[self.frame['team'] == 'ball', ['x', 'y']].values
        if ball_coords.size == 0:
            ball_coords = np.array([np.nan, np.nan]).reshape(1, -1)

        dist_to_ball = euclidean(player_coords, ball_coords)

        S_ratio = (speed / 13) ** 2  # we set max_speed to 13 m/s
        RADIUS = radius_calc(dist_to_ball)

        S_matrix = np.array([[RADIUS * (1 + S_ratio) / 2, 0], [0, RADIUS * (1 - S_ratio) / 2]])
        R_matrix = np.array([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        COV_matrix = np.dot(np.dot(np.dot(R_matrix, S_matrix), S_matrix), np.linalg.inv(R_matrix))

        norm_fact = (1 / 2 * np.pi) * (1 / np.sqrt(np.linalg.det(COV_matrix)))
        mu_play = player_coords + speed * np.array([np.cos(theta), -np.sin(theta)]) / 2

        intermed_scalar_player = np.asarray(np.dot(np.dot((player_coords - mu_play), np.linalg.inv(COV_matrix)),
                                                   np.transpose((player_coords - mu_play)))).reshape(1, 1)
        player_influence = norm_fact * np.exp(- 0.5 * intermed_scalar_player[0, 0])

        intermed_scalar_point = np.asarray(np.dot(np.dot((point - mu_play), np.linalg.inv(COV_matrix)),
                                                  np.transpose((point - mu_play)))).reshape(1, 1)
        point_influence = norm_fact * np.exp(- 0.5 * intermed_scalar_point[0, 0])

        return point_influence / player_influence

    def pitch_control(self, x_point, y_point):
        """Computes the pitch control values for team in possession over a coordinate (x, y)"""

        if self.team_in_possession == 'home':
            team_out_of_possession = 'away'
        elif self.team_in_possession == 'away':
            team_out_of_possession = 'home'
        else:
            ValueError

        attacking_ids = self.frame.loc[self.frame['team'] == self.team_in_possession].index
        attacking_control = self.vec_influence(x_point, y_point, attacking_ids)
        attacking_score = np.sum(attacking_control)

        defending_ids = self.frame.loc[self.frame['team'] == team_out_of_possession].index
        defending_control = self.vec_influence(x_point, y_point, defending_ids)
        defending_score = np.sum(defending_control)

        return 1 / (1 + np.exp(-(attacking_score - defending_score)))

    def display_control(self, grid_size=(30, 15), figsize=(11, 7)):
        x, y = np.meshgrid(np.arange(0.5, 105.5, 1), np.arange(0.5, 68.5, 1))
        # infl is an array of shape num_points with values in [0,1] accounting for the pitch control
        infl = self.vec_control(x, y)

        fig, ax = draw_pitch()
        ax = ax.imshow(infl, vmin=0., vmax=1., cmap=plt.cm.coolwarm, origin='lower',
                       extent=[x.min(), x.max(), y.min(), y.max()])
        colours = {'away': 'blue', 'home': 'red', 'ball': 'black'}
        ax.scatter(self.frame['x'].values, self.frame['y'].values, c=self.frame['team'].map(colours),
                   s=self.frame['team'].map({'away': 30, 'home': 30, 'ball': 10}), zorder=20)
        for i, player in self.frame.iterrows():
            if player['team'] != 'ball':
                ax.add_patch(plt.arrow(player['x'], player['y'], player['dx'] * 10, player['dy'] * 10,
                                       head_width=1, zorder=10, lw=0.1, color=colours[player['team']]))
        plot = ax.pcolor(infl, cmap=plt.cm.coolwarm, vmin=0., vmax=1.)
        cb = fig.colorbar(plot, fraction=0.0278, pad=0, orientation='horizontal')
        cb.outline.set_edgecolor('white')
        cb.ax.tick_params(labelsize=7, color='white', labelcolor='black')
        for l in cb.ax.yaxis.get_ticklabels():
            l.set_family('PT Sans')
        fig.show()


if __name__ == '__main__':
    with open('all_games_events.pkl', 'rb') as file:
        all_games_events = pickle.load(file)


    def calculate_actions_pitch_control(event_df):
        actions_df = filter_actions(event_df.copy())
        actions_df.loc[:, 'frame'] = actions_df.loc[:, 'time'].round(-2)
        game_id = actions_df['id_match'].values[0]
        actions_df = actions_df.set_index(['id_half', 'frame'])
        idx = actions_df.index
        try:
            tracking_file = glob.glob('./jenks-clusters/*' + game_id + '*')[0]
            player_file = glob.glob('../raw-data/SportVU/player-data/*' + game_id + '*')[0]
            tracking_data = pd.read_parquet(tracking_file).reset_index('object')
            tracking_data = add_movement(tracking_data)
            tracking_data['player_id'] = np.where(tracking_data['team'] == 'ball', '-9999', tracking_data['player_id'])
            player_data = read_playerinfo_sportvu(player_file)
            actions_df = actions_df \
                .reset_index() \
                .merge(player_data[['player_id_event', 'player_id_tracking']],
                       left_on='id_actor', right_on='player_id_event', how='left') \
                .set_index(['id_half', 'frame'])
            tracking_data = tracking_data[['object', 'player_id', 'team', 'x', 'y', 'dx', 'dy']].loc[idx]
            # Creating unique index (named later as unique_id) is needed for different handling of two events within
            # the same frame
            tracking_data = tracking_data.set_index(tracking_data.groupby([tracking_data.index, 'object']).cumcount(),
                                                    append=True)
            actions_df.set_index(actions_df.groupby(actions_df.index).cumcount(), append=True, inplace=True)
            tracking_data['team_in_possession'] = actions_df['team']
            tracking_data.index.names = ['id_half', 'frame', 'unique_id']
            actions_df.index.names = ['id_half', 'frame', 'unique_id']

            controllers = tracking_data \
                .groupby(['id_half', 'frame', 'unique_id']) \
                .apply(lambda frame_df: Controller(frame_df
                                                   .reset_index('unique_id', drop=True)
                                                   .set_index('player_id', append=True),
                                                   frame_df['team_in_possession'][0]))

            actions_df['controller'] = controllers
            actions_df['start_pitch_control'] = actions_df \
                .apply(lambda x: x['controller'].pitch_control(*x[['start_x', 'start_y']].values.reshape(-1)), axis=1)

            if 'next_action_frame' not in actions_df.columns and 'next_action_frame' not in actions_df.index.names:
                actions_df.loc[:, 'next_action_frame'] = actions_df.loc[:, 'next_action_time'].round(-2)

            actions_df = actions_df.reset_index().set_index(['id_half', 'next_action_frame'])
            actions_df.set_index(actions_df.groupby(actions_df.index).cumcount(), append=True, inplace=True)
            actions_df['controller'] = controllers

            mask = actions_df['controller'].notna()
            actions_df.loc[mask, 'end_pitch_control'] = actions_df.loc[mask, :] \
                .apply(lambda x: x['controller'].pitch_control(*x[['end_x', 'end_y']].values.reshape(-1)), axis=1)

            return actions_df.reset_index()[['id_match', 'event_id', 'start_pitch_control', 'end_pitch_control']]

        except IndexError:
            return None

        except Exception as e:
            print(game_id, e)

    num_cores = multiprocessing.cpu_count()
    all_games_actions_pitch_control = Parallel(n_jobs=num_cores, verbose=10)(delayed(calculate_actions_pitch_control)(game_events)
                                                                             for game_events in all_games_events)
    all_games_actions_pitch_control = [game_events for game_events in all_games_actions_pitch_control
                                       if game_events is not None]

    pickle.dump(all_games_actions_pitch_control, open('all_games_actions_pitch_control_jenks.pkl', 'wb'))
