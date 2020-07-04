import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from matplotlib.colors import from_levels_and_colors
plt.interactive(False)
from matplotlib.patches import Ellipse
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from read_data import *
from shapely.geometry import Polygon

X_SIZE = 105.0
Y_SIZE = 68.0

BOX_HEIGHT = (16.5 * 2 + 7.32)
BOX_WIDTH = 16.5

GOAL = 7.32

GOAL_AREA_HEIGHT = 5.4864 * 2 + GOAL
GOAL_AREA_WIDTH = 5.4864

# The following functions were build on top of draw_pitch(), draw_patches() and draw_frame() functions from:
# https://github.com/rjtavares/football-crunching/blob/master/notebooks/using%20voronoi%20diagrams.ipynb


def draw_pitch():
    """Sets up field
    Returns matplotlib fig and axes objects.
    """

    fig = plt.figure(figsize=(X_SIZE / 20, Y_SIZE / 20), dpi=100)
    fig.patch.set_facecolor('#DDDDDD')

    axes = fig.add_subplot(1, 1, 1, facecolor='#DDDDDD')

    plt.axis('off')

    axes.set_xlim(-5, 110)
    axes.set_ylim(73, -5)

    axes = draw_patches(axes)

    return fig, axes


def draw_pitch_with_swarm():
    """Sets up field with swarm plot of data projected onto x axis.
    Returns matplotlib fig and axes objects.
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, subplot_kw={'facecolor': '#DDDDDD'}, gridspec_kw={'height_ratios': [3, 1]},
                                   facecolor='#DDDDDD', figsize=(X_SIZE / 15, Y_SIZE / 15 * 4 / 3))
    ax1.set_xlim(-5, 110)
    ax1.set_ylim(73, -5)
    ax2.set_xlim(-5, 110)
    ax2.set_ylim(0.98, 1.02)

    ax1 = draw_patches(ax1)

    ax1.axis('off')
    ax2.axis('off')

    return fig, ax1, ax2


def draw_patches(axes):
    # pitch
    axes.add_patch(plt.Rectangle((0, 0), 105, 68, edgecolor="white", facecolor="none", alpha=1))

    # half-way line
    axes.add_line(plt.Line2D([52.5, 52.5], [68, 0], c='w'))

    # penalty areas
    axes.add_patch(plt.Rectangle((105 - BOX_WIDTH, (68 - BOX_HEIGHT) / 2), BOX_WIDTH, BOX_HEIGHT,
                                 ec='w', fc='none'))
    axes.add_patch(plt.Rectangle((0, (68 - BOX_HEIGHT) / 2), BOX_WIDTH, BOX_HEIGHT,
                                 ec='w', fc='none'))

    # goal areas
    axes.add_patch(
        plt.Rectangle((105 - GOAL_AREA_WIDTH, (68 - GOAL_AREA_HEIGHT) / 2), GOAL_AREA_WIDTH, GOAL_AREA_HEIGHT,
                      ec='w', fc='none'))
    axes.add_patch(plt.Rectangle((0, (68 - GOAL_AREA_HEIGHT) / 2), GOAL_AREA_WIDTH, GOAL_AREA_HEIGHT,
                                 ec='w', fc='none'))

    # goals
    axes.add_patch(plt.Rectangle((105, (68 - GOAL) / 2), 1, GOAL, ec='w', fc='none'))
    axes.add_patch(plt.Rectangle((0, (68 - GOAL) / 2), -1, GOAL, ec='w', fc='none'))

    # halfway circle
    axes.add_patch(Ellipse((52.5, 34), 2 * 9.15, 2 * 9.15, ec='w', fc='none'))

    return axes


def draw_frame(df, half_id, frame_id, object_id, team_id, half_value, t, fps=10,
               teams={'home': ['0', '3'], 'away': ['1', '4']}, colours={'home': '#BD3A4B', 'away': '#00529F'},
               defending_team='away', display_num=False):

    f = int(t * fps)

    fig, ax = draw_pitch()

    df = df.reset_index().set_index([half_id, frame_id, object_id])
    df = normalise_index(df)
    df_frame = df.loc[(half_value, f)]

    df_frame.sort_index(ascending=False, inplace=True)

    for pid in df_frame.index:
        if pid == 'ball':
            size = 0.6
            colour = 'black'
            edge = 'black'
        else:
            size = 3
            colour = 'white'
            if df_frame.loc[pid][team_id] in teams['home']:
                edge = colours['home']
            else:
                edge = colours['away']

        ax.add_artist(Ellipse((df_frame.loc[pid]['x'],
                               df_frame.loc[pid]['y']),
                              size, size,
                              edgecolor=edge,
                              linewidth=2,
                              facecolor=colour,
                              alpha=1,
                              zorder=20))
        if display_num:
            plt.text(df_frame.loc[pid, 'x'] - 1, df_frame.loc[pid, 'y'] - 1.3, str(df_frame.loc[pid, 'jersey_number']),
                     fontsize=8, color='black', zorder=30)

    return fig, ax, df_frame


def swarm_plot(ax, x, colour, labels=False):
    ax.clear()
    x = x[~np.isnan(x)]
    if len(x) > 0:
        kde = gaussian_kde(x)
        density = kde(x)
        np.random.seed(42)
        jitter = np.random.rand(*x.shape) - .5
        y = 1 + (density * jitter * .1 * 2)
        cmap, norm = from_levels_and_colors([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
                                            ['white', 'blue', '#BD3A4B', 'green', 'pink', 'orange'])
        ax.scatter(x, y, s=30, c=colour, cmap=cmap, norm=norm)

        if labels:
            for i, label in enumerate(x):
                ax.annotate(i, (x[i], y[i]))
        ax.set_xlim(-5, 110)
        ax.set_ylim(.98, 1.02)
    return ax


def draw_frame_with_swarm(df, half_id, frame_id, object_id, team_id, half_value, t, fps=10,
                          teams={'home': ['0', '3'], 'away': ['1', '4']}, colours={'home': 'gray', 'away': '#00529F'},
                          defending_team='away', display_num=False):
    f = int(t * fps)

    fig, ax1, ax2 = draw_pitch_with_swarm()
    df = df.reset_index().set_index([half_id, frame_id, object_id])
    df = normalise_index(df)
    df_frame = df.loc[(half_value, f)]
    df_frame.sort_index(ascending=False, inplace=True)

    for pid in df_frame.index:
        if pid == 'ball':
            size = 0.6
            colour = 'black'
            edge = 'black'
        else:
            size = 3
            colour = 'white'
            if df_frame.loc[pid][team_id] in teams['home']:
                edge = colours['home']
            else:
                edge = colours['away']

        ax1.add_artist(Ellipse((df_frame.loc[pid]['x'],
                                df_frame.loc[pid]['y']),
                               size, size,
                               edgecolor=edge,
                               linewidth=2,
                               facecolor=colour,
                               alpha=1,
                               zorder=20))
        if display_num:
            ax1.text(df_frame.loc[pid, 'x'] - 1, df_frame.loc[pid, 'y'] - 1.3, str(df_frame.loc[pid, 'jersey_number']),
                     fontsize=8, color='black', zorder=30)

        df_defenders = df_frame.loc[df_frame[team_id].isin(teams[defending_team])]
        ax2 = swarm_plot(ax2, df_defenders['x'], colour=df_defenders['line'])
    return fig, ax1, ax2, df_frame


def draw_lines(df, half_id, frame_id, object_id, team_id, half_value, t, fps=10,
               teams={'home': ['0', '3'], 'away': ['1', '4']}, colours={'home': '#BD3A4B', 'away': '#00529F'},
               defending_team='away', display_num=False):

    fig, ax, df_frame = draw_frame(df, half_id, frame_id, object_id, team_id, half_value, t, fps=fps, teams=teams,
                                   colours=colours, display_num=display_num)

    df_defenders = df_frame[df_frame[team_id].isin(teams[defending_team])]

    df_defenders = df_defenders.groupby('line').apply(lambda x: x.sort_values('y')).drop('line', axis=1)
    df_defenders['x_next'] = df_defenders.groupby('line')['x'].shift(-1)
    df_defenders['y_next'] = df_defenders.groupby('line')['y'].shift(-1)
    for (lid, pid) in df_defenders.index:
        ax.add_line(plt.Line2D([df_defenders.loc[(lid, pid), 'x'], df_defenders.loc[(lid, pid), 'x_next']],
                               [df_defenders.loc[(lid, pid), 'y'], df_defenders.loc[(lid, pid), 'y_next']],
                               c=colours[defending_team], zorder=10))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    return fig, ax


def normalise_index(df):
    indices = df.index.get_level_values('frame').unique().sort_values()

    dict = {}
    for key, value in (zip(indices, range(0, len(indices)))):
        dict[key] = value

    df = df.rename(index=dict)
    return df


def draw_swarm_plot(df, half_id, frame_id, object_id, team_id, half_value, t, fps=10,
                    teams={'home': ['0', '3'], 'away': ['1', '4']}, colours={'home': 'gray', 'away': '#00529F'},
                    defending_team='away', display_num=False):
    f = int(t * fps)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(-5, 110)
    ax.set_ylim(0.98, 1.02)
    ax.axis('off')

    df = df.reset_index().set_index([half_id, frame_id, object_id])
    df = normalise_index(df)
    df_frame = df.loc[(half_value, f)]

    df_frame.sort_index(ascending=False, inplace=True)

    df_defenders = df_frame.loc[df_frame[team_id].isin(teams[defending_team])]
    ax = swarm_plot(ax, df_defenders['x'], colour=df_defenders['line'])

    return fig, ax, df_frame


def draw_clustering(df, half_id, frame_id, object_id, team_id, half_value, t, fps=10,
                    teams={'home': ['0', '3'], 'away': ['1', '4']}, colours={'home': 'gray', 'away': '#00529F'},
                    defending_team='away', display_num=False):

    fig, ax, df_frame = draw_frame(df, half_id, frame_id, object_id, team_id, half_value, t, fps=fps, teams=teams,
                                   colours=colours, display_num=display_num)

    df_defenders = df_frame[df_frame[team_id].isin(teams[defending_team])]

    df_defenders = df_defenders \
        .groupby('line') \
        .apply(lambda x: x.sort_values('y')) \
        .drop('line', axis=1) \
        .reset_index(level='line')
    size = 3
    colors = {0.0: 'blue', 1.0: 'red', 2.0: 'green', 3.0: 'pink', 4.0: 'orange', 99.0: 'grey', -1.0: 'white'}
    for pid in df_defenders.index:
        color = df_defenders['line'].apply(lambda x: colors[x])[pid]
        ax.add_artist(Ellipse((df_frame.loc[pid]['x'],
                               df_frame.loc[pid]['y']),
                              size, size,
                              edgecolor=colours[defending_team],
                              linewidth=2,
                              facecolor=color,
                              alpha=1,
                              zorder=20))
        if display_num:
            plt.text(df_frame.loc[pid, 'x'] - 1, df_frame.loc[pid, 'y'] - 1.3, str(df_frame.loc[pid, 'jersey_number']),
                     fontsize=8, color='black', zorder=30)

    return fig, ax, df_frame


def draw_clustering_with_swarm(df, half_id, frame_id, object_id, team_id, half_value, t, fps=10,
                               teams={'home': ['0', '3'], 'away': ['1', '4']},
                               colours={'home': 'gray', 'away': '#00529F'},
                               defending_team='away', display_num=False):
    fig, ax1, ax2, df_frame = draw_frame_with_swarm(df, half_id, frame_id, object_id, team_id, half_value, t, fps=fps,
                                                    teams=teams, colours=colours, display_num=display_num)

    df_defenders = df_frame[df_frame[team_id].isin(teams[defending_team])]
    df_defenders = df_defenders \
        .groupby('line') \
        .apply(lambda x: x.sort_values('y')) \
        .drop('line', axis=1) \
        .reset_index(level='line')
    size = 3
    colors = {0.0: 'blue', 1.0: 'red', 2.0: 'green', 3.0: 'pink', 4.0: 'orange', 5.0: 'purple',
              99.0: 'grey', -1.0: 'white'}
    for pid in df_defenders.index:
        color = df_defenders['line'].apply(lambda x: colors[x])[pid]
        ax1.add_artist(Ellipse((df_frame.loc[pid]['x'],
                                df_frame.loc[pid]['y']),
                               size, size,
                               edgecolor=colours[defending_team],
                               linewidth=2,
                               facecolor=color,
                               alpha=1,
                               zorder=20))
        if display_num:
            ax1.text(df_frame.loc[pid]['x'] - 1, df_frame.loc[pid]['y'] - 1.3, str(df_frame.loc[pid]['jersey_number']),
                     fontsize=8, color='black', zorder=30)

        ax2 = swarm_plot(ax2, df_defenders['x_mean'], colour=df_defenders['line'])
    return fig, ax1, ax2, df_frame


def draw_voronoi(df, half_id, frame_id, object_id, team_id, half_value, t,
                 colours={'0': 'gray', '1': 'blue', '3': 'gray', '4': 'blue'},
                 display_num=False):

    fig, ax, df_frame = draw_frame(df, half_id=half_id, frame_id=frame_id, object_id=object_id,
                                   team_id=team_id, half_value=half_value, t=t, fps=10, display_num=display_num)
    df_voronoi, vor = calculate_voronoi(df_frame[df_frame['team_id'] != '99'])
    polygons = {}
    pitch = Polygon(((0, 0), (105, 0), (105, 68), (0, 68)))
    for index, region in enumerate(vor.regions):
        if not -1 in region:
            if len(region) > 0:
                try:
                    pl = df_voronoi[df_voronoi['region_id'] == index]
                    polygon = Polygon([vor.vertices[i] for i in region]).intersection(pitch)
                    colour = pl['team_id'].apply(lambda x: colours[x]).values[0]
                    x, y = polygon.exterior.xy
                    plt.fill(x, y, c=colour, alpha=0.30)
                    polygons[pl.index[0]] = polygon
                except IndexError:
                    pass
                except AttributeError:
                    pass
    return fig, ax
