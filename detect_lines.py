import glob
import os
from read_data import *
from pandarallel import pandarallel
from jenkspy import jenks_breaks


class LineDetector:

    def __init__(self, frame_df, n_classes=3, min_gvf=1):
        self.frame_df = frame_df
        self.n_classes = n_classes
        self.min_gvf = min_gvf
        self.fit = self.cluster_team

    def cluster_team(self, x):
        if not isinstance(self.n_classes, list):
            n_classes = [self.n_classes]
        else:
            n_classes = self.n_classes
        array = self.frame_df.loc[self.frame_df[x].notnull(), x]

        for n in n_classes:

            if len(array) < n:
                return pd.Series(pd.NA, index=self.frame_df.index)

            else:
                labels1, gvf1 = self._assign_classes(array, n, include_lowest=True)
                labels2, gvf2 = self._assign_classes(array, n, include_lowest=False)

                gvf = max(gvf1, gvf2)
                if gvf < self.min_gvf and n != n_classes[-1]:
                    continue
                else:
                    if gvf1 > gvf2:
                        labels = labels1
                    else:
                        labels = labels2

                    return pd.Series(labels, index=self.frame_df.index)

    def _assign_classes(self, array, n_classes, include_lowest=True):
        """Deals with the issue of assigning border cases."""
        array = np.array(array)

        bins = np.array(jenks_breaks(array, n_classes))

        if include_lowest:
            bins[0] = np.nextafter(bins[0], -999)
            classes = np.digitize(array, bins=bins, right=True)
        else:
            bins[n_classes] = np.nextafter(bins[n_classes], 999)
            classes = np.digitize(array, bins=bins, right=False)

        gvf = self._goodness_of_variance(array, classes)

        return classes, gvf

    def _goodness_of_variance(self, array, classes):

        sdam = np.sum((array - np.mean(array)) ** 2)
        class_means = [np.mean(array[classes == i]) for i in classes]
        sdcm = np.sum((array - class_means) ** 2)
        gvf = 1 - sdcm / sdam

        return gvf


def make_clusters(df, n_classes, order=False, parallel=True, **kwargs):
    """
    Takes tracking data in the long format and returns

    This still needs to be updated to incorporate differences fot team attacking from right to left.
    """
    if not isinstance(n_classes, list):
        n_classes = [n_classes]

    # To prevent immediate switching the line we average the x coordinate over 2 seconds.
    df['x_mean'] = df['x'] \
        .groupby(['id_half', 'object']) \
        .rolling(20, min_periods=1) \
        .mean() \
        .reset_index([0, 1], drop=True) \
        .sort_index(level=['id_half', 'frame', 'object'])

    df['team'] = df['team_id'].replace({'0': 'home', '3': 'home', '1': 'away', '4': 'away', '99': 'ball'})

    df['gk'] = df['team_id'].isin(['3', '4'])

    clusters = df \
        .set_index('team_id', append=True) \
        .groupby(['id_half', 'frame', 'team_id']) \
        .filter(lambda x: x.shape[0] > 3)

    if parallel:
        pandarallel.initialize(use_memory_fs=False, **kwargs)

        # index and part are helper columns to pass lesser chunks to parallel_apply function - pandarallel is blazing
        # fast but apparently produces many internal data copies as it can run out of RAM very quickly
        clusters['index'] = clusters.groupby(['id_half', 'frame', 'team_id']).ngroup()
        clusters['part'] = pd.cut(clusters['index'], bins=16)
        groups = []
        for i, group in enumerate(clusters.groupby('part')):
            print(i)
            groups.append(group[1]
                .groupby(['id_half', 'frame', 'team_id'])
                .parallel_apply(lambda teamframe_df: LineDetector(teamframe_df, n_classes=3).fit('x_mean'))
                .to_frame(name='line'))
        clusters = pd.concat(groups)
    else:
        clusters = clusters \
            .groupby(['id_half', 'frame', 'team_id'], group_keys=False) \
            .apply(lambda teamframe_df: LineDetector(teamframe_df, n_classes=3).fit('x_mean')) \
            .to_frame(name='line')
    df_clustered = df.merge(clusters, how='left', left_on=['id_half', 'frame', 'object'],
                            right_on=['id_half', 'frame', 'object'])

    # Create separate clusters for GKs (marked as 0 for now) and mark unclustered objects (marked as -1)
    df_clustered['line'] = np.where(df_clustered['gk'] == True, 0, df_clustered['line'])
    df_clustered['line'].fillna(-1, inplace=True)

    if order:
        # Sorting clusters by the mean of x coordinate in order to assign constant order of cluster numbering (colours)
        # later.
        df_clustered = df_clustered \
            .assign(cluster_centre=lambda x: x.groupby(['id_half', 'frame', 'team', 'line'])['x'].transform('mean')) \
            .groupby(['id_half', 'frame', 'team'], group_keys=False) \
            .apply(lambda x: x.sort_values('cluster_centre', na_position='first'))

        df_clustered['line'] = df_clustered \
            .groupby(['id_half', 'frame', 'team'], group_keys=False) \
            .apply(lambda x: x['line'].replace({**dict({99: 99, -1: -1}),
                                                **dict(zip(x['line'][x['line'].isin(list(range(0, len(n_classes) + 1)))].unique(),
                                                           list(range(0, len(n_classes) + 1))))}))

    # To prevent momentaneous, incorrect clusters we are removing assignments that last for less than 1 second.
    line_grpd = df_clustered.groupby(['id_half', 'object'])['line']
    df_clustered['tmp'] = (line_grpd.shift(0) != line_grpd.shift(1))
    df_clustered['rleid'] = df_clustered \
        .groupby('object')['tmp'] \
        .cumsum() \
        .astype(int)
    df_clustered.drop('tmp', axis=1, inplace=True)

    time_in_line = df_clustered \
        .reset_index() \
        .groupby(['object', 'rleid', 'line']) \
        .apply(lambda x: x['frame'].iat[-1] - x['frame'].iat[0] + 100) \
        .to_frame('time_in_line') \
        .reset_index('line')
    time_in_line['prev_time_in_line'] = time_in_line \
        .groupby('object', group_keys=False) \
        .apply(lambda x: x['time_in_line'].shift(1))
    time_in_line['next_time_in_line'] = time_in_line \
        .groupby('object', group_keys=False) \
        .apply(lambda x: x['time_in_line'].shift(-1))

    time_in_line['last_valid_line'] = np.where(time_in_line['time_in_line'] >= 1000, time_in_line['line'], np.nan)
    time_in_line['last_valid_line'] = time_in_line.groupby('object')['last_valid_line'].ffill()

    time_in_line['prev_line'] = time_in_line \
        .groupby('object', group_keys=False) \
        .apply(lambda x: x['line'].shift(1))
    time_in_line['next_line'] = time_in_line \
        .groupby('object', group_keys=False) \
        .apply(lambda x: x['line'].shift(-1))

    time_in_line['new_line'] = np.where(time_in_line['time_in_line'] < 1000,
                                        time_in_line['last_valid_line'],
                                        time_in_line['line'])

    df_clustered = df_clustered \
        .reset_index() \
        .merge(time_in_line, how='left', on=['object', 'line', 'rleid']) \
        .set_index(['id_half', 'frame', 'object'])

    df_clustered['line'] = df_clustered['new_line']
    df_clustered.drop('new_line', axis=1, inplace=True)

    return df_clustered


if __name__ == '__main__':

    tracking_dir = '../raw-data/SportVU/tracking-data/'
    for tracking_file in tqdm(os.listdir(tracking_dir)):
        print(tracking_file)
        tracking_data = tracking_to_long(read_tracking_sportvu(tracking_dir + tracking_file))
        game_id = re.findall('[0-9]{7}', tracking_file)[0]
        event_file = glob.glob('../raw-data/SportVU/event-data/*' + game_id + '.xml')[0]
        tracking_data = scale_coordinates(tracking_data, [('x', 'y')], *extract_pitch_dimensions(event_file))
        tracking_data_clustered = make_clusters(tracking_data, n_classes=3, order=True,
                                                parallel=True, nb_workers=12, progress_bar=True)
        tracking_data_clustered.to_parquet('./jenks-clusters/' + tracking_file.split('.')[0] + '.parquet')
