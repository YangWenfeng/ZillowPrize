import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint

"""
http://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size/
"""

def preprocess_raw_latlng(raw_df):
    new_df = raw_df.copy()

    new_df = new_df[['latitude', 'longitude']] / 1e6

    A = new_df['latitude'].isnull()
    B = new_df['longitude'].isnull()

    new_df['latitude'][B] = np.NAN
    new_df['longitude'][A] = np.NAN

    return new_df

def get_coordinates(df):
    new_df = df[['latitude', 'longitude']]

    new_df = new_df[new_df['latitude'].notnull()]
    coordinates = new_df.as_matrix(columns=['latitude', 'longitude'])

    return coordinates

def cluster_latlng(coordinates, m_distance, min_samples=1):
    kms_per_radian = 6371.0088
    epsilon = m_distance / 1000.0 / kms_per_radian
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree',
                    metric='haversine').fit(np.radians(coordinates))

    cluster_labels = dbscan.labels_
    num_clusters = len(set(cluster_labels))
    print 'DBSCAN [m_distance=%d, min_samples%d] cluster %d down to %d clusters' % (
        m_distance, min_samples, len(coordinates), num_clusters)

    return dbscan

def get_centroid_dict(dbscan, coordinates):
    cluster_label = dbscan.labels_
    num_clusters = len(set(cluster_label))
    clusters = pd.Series([coordinates[cluster_label == n] for n in xrange(num_clusters)])

    def get_centermost_point(cluster):
        centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
        centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
        return tuple(centermost_point)

    print 'Find the point in each cluster that is closest to its centroid.'
    centermost_points = clusters.map(get_centermost_point)

    # unzip the list of centermost points (latitude, longitude) tuples into separate latitude and longitude lists
    latitude, longitude = zip(*centermost_points)

    ret = dict()
    for label in cluster_label:
        ret[label] = (latitude[label], longitude[label], )
    return ret

def replace_predict_cluster_df(dbscan, centroid_dict, df):
    new_df = df.copy()

    for i in new_df.index:
        lat, lng = new_df['latitude'][i], new_df['longitude'][i]

        if np.isnan(lat) or np.isnan(lng):
            continue

        X = [[lat, lng]]
        label = dbscan.fit_predict(X)[0]
        new_lat, new_lng = centroid_dict[label]
        new_df['latitude'][i] = new_lat
        new_df['longitude'][i] = new_lng

    return new_df


if __name__ == "__main__":
    import common_utils as cu

    # read train data.
    X, y = cu.get_train_data(encode_non_object=False)

    # preprocess lat&lng
    print 'Preprocess latitude & longitude.'
    X = preprocess_raw_latlng(X)

    print 'Run DBSCAN.'
    coordinates = get_coordinates(X)
    dbscan = cluster_latlng(coordinates, m_distance=1500, min_samples=1)
    centroid_dict = get_centroid_dict(dbscan, coordinates)

    print 'Predict '
    X = replace_predict_cluster_df(dbscan, centroid_dict, X)

    print 'Done!!'
