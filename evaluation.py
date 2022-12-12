from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

from knn_reg import kNNRegression, gridSearch

import timeit


print('housing')
housing_df = pd.read_csv('./data/boston.csv')
housing_X = housing_df.drop('MEDV', axis=1)
housing_y = housing_df['MEDV']
print('\tKnnRegresor_Impl')

start = timeit.default_timer()
best_k = gridSearch(
        housing_X,
        housing_y,
        range(1,20),
        est_min=float('inf'),
        est_better=(lambda curr, max: curr < max)
    )
stop = timeit.default_timer()
print(f'\ttook {stop - start:.4f}')
print(f'\tbest k: {best_k:.4f}')

start = timeit.default_timer()
CV_knn_impl = cross_val_score(kNNRegression(neighbors=best_k), housing_X, housing_y, cv=5)
stop = timeit.default_timer()
print(f'\ttook {stop - start:.4f}')
print(f'\tCV of the self implemented KNN: mean {np.mean(CV_knn_impl):.4f} values {CV_knn_impl}')


print('\tKNeighborsRegressor')
start = timeit.default_timer()
CV_knn = cross_val_score(KNeighborsRegressor(), housing_X, housing_y, cv=5)
stop = timeit.default_timer()
print(f'\ttook {stop - start:.4f}')
print(f'\tCV of the sklearn KNN: mean {np.mean(CV_knn):.4f} values {CV_knn}')


print('\tRandomForest')
start = timeit.default_timer()
CV_tree = cross_val_score(RandomForestRegressor(), housing_X, housing_y, cv=5)
stop = timeit.default_timer()
print(f'\ttook {stop - start:.4f}')
print(f'\tCV of the sklearn RandomForest: mean {np.mean(CV_tree):.4f} values {CV_tree}')


print('song')
start = timeit.default_timer()
song_df = pd.read_csv('./data/song_data.csv')
enc = LabelEncoder().fit(song_df['song_name'])

song_df_tf = song_df.copy()
song_df_tf['song_name'] = enc.transform(song_df['song_name'])

song_X = song_df_tf.drop('song_popularity', axis=1)
song_y = song_df_tf['song_popularity']
start = timeit.default_timer()
best_k = 10
# best_k = gridSearch(
#         song_X,
#         song_y,
#         range(1,20),
#         est_min=float('inf'),
#         est_better=(lambda curr, max: curr < max)
#     )
stop = timeit.default_timer()
print(f'\ttook {stop - start:.4f}')
print(f'\tbest k: {best_k:.4f}')

start = timeit.default_timer()
CV_knn_impl = cross_val_score(kNNRegression(neighbors=best_k), song_X, song_y, cv=5)
stop = timeit.default_timer()
print(f'\ttook {stop - start:.4f}')
print(f'\tCV of the self implemented KNN: mean {np.mean(CV_knn_impl):.4f} values {CV_knn_impl}')


print('\tKNeighborsRegressor')
start = timeit.default_timer()
CV_knn = cross_val_score(KNeighborsRegressor(), song_X, song_y, cv=5)
stop = timeit.default_timer()
print(f'\ttook {stop - start:.4f}')
print(f'\tCV of the sklearn KNN: mean {np.mean(CV_knn):.4f} values {CV_knn}')


print('\tRandomForest')
start = timeit.default_timer()
CV_tree = cross_val_score(RandomForestRegressor(), song_X, song_y, cv=5)
stop = timeit.default_timer()
print(f'\ttook {stop - start:.4f}')
print(f'\tCV of the sklearn RandomForest: mean {np.mean(CV_tree):.4f} values {CV_tree}')