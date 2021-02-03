import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats.mstats import winsorize


def main():
    data = [[0], [0.2], [0.1], [0.01], [0.5], [100], [1000], [0.03]]
    print(data)
    print('sqrt', np.sqrt(data))
    scaler = StandardScaler()
    scaler.fit(data)
    print('standard', scaler.transform(data))
    scaler = MinMaxScaler()
    scaler.fit(data)
    print('minmax', scaler.transform(data))
    print('log1', np.log1p(data))
    print('winsorize', winsorize(data, limits=[0.3, 0.3]))


if __name__ == '__main__':
    main()
