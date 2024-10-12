# tidal/data_utils.py

import numpy as np

def normalize_data(data):
    """
    Normalize the input data to the range [0, 2π] for mapping to the IOT.
    """
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return 2 * np.pi * (data - min_vals) / (max_vals - min_vals)

def map_to_iot(data, iot):
    """
    Map normalized data to the IOT surface.
    """
    u = data[:, 0]
    v = data[:, 1]
    x = (iot.R + iot.r * np.cos(v)) * np.cos(u)
    y = (iot.R + iot.r * np.cos(v)) * np.sin(u)
    z = iot.r * np.sin(v)
    return np.column_stack((x, y, z))

def iot_to_data(iot_coords, iot):
    """
    Map IOT coordinates back to the original data space.
    """
    x, y, z = iot_coords[:, 0], iot_coords[:, 1], iot_coords[:, 2]
    u = np.arctan2(y, x)
    v = np.arctan2(z, np.sqrt((x**2 + y**2) - iot.R**2))
    return np.column_stack((u, v))

class IOTDataMapper:
    def __init__(self, iot):
        self.iot = iot

    def fit(self, data):
        self.data_min = np.min(data, axis=0)
        self.data_max = np.max(data, axis=0)

    def transform(self, data):
        normalized_data = normalize_data(data)
        return map_to_iot(normalized_data, self.iot)

    def inverse_transform(self, iot_coords):
        data_coords = iot_to_data(iot_coords, self.iot)
        return data_coords * (self.data_max - self.data_min) / (2 * np.pi) + self.data_min