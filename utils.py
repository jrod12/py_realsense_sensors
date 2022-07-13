import config as cfg
import numpy as np
from math import cos, sin
from pykalman import KalmanFilter
import sparse


class DataTransformation:
    def __init__(self):
        self.transformation_matrix, self.transformation_matrix_p0 = self.load_transformation_matrix()

    def load_transformation_matrix(self):
        if cfg.CAMERA == 'left':
            transformation_matrix = np.loadtxt(cfg.TRANSFORMATION_MATRIX_LEFT_FILENAME)
            transformation_matrix_p0 = np.loadtxt(cfg.TRANSFORMATION_MATRIX_POINT_LEFT_FILENAME)
        else:
            transformation_matrix = np.loadtxt(cfg.TRANSFORMATION_MATRIX_FILENAME)
            transformation_matrix_p0 = np.loadtxt(cfg.TRANSFORMATION_MATRIX_POINT_FILENAME)
        transformation_matrix = transformation_matrix[:, [2, 0, 1]]

        return transformation_matrix, transformation_matrix_p0

    def transformate_data(self, input_data, inverse_transformation=False):
        if inverse_transformation:
            tm = self.transformation_matrix.T
            p0 = -self.transformation_matrix_p0
        else:
            tm = self.transformation_matrix#.copy()
            p0 = self.transformation_matrix_p0#.copy()
        input_data -= np.repeat(p0.reshape(1, 3), repeats=input_data.shape[0], axis=0)
        result = np.dot(input_data, tm)

        if cfg.CAMERA == 'left' and not inverse_transformation:
            result = np.dot(result, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))

        return result

    def sparse_points(self, x):
        POINTS_PER_METER_SPARSE = 250
        x_input = x.copy()
        x_input = self.transformate_data(x_input)
        translation_vector = np.array([1.5, 1.5, 4])
        x_input += translation_vector
        x_input *= POINTS_PER_METER_SPARSE
        x_input = x_input.astype(np.int)
        result = sparse.COO(x_input.T, np.ones((len(x_input))), shape=(750, 750, 1375))
        return result


def get_equidistant_points(p1, p2, parts):
    return zip(np.linspace(p1[0], p2[0], parts + 1),
               np.linspace(p1[1], p2[1], parts + 1),
               np.linspace(p1[2], p2[2], parts + 1))


def robot_kinematics(joint0, joint1, joint2):
    r1 = np.array([0, 0, 0.19577228366977298])
    r2 = np.array([0.21630179653301265, 0, 0])
    data2 = np.zeros((3, 3))
    joint2_corrected = joint2
    data2[2, :] = rotate_y_axis(r2, joint2_corrected)
    data2[1, :] = rotate_y_axis(r1, joint1 - 90)
    data2[2, :] = data2[1, :] + data2[2, :]
    data2 = rotate_z_axis(data2, (90 - joint0))
    return data2


def robot_kinematics_wrapper(y):
    result = {'x': [], 'y': [], 'z': []}
    for angles in y:
        end_effector_pos = robot_kinematics(angles[0], angles[1], angles[2])[2, :]
        result['x'].append(end_effector_pos[0])
        result['y'].append(end_effector_pos[1])
        result['z'].append(end_effector_pos[2])
    return result


def rotate_y_axis(input_data, angle):
    angle = np.radians(angle)
    rotation_matrix_y = np.array([[cos(angle), 0, -sin(angle)], [0, 1, 0], [-sin(angle), 0, cos(angle)]])
    input_data = input_data.dot(rotation_matrix_y)
    return input_data  # TODO: RETURN BEZ SENSU SKORO ZMIENIAM WEJSCIE


def rotate_z_axis(input_data, angle):
    angle = np.radians(angle)
    rotation_matrix_z = np.array([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]])
    input_data = input_data.dot(rotation_matrix_z)
    return input_data


def rotate_x_axis(input_data, angle):
    angle = np.radians(angle)
    rotation_matrix_z = np.array([[1, 0, 0], [0, cos(angle), -sin(angle)], [0, sin(angle), cos(angle)]])
    input_data = input_data.dot(rotation_matrix_z)
    return input_data


def kalman_filter(signal):
    print('Kalman filter is working...')
    signal_dim = signal.shape[1]
    signal_filtered = np.zeros_like(signal)

    filters = []
    for i in range(signal_dim):
        filters.append(KalmanFilter().em(signal[:, i]))
        filters[i].transition_covariance *= 3e-1

        signal_filtered[:, i] = np.squeeze(filters[i].filter(signal[:, i])[0])

    return signal_filtered
