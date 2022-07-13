import config as cfg
import os
os.environ["MKL_NUM_THREADS"] = cfg.SKLEARN_NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = cfg.SKLEARN_NUM_THREADS
os.environ["OMP_NUM_THREADS"] = cfg.SKLEARN_NUM_THREADS
import joblib
from models.regressor import Regressor
import scipy.cluster.vq as scipy_cluster_vq
import numpy as np
from scipy.optimize import minimize
import utils
from scipy.spatial.distance import cdist


class GraphRegressor(Regressor):
    def __init__(self):
        self.init_weights = None,
        self.init_angles = None
        self.huber = self.load_huber_regressor()
        self.data_transformation = utils.DataTransformation()

    def load_huber_regressor(self):
        if cfg.CAMERA == 'left':
            path = 'models/HuberRegressorLeft2.joblib'
        elif cfg.CAMERA == 'right':
            path = 'models/HuberRegressorRight.joblib'
        else:
            raise ValueError('Incorrect camera name [config.ini]')
        return joblib.load(path)

    def do_kmeans(self, x):
        """ find robot structure using kmeans """
        kmeans_result = scipy_cluster_vq.kmeans(x, k_or_guess=cfg.KMEANS_CENTERS, iter=5, thresh=1e-2)
        self.points = self.data_transformation.transformate_data(kmeans_result[0])

    def fit_graph(self):
        graph = np.diag(np.ones(2), 1)
        if self.init_weights is not None:
            self.graphclust = GraphClust(graph, nodes=self.init_weights, num_intermediate=cfg.INTERMEDIATES,
                                         init_angles=self.init_angles)
        else:
            self.graphclust = GraphClust(graph, nodes=None, num_intermediate=cfg.INTERMEDIATES)

        self.init_weights, s, self.init_angles = self.graphclust.fit(self.points)

    def do_regression(self):
        return self.huber.predict(self.init_weights.T.reshape(1, -1))

    def predict(self, x):
        self.do_kmeans(x)
        self.fit_graph()
        angles = self.do_regression().flatten()

        return angles


class GraphClust:
    def __init__(self, graph, nodes, num_intermediate=20, init_angles=None):
        self.graph = graph
        self.nodes = nodes
        self.k = graph.shape[0]
        self.num_intermediate = num_intermediate
        self.init_angles = init_angles

    @staticmethod
    def expand_c(C, graph, num_intermediate=10):
        k = C.shape[0]
        indi, indj = graph.nonzero()
        ec = np.zeros((len(indi) * (num_intermediate - 1) + 1, C.shape[1]))
        ec[0:k, :] = C
        for i, j in zip(indi, indj):
            l = np.linspace(0, 1, num_intermediate)
            l = l[1:-1]
            for t in l:
                ec[k, :] = C[i, :] * (1 - t) + C[j, :] * t
                k += 1
        return ec

    def fit(self, x):

        x0 = None
        bd = [[-150, 150], [-100, 100], [-100, 100]]
        if self.init_angles is not None:
            a0 = self.init_angles
        else:
            a0 = [0, 45, 45]

        res = minimize(self.fc_alpha, a0, args=(x0, x, self.graph), bounds=bd, method='L-BFGS-B',
                       options={'disp': None, 'iprint': -1, 'maxiter': 10})
        angles = res['x']
        c = utils.robot_kinematics(angles[0], angles[1], angles[2])

        d = self.dist(x, c)
        s = np.argmin(d, axis=1)
        self.ec = GraphClust.expand_c(c, self.graph, self.num_intermediate)
        return c, s, angles

    def dist(self, X, C):
        return np.add.outer(np.sum(X * X, axis=1), np.sum(C * C, axis=1)) - 2 * X.dot(C.T)

    def fc_alpha(self, angles, x0, x, graph):
        C_kinematic = utils.robot_kinematics(angles[0], angles[1], angles[2])
        x = self.fc(C_kinematic, x, graph)

        return x

    def fc(self, C, x, graph):
        ec = self.expand_c(C, graph, num_intermediate=5)
        d = cdist(x, ec)
        d2 = np.min(d, axis=1)
        d3 = np.min(d, axis=0)

        f = np.sum(d2) + np.sum(d3)

        return f
