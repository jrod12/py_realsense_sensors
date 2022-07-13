import config as cfg
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import utils
from models.graph import GraphRegressor
import sys

if sys.platform != 'win32':
    from models.nn import ConvNetRegressor
from models.uarm_angles import UarmAngles
import cv2
import skimage
import copy
import visualisation
from ttictoc import tic, toc

TIMESTAMP_DIVISOR = 1000


class Tracker:
    def __init__(self, device_from_file):
        self.filename = device_from_file
        self.kmeans_robot_structure_centers = cfg.KMEANS_CENTERS
        self.pipeline, self.align, self.pc = self.initialize_camera()
        self.kmeans_result_positive, self.kmeans_result_negative = self.load_background_extractor()
        self.visualisation = visualisation.Visualisation()
        self.uarm_angles = UarmAngles()
        if cfg.PREDICTOR == 'graphclust':
            self.predictor = GraphRegressor()
        elif cfg.PREDICTOR == 'nn':
            self.predictor = ConvNetRegressor()
        else:
            raise ValueError('Incorrect regressor name [config.ini]')

    def initialize_camera(self):
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, self.filename, repeat_playback=False)

        config.enable_stream(rs.stream.depth)
        config.enable_stream(rs.stream.color)
        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(real_time=False)  # no realtime play - no dropped frames
        align = rs.align(rs.stream.color)
        pc = rs.pointcloud()
        return pipeline, align, pc

    def load_background_extractor(self):

        if cfg.CAMERA == 'right':
            kmeans_std_result_positive = np.loadtxt(
                'data/20210103110338_RIGHT.bag_kmeans_resized_scale_5_color_image_positive.txt')
            kmeans_std_result_negative = np.loadtxt(
                'data/20210103110338_RIGHT.bag_kmeans_resized_scale_5_color_image_negative.txt')
        else:
            kmeans_std_result_positive = np.loadtxt(
                'data/20210103110338_LEFT.bag_kmeans_resized_scale_5_color_image_positive.txt')
            kmeans_std_result_negative = np.loadtxt(
                'data/20210103110338_LEFT.bag_kmeans_resized_scale_5_color_image_negative.txt')

        subtracted = kmeans_std_result_positive - kmeans_std_result_negative

        subtracted_digitized = subtracted < -60  # boolean image

        if cfg.PLOT_BACKGROUND_EXTRACTOR:
            visualisation.Visualisation.plot_background_extractor(kmeans_std_result_negative,
                                                                  kmeans_std_result_positive, subtracted)

        kmeans_std_result_positive = kmeans_std_result_positive * subtracted_digitized  # mask
        kmeans_std_result_positive[kmeans_std_result_positive == 0] = 255

        return kmeans_std_result_positive, kmeans_std_result_negative

    def do_tracking(self, drop_first_frames):

        first_frame_collected = False

        try:
            if drop_first_frames:
                i = 0
                while i < drop_first_frames:
                    trash = self.pipeline.wait_for_frames()
                    del trash
                    i += 1

            frames_counter = 0
            t0_depth_prev = 0

            y_true_list = []
            y_huber_list = []
            t_depth_list = []

            while frames_counter < cfg.FRAMES_NUMBER:  # MAIN LOOP
                processing_time = 0
                tic()
                rf = VideoFrame(cfg.SCALE_DIVISOR, self.align, self.pc, self.pipeline, self.kmeans_result_positive,
                                self.kmeans_result_negative)

                rf.get_frame_from_camera()

                t0_depth = rf.timestamp_depth / TIMESTAMP_DIVISOR
                if t0_depth_prev - t0_depth == 0:  # doubled frame in bag file
                    t0_depth_prev = t0_depth
                    frames_counter += 1
                    continue
                t0_depth_prev = t0_depth

                rf.remove_background()
                rf.generate_pointcloud()


                if not first_frame_collected:
                    initial_timestamp = rf.timestamp_depth / TIMESTAMP_DIVISOR
                    first_frame_collected = True
                processing_time += toc()
                if frames_counter % 10 == 0:
                    print(frames_counter)
                y_true = self.uarm_angles.predict(rf.timestamp_depth / TIMESTAMP_DIVISOR)

                tic()
                result = [self.predictor.predict(rf.points)] + [y_true]
                processing_time += toc()

                y_huber_list.append(result[0])
                y_true_list.append(result[1])
                t_depth_list.append(rf.timestamp_depth / TIMESTAMP_DIVISOR - initial_timestamp)

                if cfg.SHOW_4_IMAGES:
                    fps = 1 / processing_time
                    if cfg.PREDICTOR == 'graphclust':
                        self.visualisation.show_4_images(rf, result, self.predictor.points,
                                                         self.predictor.init_weights.T, frames_counter, fps)
                        # if frames_counter == 35:
                        #     self.visualisation.show_4_images(rf, result, self.predictor.points,
                        #                                      self.predictor.init_weights.T, frames_counter, fps, True)
                    else:
                        nodes = utils.robot_kinematics(*result[0][0])
                        self.visualisation.show_4_images(rf, result, None, nodes, frames_counter, fps)

                if cfg.OPEN3D_ENABLED:
                    if cfg.PREDICTOR == 'graphclust':
                        self.visualisation.vis_open3d(self.predictor.points, self.predictor.init_weights.T)

                frames_counter += 1

            y = np.vstack(y_true_list)
            signal = np.vstack(y_huber_list)
            signal_kalman = utils.kalman_filter(signal)
            t = np.array(t_depth_list)

            y_points = utils.robot_kinematics_wrapper(y)
            y_points_array = np.array([y_points['x'], y_points['y'], y_points['z']]).T
            signal_points = utils.robot_kinematics_wrapper(signal)
            signal_points_array = np.array([signal_points['x'], signal_points['y'], signal_points['z']]).T
            signal_kalman_points = utils.robot_kinematics_wrapper(signal_kalman)
            signal_kalman_points_array = np.array(
                [signal_kalman_points['x'], signal_kalman_points['y'], signal_kalman_points['z']]).T

            print(f'mean error angles {cfg.PREDICTOR} {cfg.CAMERA} {cfg.VIDEO_TYPE}',
                  np.mean(np.abs(y - signal), axis=0))
            print(f'mean error angles {cfg.PREDICTOR} {cfg.CAMERA} {cfg.VIDEO_TYPE} kalman',
                  np.mean(np.abs(y - signal_kalman), axis=0))

            print(
                f'mean error points {cfg.PREDICTOR} {cfg.CAMERA} {cfg.VIDEO_TYPE} {np.mean(np.abs(y_points_array - signal_points_array) * 1000, axis=0)} [mm]', )
            print(
                f'mean error points {cfg.PREDICTOR} {cfg.CAMERA} {cfg.VIDEO_TYPE} kalman {np.mean(np.abs(y_points_array - signal_kalman_points_array) * 1000, axis=0)} [mm]')

            yellow_field = visualisation.Visualisation.create_plot_angles(t, y, signal, signal_kalman)
            visualisation.Visualisation.create_plot_points(t, y_points, signal_points, signal_kalman_points,
                                                           yellow_field)
            visualisation.Visualisation.create_plot_3d(y_points, signal_kalman_points)
            plt.show()

        finally:
            pass


class VideoFrame:
    def __init__(self, scale_divider=1, align=None, pc=None, pipeline=None, kmeans_std_result_positive=None,
                 kmeans_std_result_negative=None):
        self.scale_divider = scale_divider
        self.align = align
        self.pc = pc
        self.pipeline = pipeline
        self.kmeans_std_result_positive = kmeans_std_result_positive
        self.kmeans_std_result_negative = kmeans_std_result_negative

    def get_frame_from_camera(self, resize=True):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        self.depth_frame = aligned_frames.get_depth_frame()
        self.color_frame = aligned_frames.get_color_frame()

        self.depth_image = np.asarray(self.depth_frame.get_data())
        self.color_image = np.asarray(self.color_frame.get_data())
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
        self.color_image_original = self.color_image.copy()
        self.depth_image_with_background = self.depth_image.copy()

        if self.scale_divider > 1 and resize:
            self.color_image = skimage.measure.block_reduce(self.color_image,
                                                            block_size=(self.scale_divider, self.scale_divider),
                                                            func=np.max)

        self.timestamp_depth = self.depth_frame.timestamp
        self.timestamp_color = self.color_frame.timestamp

    def remove_background(self):
        negative_indexes = np.where((abs(self.color_image - self.kmeans_std_result_positive) < abs(
            self.color_image - self.kmeans_std_result_negative)) == False)
        move = np.where(
            np.abs((self.kmeans_std_result_negative - self.color_image) > 150))
        self.color_image[negative_indexes[0], negative_indexes[1]] = 255

        tmp = self.color_image.copy()
        tmp[move] = 255

        self.color_image = tmp

        if self.scale_divider > 1:
            dim = (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT)
            self.color_image = cv2.resize(self.color_image, dim,
                                          interpolation=cv2.INTER_AREA)

        self.depth_image_with_background = copy.deepcopy(self.depth_image)
        self.depth_image[self.color_image == 255] = 0

    def generate_pointcloud(self):
        points_from_cloud = self.pc.calculate(self.depth_frame)
        v = points_from_cloud.get_vertices()
        self.points = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        self.points = self.points[np.all(self.points != 0, axis=1)]  # remove 0 rows
        self.points = np.dot(self.points, np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))  # rotate 180 degrees
