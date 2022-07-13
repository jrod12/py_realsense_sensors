import configparser
import logging
import cv2

config = configparser.ConfigParser()
config.read("config.ini")

CAMERA = config.get('main', 'Camera')
PREDICTOR = config.get('main', 'Predictor')
VIDEO_TYPE = config.get('main', 'VideoType')

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
color = (255, 0, 0)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 200, 0)
thickness = 1
BORDER = 5

OPEN3D_ENABLED = config.getboolean('visualization', 'Open3dEnabled')
PLOT_BACKGROUND_EXTRACTOR = config.getboolean('visualization', 'PlotBackgroundExtractor')
SHOW_4_IMAGES = config.getboolean('visualization', 'Show4Images')

SCALE_DIVISOR = 5
if VIDEO_TYPE == 'train':
    FRAMES_NUMBER = 4240
else:
    FRAMES_NUMBER = 980
FPS = 6
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# visualisation
ASPECT = 0.55
FONT_SIZE = 12.
LEGEND_SIZE = 7
if VIDEO_TYPE == 'test':
    T_END = 160
else:
    T_END = 700
KALMAN_LINEWIDTH = 1.8
MEASUREMENT_LINEWIDTH = 0.5
REFERENCE_LINEWIDTH = 1
RECTANGLE_COLOR = 'khaki'

INTERMEDIATES = config.getint('graph', 'Intermediates')
EPOCHS = config.get('graph', 'Epochs')
KMEANS_CENTERS = config.getint('graph', 'Kmeans')

TORCH_NUM_THREADS = config.getint('threads', 'TorchNumThreads')
SKLEARN_NUM_THREADS = config.get('threads', 'SklearnNumThreads')

TRANSFORMATION_MATRIX_FILENAME = 'data/transformation_matrix.txt'
TRANSFORMATION_MATRIX_POINT_FILENAME = 'data/transformation_matrix_p0.txt'
TRANSFORMATION_MATRIX_LEFT_FILENAME = 'data/transformation_matrix_LEFT.txt'
TRANSFORMATION_MATRIX_POINT_LEFT_FILENAME = 'data/transformation_matrix_p0_LEFT.txt'
