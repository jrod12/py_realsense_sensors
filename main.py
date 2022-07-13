import config as cfg
from tracking import Tracker
import sys

if sys.platform != 'win32':
    from models.nn import ConvNet
else:
    if cfg.PREDICTOR == 'nn':
        raise ValueError('cannot run sparse network on Windows')

if cfg.VIDEO_TYPE == 'test':
    if cfg.CAMERA == 'left':
        BAG_FILENAME = "data/20210103111902_TESTING_LEFT.bag"
    elif cfg.CAMERA == 'right':
        BAG_FILENAME = "data/20210103111902_TESTING_RIGHT.bag"
    else:
        raise ValueError('Incorrect camera type [config.ini]')
    ANGLES_FROM_ROBOT_FILENAME = 'data/20210103111902_TESTING.csv'
elif cfg.VIDEO_TYPE == 'train':
    if cfg.CAMERA == 'left':
        BAG_FILENAME = "data/20210103110338_LEFT.bag"
    elif cfg.CAMERA == 'right':
        BAG_FILENAME = "data/20210103110338_RIGHT.bag"
    else:
        raise ValueError('Incorrect camera type [config.ini]')
    ANGLES_FROM_ROBOT_FILENAME = 'data/20210103110338.csv'
else:
    raise ValueError('Incorrect video type [config.ini]')


tracker = Tracker(device_from_file=BAG_FILENAME)
tracker.do_tracking(drop_first_frames=cfg.FPS * 4)
