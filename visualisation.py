import numpy as np
import cv2
import config as cfg
import utils
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches


class Visualisation:

    def __init__(self):
        if cfg.OPEN3D_ENABLED:
            m = np.random.rand(8, 3) * 1.2 - 0.75

            self.pcd3 = o3d.geometry.PointCloud()
            self.vis3 = o3d.visualization.Visualizer()
            self.vis3.create_window(width=590, height=430)
            self.pcd3.points = o3d.utility.Vector3dVector(m)
            self.vis3.add_geometry(self.pcd3)

            ro3 = self.vis3.get_render_option()
            ro3.point_size = 7.

            self.y_true = None
            self.y_pred = None
            self.y_error = None
            self.data_transformation = utils.DataTransformation()

    def show_4_images(self, rf, vis_data, points, graph_nodes, frame_idx, fps, save_image=False):

        # TODO: CHANGE IM 1 to original color image
        im1 = cv2.cvtColor(rf.color_image_original, cv2.COLOR_GRAY2BGR)

        im2 = cv2.normalize(rf.depth_image_with_background, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_32F).astype(np.uint8)
        im2 = cv2.equalizeHist(im2)
        im2 = cv2.applyColorMap(im2,
                                cv2.COLORMAP_JET)  # colorize depth image

        im3 = cv2.normalize(rf.depth_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_32F).astype(np.uint8)
        im3 = cv2.cvtColor(im3, cv2.COLOR_GRAY2BGR)
        im3 = cv2.bitwise_not(im3)

        im4 = (np.ones_like(im3) * 255).astype(np.uint8)

        self.y_pred = vis_data[0].copy().flatten()
        self.y_true = vis_data[1].copy()
        self.y_error = np.abs(self.y_true - self.y_pred)

        if cfg.PREDICTOR == 'graphclust':
            pp = points.copy()
            pp = self.data_transformation.transformate_data(pp, inverse_transformation=True)

            pp = pp[:, :2]

            pp[:, 0] *= cfg.IMAGE_WIDTH
            pp[:, 1] *= cfg.IMAGE_HEIGHT
            pp[:, 0] += cfg.IMAGE_WIDTH // 2
            pp[:, 1] += cfg.IMAGE_HEIGHT // 2 + 300
            pp[:, 1] = cfg.IMAGE_HEIGHT - pp[:, 1]

            if cfg.CAMERA == 'left':
                pp[:, 0] = cfg.IMAGE_WIDTH - pp[:, 0]

            pp = pp.astype(int)
            for p in pp:
                im4 = cv2.circle(im4, p, radius=2, color=(0, 0, 0), thickness=2)

        gn = graph_nodes.copy()
        if cfg.PREDICTOR == 'graphclust':
            gn = gn.T

        gn = self.data_transformation.transformate_data(gn, inverse_transformation=True)
        gn = gn[:, :2]

        gn[:, 0] *= cfg.IMAGE_WIDTH
        gn[:, 1] *= cfg.IMAGE_HEIGHT
        gn[:, 0] += cfg.IMAGE_WIDTH // 2
        gn[:, 1] += cfg.IMAGE_HEIGHT // 2 + 300
        gn[:, 1] = cfg.IMAGE_HEIGHT - gn[:, 1]

        if cfg.CAMERA == 'left' and cfg.PREDICTOR == 'graphclust':
            gn[:, 0] = cfg.IMAGE_WIDTH - gn[:, 0]

        gn = gn.astype(np.int)

        neuron_1_pos = gn[0, :]
        neuron_2_pos = gn[1, :]
        neuron_3_pos = gn[2, :]

        im4 = cv2.line(im4, neuron_1_pos, neuron_2_pos, cfg.COLOR_RED, thickness=5)
        im4 = cv2.line(im4, neuron_2_pos, neuron_3_pos, cfg.COLOR_RED, thickness=5)

        im4 = cv2.putText(im4, f'frame {frame_idx + 1}/{cfg.FRAMES_NUMBER}', (10, 20), cfg.font, cfg.fontScale,
                          cfg.COLOR_BLACK,
                          thickness=2)
        im4 = cv2.putText(im4, f'FPS {fps:.2f}', (cfg.IMAGE_WIDTH - 120, 20), cfg.font, cfg.fontScale, cfg.COLOR_GREEN,
                          thickness=2)
        self.draw_table(im4)

        vline = np.zeros((im1.shape[0], cfg.BORDER, 3)).astype(np.uint8)
        hline = np.zeros((cfg.BORDER, im1.shape[1] * 2 + 3 * cfg.BORDER, 3)).astype(np.uint8)

        images = np.hstack((vline, im1, vline, im2, vline))
        images2 = np.hstack((vline, im3, vline, im4, vline))
        images = np.vstack((hline, images, hline, images2, hline))

        if save_image:
            cv2.imwrite('4_images.jpg', images)

        cv2.imshow('4 images', images)
        cv2.waitKey(1)

    def draw_table(self, image):
        table_init_x = 50
        table_init_y = 40
        table_width = 540
        x_step = table_width // 4
        txt = [['', '', '', '', ] for i in range(4)]
        for j in range(1, 4):
            txt[0][j] = f'alpha_{j - 1}'

        txt[0][0] = 'angles'
        txt[1][0] = f'{cfg.PREDICTOR}'
        txt[2][0] = 'true'
        txt[3][0] = 'error (abs)'

        for i in range(3):
            txt[1][i + 1] = f'{self.y_pred[i].item():.3f}'
            txt[2][i + 1] = f'{self.y_true[i].item():.3f}'
            txt[3][i + 1] = f'{self.y_error[i].item():.3f}'

        y_step = 40
        table_height = 4 * y_step
        for i in range(5):
            hline_start = (table_init_x, table_init_y + i * y_step)
            h_line_end = (table_init_x + table_width, table_init_y + i * y_step)
            image = cv2.line(image, hline_start, h_line_end, cfg.COLOR_BLACK, thickness=1)

            vline_start = (table_init_x + i * x_step, table_init_y)
            vline_end = (table_init_x + i * x_step, table_init_y + table_height)
            image = cv2.line(image, vline_start, vline_end, cfg.COLOR_BLACK, thickness=1)

        for i in range(4):
            for j in range(4):
                pos_x = table_init_x + i * x_step + 5
                pos_y = table_init_y + j * y_step + y_step // 2 + 5
                if j == 3:
                    current_color = cfg.COLOR_RED
                else:
                    current_color = cfg.COLOR_BLACK
                image = cv2.putText(image, txt[j][i], (pos_x, pos_y), cfg.font, cfg.fontScale, current_color,
                                    thickness=2)

    @staticmethod
    def create_plot_angles(t, y, y_pred, y_pred_kalman):
        if cfg.VIDEO_TYPE == 'test':
            joint_0_high = 100
        else:
            joint_0_high = 190

        joints_limits = ((-10, joint_0_high), (0, 80), (0, 80))

        w, h = plt.figaspect(cfg.ASPECT)
        fig = plt.figure(f'angles {cfg.PREDICTOR} {cfg.CAMERA} {cfg.VIDEO_TYPE}', figsize=(w, h))

        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)
        if cfg.CAMERA == 'left':
            direct_angle = 60
        else:
            direct_angle = 120
        eps_alpha_0 = 15
        alpha_0 = y[:, 0]
        yellow_field = np.abs(alpha_0 - direct_angle) < eps_alpha_0
        yf = np.argwhere(yellow_field == True)
        if yf.size > 0:
            rectangle_points = [yf[0]]
            for i in range(yf.size - 1):
                if yf[i + 1] - yf[i] > 1:
                    rectangle_points.append(yf[i])
                    rectangle_points.append(yf[i + 1])
            rectangle_points.append(yf[-1])

        for i in range(3):
            fig.axes[i].plot(t, y_pred_kalman[:, i], 'r', linewidth=cfg.KALMAN_LINEWIDTH, label='Kalman filter',
                             alpha=0.6)
            fig.axes[i].plot(t, y[:, i], 'c--', linewidth=cfg.REFERENCE_LINEWIDTH, label='reference')
            fig.axes[i].plot(t, y_pred[:, i], 'k', linewidth=cfg.MEASUREMENT_LINEWIDTH, label='measurement')
            fig.axes[i].grid(True)
            fig.axes[i].set_xlim(0, cfg.T_END)
            fig.axes[i].set_ylim(*joints_limits[i])

            if yf.size > 0:
                for j in range(0, len(rectangle_points), 2):
                    rect = patches.Rectangle((t[rectangle_points[j]], -50),
                                             t[rectangle_points[j + 1]] - t[rectangle_points[j]], 250, linewidth=1,
                                             edgecolor=cfg.RECTANGLE_COLOR, facecolor=cfg.RECTANGLE_COLOR, alpha=0.5)
                    fig.axes[i].add_patch(rect)
            if i == 0 and cfg.VIDEO_TYPE == 'train':
                fig.axes[i].set_yticks([0, 50, 100, 150, 200])
            else:
                fig.axes[i].set_yticks([0, 20, 40, 60, 80])

            fig.axes[i].set_ylabel(f'$\\alpha_{i}$ [deg]')
        ax1.legend(loc='upper left', prop={'size': cfg.LEGEND_SIZE})
        ax3.set_xlabel('t [s]')
        fig.tight_layout()

        if yf.size > 0:
            return rectangle_points

    @staticmethod
    def create_plot_points(t, y, y_pred, y_kalman, yellow_field):
        mm_convert_factor = 1000
        w, h = plt.figaspect(cfg.ASPECT)
        fig = plt.figure(f'points {cfg.PREDICTOR} {cfg.CAMERA} {cfg.VIDEO_TYPE}', figsize=(w, h))
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)
        keys = list(y_pred.keys())
        if cfg.VIDEO_TYPE == 'test':
            mm_limits = ((0, 350), (-400, 0), (-100, 100))
        else:
            mm_limits = ((-50, 400), (-400, 400), (-150, 150))
        for i in range(3):
            fig.axes[i].plot(t, np.array(y_kalman[keys[i]]) * mm_convert_factor, 'r',
                             linewidth=cfg.KALMAN_LINEWIDTH,
                             label='Kalman filter')
            fig.axes[i].plot(t, np.array(y[keys[i]]) * mm_convert_factor, 'c--', linewidth=cfg.REFERENCE_LINEWIDTH,
                             label='reference')
            fig.axes[i].plot(t, np.array(y_pred[keys[i]]) * mm_convert_factor, 'k', linewidth=cfg.MEASUREMENT_LINEWIDTH,
                             label='measurement')
            fig.axes[i].grid(True)

            fig.axes[i].set_ylabel(f'{keys[i]} [mm]')
            fig.axes[i].set_xlim(0, cfg.T_END)
            fig.axes[i].set_ylim(mm_limits[i][0], mm_limits[i][1])

        if yellow_field is not None:
            for j in range(0, len(yellow_field), 2):
                rect = patches.Rectangle((t[yellow_field[j]], -500),
                                         t[yellow_field[j + 1]] - t[yellow_field[j]], 1000, linewidth=1,
                                         edgecolor=cfg.RECTANGLE_COLOR, facecolor=cfg.RECTANGLE_COLOR, alpha=0.5)
                fig.axes[i].add_patch(rect)

        ax3.set_xlabel('t [s]')
        ax1.legend(loc='upper left', prop={'size': cfg.LEGEND_SIZE})
        fig.tight_layout()

    @staticmethod
    def create_plot_3d(original, pred_left_kalman):
        fig = plt.figure(f'3d points {cfg.PREDICTOR} {cfg.CAMERA} {cfg.VIDEO_TYPE}')
        ax = fig.add_subplot(111, projection='3d')
        ax = Axes3D(fig)
        ax.set_xlim(-0.1, 0.3)
        ax.set_ylim(-0.4, 0.4)
        ax.set_zlim(-0.2, 0.2)
        ax.plot(original['x'], original['y'], original['z'], 'k', label='reference')
        ax.plot(pred_left_kalman['x'], pred_left_kalman['y'], pred_left_kalman['z'], 'y', linewidth=1,
                label='measurement')
        ax.view_init(elev=35, azim=-28)
        ax.legend(loc='upper left', bbox_to_anchor=[0.1, 0.9])
        ax.set_xlabel('$\\mathbf{x~[m]}$')
        ax.set_ylabel('$\\mathbf{y~[m]}$')
        ax.set_zlabel('$\\mathbf{z~[m]}$')

    def vis_open3d(self, points, graph_nodes):
        N_POINTS_FILLING = 20

        self.neuron_1_pos = graph_nodes[:, 0]
        self.neuron_2_pos = graph_nodes[:, 1]
        self.neuron_3_pos = graph_nodes[:, 2]

        neurons1 = utils.get_equidistant_points(self.neuron_1_pos, self.neuron_2_pos, N_POINTS_FILLING)
        neurons1 = np.asarray([*neurons1])
        neurons2 = utils.get_equidistant_points(self.neuron_2_pos, self.neuron_3_pos, N_POINTS_FILLING)
        neurons2 = np.asarray([*neurons2])
        neurons = np.vstack((neurons1, neurons2))

        self.pcd3.points = o3d.utility.Vector3dVector(neurons)
        self.pcd3.points.extend(points.astype('float64').tolist())

        color_red = np.array([1., 0., 0.])
        repeats = neurons.shape[0]

        np_colors = np.tile(color_red, repeats).reshape(repeats, 3)
        np_colors_2 = np.zeros((points.shape[0], 3))

        self.pcd3.colors = o3d.utility.Vector3dVector(np.vstack((np_colors, np_colors_2)))
        self.vis3.update_geometry(self.pcd3)
        self.vis3.poll_events()
        self.vis3.update_renderer()

    @staticmethod
    def plot_background_extractor(kmeans_std_result_negative, kmeans_std_result_positive, subtracted):
        plt.figure()
        plt.imshow(kmeans_std_result_negative, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('Negative Centers')
        plt.tight_layout()
        plt.figure()
        plt.imshow(kmeans_std_result_positive, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('Positive Centers')
        plt.tight_layout()
        plt.figure()
        plt.imshow(kmeans_std_result_positive - kmeans_std_result_negative, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('Difference')
        plt.tight_layout()
        plt.show()
