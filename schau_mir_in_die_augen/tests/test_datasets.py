import unittest
import numpy as np

import schau_mir_in_die_augen.process.filtering as filters
import schau_mir_in_die_augen.process.conversion as converters
from schau_mir_in_die_augen.process.trajectory import calculate_derivation
from schau_mir_in_die_augen.datasets.DemoDataset import DemoDataset


class TestDatasets(unittest.TestCase):

    def setUp(self):
        self.xy = np.random.randint(-100, 1500, (30, 2))

    def test_pixel_angle_convert(self):
        x = [0, 0, 1, 2, 0, 3]
        y = [0, 0, 0, 0, 0, 3]
        xy = np.asarray([x, y]).T
        screen_size_mm = np.asarray([474, 297])
        screen_res = np.asarray([1680, 1050])
        pix_per_mm = screen_res / screen_size_mm
        screen_dist_mm = 550
        params = [pix_per_mm, screen_dist_mm, screen_res]
        a = converters.convert_angles_deg_to_shifted_pixel_coordinates(
            converters.convert_shifted_pixel_coordinates_to_angles_deg(xy, *params), *params)
        self.assertTrue(np.allclose(xy, a))
        b = converters.convert_shifted_pixel_coordinates_to_angles_deg(
            converters.convert_angles_deg_to_shifted_pixel_coordinates(xy, *params), *params)
        self.assertTrue(np.allclose(b, xy))

    def test_savgol_filter(self):
        ds = DemoDataset()
        xy = ds.load_data(user='User 1', case='Case 1')
        angle = converters.convert_shifted_pixel_coordinates_to_angles_deg(xy, *ds.get_screen_params().values())
        smoothed_angle = filters.savgol_filter_trajectory(angle)

        diff_x = smoothed_angle[:, 0][:-1] - smoothed_angle[:, 0][1:]
        vel_x = diff_x * ds.sample_rate
        diff_y = smoothed_angle[:, 1][:-1] - smoothed_angle[:, 1][1:]
        vel_y = diff_y * ds.sample_rate

        smoothed_vel_xy = np.asarray([vel_x, vel_y]).T
        smoothed_vel_xy_feat = calculate_derivation(smoothed_angle, sample_rate=ds.sample_rate)
