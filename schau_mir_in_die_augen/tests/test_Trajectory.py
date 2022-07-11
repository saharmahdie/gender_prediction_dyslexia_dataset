from unittest import TestCase
import numpy as np

from schau_mir_in_die_augen.process.trajectory import Trajectory
from schau_mir_in_die_augen.process.trajectory import calculate_derivation
from schau_mir_in_die_augen.trajectory_classification.trajectory_split import ivt, eye_movement_classification, \
    EyeMovementClassifier
import schau_mir_in_die_augen.process.conversion as converters
import schau_mir_in_die_augen.process.filtering as filters


def random_screen_params():
    return {'pix_per_mm': np.random.rand(2)*100+1,
            'screen_dist_mm': np.random.rand()*1000+100,
            'screen_res': np.random.rand(2)*1000+100}


class TestIvt(TestCase):

    def setUp(self):
        np.random.seed(0)
        self.xy = np.random.rand(100, 2)*500
        self.screen_params = random_screen_params()
        self.screen_paramsNew = {'pix_per_mm': self.screen_params['pix_per_mm'],
                                 'screen_dist_mm': self.screen_params['screen_dist_mm'],
                                 'fov_x': self.screen_params['screen_res'][0],
                                 'fov_y': self.screen_params['screen_res'][1]}
        self.sample_rate = 1 + np.random.randint(1000)

        self.vel_threshold = np.random.rand() * 100
        self.min_fix_duration = np.random.rand()

    def test_conversion(self):

        trajectory_a = Trajectory([[1, 2]], 'pixel',
                                  pix_per_mm=[34, 56], screen_dist_mm=78)
        trajectory_b = Trajectory(np.asarray([[1, 2], [3, 4]]), 'pixel',
                                  pix_per_mm=(56, 78), screen_dist_mm=90)
        trajectory_c = Trajectory(np.asarray([[1, 2, 3], [4, 5, 6]]).T, 'pixel',
                                  pix_per_mm=np.asarray([78, 90]), screen_dist_mm=1011)

        result = trajectory_a.pixel2rad([100, 100])
        result = trajectory_b.pixel2rad([100, 100])
        result = trajectory_c.pixel2rad([100, 100])

        result = trajectory_a.rad2pixel([1, 1])
        result = trajectory_b.rad2pixel([1, 1])
        result = trajectory_c.rad2pixel([1, 1])

        # todo: test values

    def test_definition(self):
        trajectory = Trajectory(self.xy, 'pixel', sample_rate=self.sample_rate, **self.screen_paramsNew)

        self.assertEqual(self.sample_rate, trajectory.sample_rate)
        self.assertEqual(self.screen_paramsNew['fov_x'], trajectory.get_fov('pixel')[0])
        self.assertEqual(self.screen_paramsNew['fov_y'], trajectory.get_fov('pixel')[1])
        self.assertEqual(self.screen_paramsNew['pix_per_mm'][0], trajectory.pix_per_mm[0])
        self.assertEqual(self.screen_paramsNew['pix_per_mm'][1], trajectory.pix_per_mm[1])
        self.assertEqual(self.screen_paramsNew['screen_dist_mm'], trajectory.screen_dist_mm)

    def test_compare(self):
        # two ways
        # see feature_extraction

        trajectory = Trajectory(self.xy, 'pixel_shifted', sample_rate=self.sample_rate, **self.screen_paramsNew)
        _ = trajectory.velocity  # this triggers calculation of velocity, so it is saved and should be deleted.
        # this was a bug before

        # convert to angle
        angle = converters.convert_shifted_pixel_coordinates_to_angles_deg(self.xy, **self.screen_params)
        trajectory.convert_to('angle_deg')
        self.assertTrue(np.allclose(angle, trajectory.xy))

        # savgol filter
        smoothed_angle = filters.savgol_filter_trajectory(angle)
        trajectory.apply_filter('savgol')
        self.assertTrue(np.allclose(smoothed_angle, trajectory.xy))

        # velocity
        smoothed_vel_xy = calculate_derivation(smoothed_angle, sample_rate=self.sample_rate)
        smoothed_vel = np.linalg.norm(smoothed_vel_xy, axis=1)
        self.assertTrue(np.allclose(smoothed_vel, trajectory.velocity))

        sac1, fix1 = ivt(smoothed_vel, self.vel_threshold, self.min_fix_duration, self.sample_rate)
        sample_start, sample_type = eye_movement_classification(
            trajectory, EyeMovementClassifier(
                name='IVT',
                parameter={'vel_threshold': self.vel_threshold,
                                             'min_fix_duration': self.min_fix_duration,
                                             'sample_rate': self.sample_rate}))

        sac2 = [(sample_start[idd], sample_start[idd+1])
                for (idd, sample) in enumerate(sample_type) if sample == 'saccade']
        fix2 = [(sample_start[idd], sample_start[idd+1])
                for (idd, sample) in enumerate(sample_type) if sample == 'fixation']

        # fixing last element of ivt
        # ... -.-
        if len(sac1) and len(fix1):
            if sac1[-1][1] > fix1[-1][1]:
                sac1[-1] = (sac1[-1][0], sac1[-1][1] + 1)
            else:
                fix1[-1] = (fix1[-1][0], fix1[-1][1] + 1)
        elif len(sac1):
            sac1[-1] = (sac1[-1][0], sac1[-1][1] + 1)
        elif len(fix1):
            fix1[-1] = (fix1[-1][0], fix1[-1][1] + 1)

        self.assertEqual(sac1, sac2)
        self.assertEqual(fix1, fix2)

        # convert back to pixel
        smoothed_pixels = converters.convert_angles_deg_to_shifted_pixel_coordinates(smoothed_angle,
                                                                                     **self.screen_params)
        self.assertTrue(np.allclose(smoothed_pixels, trajectory.get_trajectory('pixel_shifted')))
