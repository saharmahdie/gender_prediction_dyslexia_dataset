from unittest import TestCase
import numpy as np

from schau_mir_in_die_augen.process.trajectory import Trajectory, Trajectories

def random_screen_params_new():
    return {'pix_per_mm': np.random.rand(2)*100+1,
            'screen_dist_mm': np.random.rand()*1000+100,
            'fov_x': np.random.rand(1)*1000+100,
            'fov_y': np.random.rand(1)*1000+100}

class TestIvt(TestCase):

    def setUp(self):

        np.random.seed(0)

        size = 100

        self.trajectory_a = Trajectory(xy=np.random.rand(size, 2), kind='pixel', user='A',
                                       **random_screen_params_new())
        self.trajectory_b = Trajectory(xy=np.random.rand(size, 2), kind='angle_deg', user='B',
                                       **random_screen_params_new())
        self.trajectory_c = Trajectory(xy=np.random.rand(size, 2), kind='angle_rad', user='C',
                                       **random_screen_params_new())
        self.trajectory_d = Trajectory(xy=np.random.rand(size, 2), kind='pixel_shifted', user='D',
                                       **random_screen_params_new())
        self.trajectory_e = Trajectory(xy=np.random.rand(size, 2), kind='pixel_image', user='E',
                                       **random_screen_params_new())
        self.trajectory_f = Trajectory(xy=np.random.rand(size, 2), kind='pixel', user='F',
                                       **random_screen_params_new())

        self.rand_float = np.random.rand()
        self.rand_floats = np.random.rand(2)

    def test_handling(self):

        trajectories_a = Trajectories()
        self.assertEqual(0, len(trajectories_a))
        trajectories_a.append(self.trajectory_a)
        self.assertEqual(1, len(trajectories_a))

        trajectories_b = Trajectories(self.trajectory_b)
        self.assertEqual(1, len(trajectories_b))

        trajectories_c = Trajectories([self.trajectory_c, self.trajectory_d])
        self.assertEqual(2, len(trajectories_c))
        trajectories_c.append([self.trajectory_e, self.trajectory_f])
        self.assertEqual(4, len(trajectories_c))

        trajectories_a.append(trajectories_b._trajectories)
        self.assertEqual(2, len(trajectories_a))
        trajectories_a.append(trajectories_c._trajectories)
        self.assertEqual(6, len(trajectories_a))

        self.assertEqual(['A', 'B', 'C', 'D', 'E', 'F'], trajectories_a.users)

        trajectories_a.drop(0)
        self.assertEqual(['B', 'C', 'D', 'E', 'F'], trajectories_a.users)

        trajectories_a.drop([1, 2])
        self.assertEqual(['B', 'E', 'F'], trajectories_a.users)

        trajectories_a.select_users(['B', 'F'])
        self.assertEqual(['B', 'F'], trajectories_a.users)

        trajectories_a.select_users(['A'])
        self.assertEqual(0, len(trajectories_a))
        self.assertEqual([], trajectories_a.users)

    def test_copy(self):

        result_single = self.trajectory_a.copy()
        self.assertTrue(result_single == self.trajectory_a)
        self.assertFalse(result_single is self.trajectory_a)

        self.assertTrue(result_single.xy is self.trajectory_a.xy)

    def test_scaling(self):

        copy_trajectory_a_xy = self.trajectory_a.xy.copy()
        result_single = self.trajectory_a * self.rand_float
        self.assertTrue(np.array_equal(self.trajectory_a.xy * self.rand_float, result_single.xy))
        self.assertFalse(np.array_equal(self.trajectory_a.xy, result_single.xy))
        self.assertTrue(np.array_equal(copy_trajectory_a_xy, self.trajectory_a.xy))

        result_multi = self.trajectory_a * self.rand_floats
        result_manual = self.trajectory_a.xy.copy()
        result_manual[:, 0] = result_manual[:, 0] * self.rand_floats[0]
        result_manual[:, 1] = result_manual[:, 1] * self.rand_floats[1]
        self.assertTrue(np.array_equal(result_manual, result_multi.xy))

        result_single = self.trajectory_a / self.rand_float
        self.assertTrue(np.array_equal(self.trajectory_a.xy / self.rand_float, result_single.xy))

    def test_addition(self):

        result_single = self.trajectory_a + self.trajectory_b
        self.assertTrue(np.array_equal(self.trajectory_a.xy + self.trajectory_b.get_trajectory('pixel'),
                                       result_single.xy))

        trajectories_a = Trajectories([self.trajectory_c, self.trajectory_d])
        trajectories_b = Trajectories([self.trajectory_e, self.trajectory_f])

        result_multi = trajectories_a + trajectories_b
        self.assertTrue(np.array_equal([self.trajectory_c.xy + self.trajectory_e.get_trajectory('angle_rad'),
                                        self.trajectory_d.xy + self.trajectory_f.xy],
                                       result_multi.xys))

    def test_extension(self):

        result_single = self.trajectory_a.deepcopy()
        result_single.extend(self.trajectory_b, ignore_infos=True)
        self.assertTrue(np.array_equal(np.concatenate([self.trajectory_a.xy,
                                                       self.trajectory_b.get_trajectory('pixel')]),
                                       result_single.xy))

        trajectories_a = Trajectories([self.trajectory_c, self.trajectory_d])
        trajectories_b = Trajectories([self.trajectory_e, self.trajectory_f])

        result_multi = trajectories_a.deepcopy()
        result_multi.extend(trajectories_b, ignore_infos=True)
        self.assertTrue(np.array_equal([np.concatenate([self.trajectory_c.xy,
                                                        self.trajectory_e.get_trajectory('angle_rad')]),
                                        np.concatenate([self.trajectory_d.xy, self.trajectory_f.xy])],
                                       result_multi.xys))
