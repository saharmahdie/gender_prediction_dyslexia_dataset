import numpy as np


def convert_angles_deg_to_shifted_pixel_coordinates(ab, pix_per_mm, screen_dist_mm, screen_res):
    """Converts trajectory measured in viewing angles to a trajectory in screen pixels.
    Angles are measured from the center viewpoint in horizontal and vertical direction.

    :param ab: ndarray
        2D array of gaze points defined through angles (a,b) horizontal and vertical
    :param pix_per_mm: ndarray
        screen density in x and y
    :param screen_dist_mm: float
        distance between subject and screen
    :param screen_res: ndarray
        screen resolution in pixels x,y
    :return: ndarray
        trajectory in screen pixel coordinates xy with same shape as ab
    """
    return pix_per_mm * np.tan(np.radians(ab)) * screen_dist_mm + screen_res / 2

def convert_angles_deg_to_pixel_coordinates(ab, pix_per_mm, screen_dist_mm):
    return pix_per_mm * np.tan(np.radians(ab)) * screen_dist_mm


def convert_shifted_pixel_coordinates_to_angles_deg(xy, pix_per_mm, screen_dist_mm, screen_res):
    """ convert pixel coordinates into angles coordinates

    @See convert_angles_deg_to_shifted_pixel_coordinates

    :return: ndarray
        trajectory in angles ab with same shape as xy
    """

    return np.degrees(np.arctan2((xy - screen_res/2), screen_dist_mm * pix_per_mm))

def convert_pixel_coordinates_to_angles_deg(xy, pix_per_mm, screen_dist_mm):
    return np.degrees(np.arctan2(xy, screen_dist_mm * pix_per_mm))


def convert_gaze_pixel_into_eye_coordinates(xy_pix, pix_per_mm, screen_dist_mm, screen_res):
    """Converts pixel gaze point to mm 3d point.

    :param xy: ndarray
        2D array of gaze points (x,y)
    :param pix_per_mm: ndarray
        screen density in x and y
    :param screen_dist_mm: float
        distance between subject and screen
    :param screen_res: ndarray
        screen resolution in pixels x,y
    :return: ndarray
        trajectory 3d mm coordinates with shape as xyz
        x = (w/w_pix) x_screen - (w/2)
        y = (h/h_pix) y_screen - (h/2)
        z = screen_dist_mm
    """
    assert xy_pix.shape[1] == 2
    assert xy_pix.shape[0] >= 1  # check if the input has at least 1 row

    screen_size_mm = screen_res / pix_per_mm    # we use get_screen_params function without screen_size_mm in our previous calculations, so we calculate it explicitly.
    xy_mm = (1/pix_per_mm) * (xy_pix) - screen_size_mm / 2
    z_mm = np.full((1, len(xy_mm)), screen_dist_mm)  # create a 1 x n array filled with z coordinate
    xyz_mm = np.concatenate((xy_mm, z_mm.T), axis=1)

    return xyz_mm