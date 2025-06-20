from typing import Tuple

import numpy as np


def camera_from_world_transform(d: float = 1.0) -> np.ndarray:
    """Define a transformation matrix in homogeneous coordinates that
    transforms coordinates from world space to camera space, according
    to the coordinate systems in Question 1.


    Args:
        d (float, optional): Total distance of displacement between world and camera
            origins. Will always be greater than or equal to zero. Defaults to 1.0.

    Returns:
        T (np.ndarray): Left-hand transformation matrix, such that c = Tw
            for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
    """
    T = np.eye(4)
    # YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    theta = (3*np.pi) / 4
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])

    t = np.array([0, 0, d])

    T[:3, :3] = R
    T[:3, 3] = t

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # END YOUR CODE
    assert T.shape == (4, 4)
    return T


def apply_transform(T: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray]:
    """Apply a transformation matrix to a set of points.

    Hint: You'll want to first convert all of the points to homogeneous coordinates.
    Each point in the (3,N) shape edges is a length 3 vector for x, y, and z, so
    appending a 1 after z to each point will make this homogeneous coordinates.

    You shouldn't need any loops for this function.

    Args:
        T (np.ndarray):
            Left-hand transformation matrix, such that c = Tw
                for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
        points (np.ndarray):
            Shape = (3,N) where 3 means 3D and N is the number of points to transform.

    Returns:
        points_transformed (Tuple of np.ndarray):
            Transformed points.
            Shape = (3,N) where 3 means 3D and N is the number of points.
    """
    N = points.shape[1]
    assert points.shape == (3, N)

    # You'll replace this!
    points_transformed = np.zeros((3, N))

    # YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    homo_coords = np.vstack([points, np.ones(N)])
    trans_points = T @ homo_coords
    points_transformed = trans_points[:3] / trans_points[3]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # END YOUR CODE

    assert points_transformed.shape == (3, N)
    return points_transformed


def intersection_from_lines(
    a_0: np.ndarray, a_1: np.ndarray, b_0: np.ndarray, b_1: np.ndarray
) -> np.ndarray:
    """Find the intersection of two lines (infinite length), each defined by a
    pair of points.

    Args:
        a_0 (np.ndarray): First point of first line; shape `(2,)`.
        a_1 (np.ndarray): Second point of first line; shape `(2,)`.
        b_0 (np.ndarray): First point of second line; shape `(2,)`.
        b_1 (np.ndarray): Second point of second line; shape `(2,)`.

    Returns:
        np.ndarray:
    """
    # Validate inputs
    assert a_0.shape == a_1.shape == b_0.shape == b_1.shape == (2,)
    assert a_0.dtype == a_1.dtype == b_0.dtype == b_1.dtype == np.float64

    # Intersection point between lines
    out = np.zeros(2)

    # YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    A1 = a_1[1] - a_0[1]
    A2 = b_1[1] - b_0[1]

    B1 = a_0[0] - a_1[0]
    B2 = b_0[0] - b_1[0]

    C1 = A1 * a_0[0] + B1 * a_0[1]
    C2 = A2 * b_0[0] + B2 * b_0[1]

    det = A1 * B2 - A2 * B1

    if det == 0: return out

    X = (B2 * C1 - B1 * C2) / det
    Y = (A1 * C2 - A2 * C1) / det
 
    out[0] = X
    out[1] = Y

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # END YOUR CODE

    assert out.shape == (2,)
    assert out.dtype == np.float64

    return out


def optical_center_from_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> np.ndarray:
    """Compute the optical center of our camera intrinsics from three vanishing
    points corresponding to mutually orthogonal directions.

    Hints:
    - Your `intersection_from_lines()` implementation might be helpful here.
    - It might be worth reviewing vector projection with dot products.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v2 (np.ndarray): Vanishing point in image space; shape `(2,)`.

    Returns:
        np.ndarray: Optical center; shape `(2,)`.
    """
    assert v0.shape == v1.shape == v2.shape == (2,), "Wrong shape!"

    optical_center = np.zeros(2)

    # YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    v0v1 = v1 - v0
    perp_v0v1 = np.array([-v0v1[1], v0v1[0]])
    v2_perp= v2 + perp_v0v1 

    v0v2 = v2 - v0
    perp_v0v2 = np.array([-v0v2[1], v0v2[0]])
    v1_perp = v1 + perp_v0v2 

    v1v2 = v2 - v1
    perp_v1v2 = np.array([-v1v2[1], v1v2[0]])
    v0_perp = v0 + perp_v1v2
    
    optical_center = intersection_from_lines(v0, v0_perp, v1, v1_perp)
    
    if np.all(optical_center == 0):
        optical_center = intersection_from_lines(v0, v0_perp, v2, v2_perp)
        if np.all(optical_center == 0):
            optical_center = intersection_from_lines(v1, v1_perp, v2, v2_perp)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # END YOUR CODE

    assert optical_center.shape == (2,)
    return optical_center


def focal_length_from_two_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, optical_center: np.ndarray
) -> np.ndarray:
    """Compute focal length of camera, from two vanishing points and the
    calibrated optical center.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        optical_center (np.ndarray): Calibrated optical center; shape `(2,)`.

    Returns:
        float: Calibrated focal length.
    """
    assert v0.shape == v1.shape == optical_center.shape == (2,), "Wrong shape!"

    f = None

    # YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    v0_trans = v0 - optical_center
    v1_trans = v1 - optical_center
    f = np.sqrt(-np.dot(v0_trans, v1_trans))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # END YOUR CODE

    return float(f)


def physical_focal_length_from_calibration(
    f: float, sensor_diagonal_mm: float, image_diagonal_pixels: float
) -> float:
    """Compute the physical focal length of our camera, in millimeters.

    Args:
        f (float): Calibrated focal length, using pixel units.
        sensor_diagonal_mm (float): Length across the diagonal of our camera
            sensor, in millimeters.
        image_diagonal_pixels (float): Length across the diagonal of the
            calibration image, in pixels.

    Returns:
        float: Calibrated focal length, in millimeters.
    """
    f_mm = None

    # YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    f_mm = f * sensor_diagonal_mm / image_diagonal_pixels

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # END YOUR CODE

    return f_mm
