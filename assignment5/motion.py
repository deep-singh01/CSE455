import numpy as np
from skimage.transform import pyramid_gaussian


def lucas_kanade(img1, img2, keypoints, window_size=5):
    """Estimate flow vector at each keypoint using Lucas-Kanade method.

    Args:
        img1 - Grayscale image of the current frame. Flow vectors are computed
            with respect to this frame.
        img2 - Grayscale image of the next frame.
        keypoints - Keypoints to track. Numpy array of shape (N, 2).
        window_size - Window size to determine the neighborhood of each keypoint.
            A window is centered around the current keypoint location.
            You may assume that window_size is always an odd number.
    Returns:
        flow_vectors - Estimated flow vectors for keypoints. flow_vectors[i] is
            the flow vector for keypoint[i]. Numpy array of shape (N, 2).

    Hints:
        - You may use np.linalg.inv to compute inverse matrix
    """
    assert window_size % 2 == 1, "window_size must be an odd number"

    flow_vectors = []
    w = window_size // 2
    img_height, img_width = img1.shape

    # Compute partial derivatives
    Iy, Ix = np.gradient(img1)
    It = img2 - img1

    # For each [y, x] in keypoints, estimate flow vector [vy, vx]
    # using Lucas-Kanade method and append it to flow_vectors.
    for y, x in keypoints:
        # Keypoints can be located between integer pixels (subpixel locations).
        # For simplicity, we round the keypoint coordinates to nearest integer.
        # In order to achieve more accurate results, image brightness at subpixel
        # locations can be computed using bilinear interpolation.
        y, x = int(round(y)), int(round(x))

        # initialize flow values to be zero
        vx = vy = 0 
        if img1[y, x] == 0 or y < w or y > img_height-w-1 or x < w or x > img_width-w-1:
            continue

        ### YOUR CODE HERE
        
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x1 = Ix[y - w:y + w + 1, x - w:x + w + 1].flatten()
        y1 = Iy[y - w:y + w + 1, x - w:x + w + 1].flatten()

        A = np.vstack((x1, y1)).T
        b = -It[y - w:y + w + 1, x - w:x + w + 1].flatten()

        try:
            vx, vy = np.linalg.solve(A.T @ A, A.T @ b)
        except np.linalg.LinAlgError:
            pass
            
        flow_vectors.append([vy, vx])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ### END YOUR CODE

    flow_vectors = np.array(flow_vectors)

    return flow_vectors


def iterative_lucas_kanade(img1, img2, keypoints, window_size=9, num_iters=7, g=None):
    """Estimate flow vector at each keypoint using iterative Lucas-Kanade method.

    Args:
        img1 - Grayscale image of the current frame. Flow vectors are computed
            with respect to this frame.
        img2 - Grayscale image of the next frame.
        keypoints - Keypoints to track. Numpy array of shape (N, 2).
        window_size - Window size to determine the neighborhood of each keypoint.
            A window is centered around the current keypoint location.
            You may assume that window_size is always an odd number.
        num_iters - Number of iterations to update flow vector.
        g - Flow vector guessed from previous pyramid level.
    Returns:
        flow_vectors - Estimated flow vectors for keypoints. flow_vectors[i] is
            the flow vector for keypoint[i]. Numpy array of shape (N, 2).
    """
    assert window_size % 2 == 1, "window_size must be an odd number"
    img_height, img_width = img1.shape

    # Initialize g as zero vector if not provided
    if g is None:
        g = np.zeros(keypoints.shape)

    flow_vectors = []
    w = window_size // 2

    # Compute spatial gradients
    Iy, Ix = np.gradient(img1)

    for y, x, gy, gx in np.hstack((keypoints, g)):
        v = np.zeros(2)  # Initialize flow vector as zero vector
        y1 = int(round(y))
        x1 = int(round(x))
        if img1[y1, x1] == 0 or y1 < w or y1 > img_height-w-1 or x1 < w or x1 > img_width-w-1:
            continue

        # TODO: Compute inverse of G at point (x1, y1)
        ### YOUR CODE HERE
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        y_1a, x_1a = max(y1-w, 0), max(x1-w, 0)
        y_1b, x_1b = y1 + w + 1, x1 + w + 1
        A = np.hstack((Ix[y_1a:y_1b, x_1a:x_1b].reshape(-1, 1), Iy[y_1a:y_1b, x_1a:x_1b].reshape(-1, 1)))
        inv_G = np.linalg.inv(np.dot(A.T, A))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ### END YOUR CODE

        # Iteratively update flow vector
        for k in range(num_iters):
            vx, vy = v
            # Refined position of the point in the next frame
            y2 = int(round(y + gy + vy))
            x2 = int(round(x + gx + vx))

            if (y2 < w or y2 > img_height-w-1 or x2 < w or x2 > img_width-w-1):
                continue

            # TODO: Compute bk and vk = inv(G) x bk
            ### YOUR CODE HERE
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            y_2a, x_2a = max(y2 - w, 0), max(x2 - w, 0)
            y_2b, x_2b = y2 + w + 1, x2 + w + 1

            bk = (img1[y_1a:y_1b, x_1a:x_1b] - img2[y_2a:y_2b, x_2a:x_2b]).reshape(-1, 1)
            vk = (inv_G @ (A.T @ bk)).flatten()

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            ### END YOUR CODE

            # Update flow vector by vk
            v += vk

        vx, vy = v
        flow_vectors.append([vy, vx])

    return np.array(flow_vectors)


def pyramid_lucas_kanade(
    img1, img2, keypoints, window_size=9, num_iters=7, level=2, scale=2
):

    """Pyramidal Lucas Kanade method

    Args:
        img1 - same as lucas_kanade
        img2 - same as lucas_kanade
        keypoints - same as lucas_kanade
        window_size - same as lucas_kanade
        num_iters - number of iterations to run iterative LK method
        level - Max level in image pyramid. Original image is at level 0 of
            the pyramid.
        scale - scaling factor of image pyramid.

    Returns:
        d - final flow vectors
    """

    # Build image pyramids of img1 and img2
    pyramid1 = tuple(pyramid_gaussian(img1, max_layer=level, downscale=scale))
    pyramid2 = tuple(pyramid_gaussian(img2, max_layer=level, downscale=scale))

    # Initialize pyramidal guess
    g = np.zeros(keypoints.shape)

    for L in range(level, -1, -1):
        ### YOUR CODE HERE
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        p = keypoints / (scale ** L)
        d = iterative_lucas_kanade(
            pyramid1[L], pyramid2[L], p,
            window_size, num_iters, g
        )

        if L > 0:
            g = scale * (g + d)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ### END YOUR CODE

    d = g + d
    return d


def compute_error(patch1, patch2):
    """Compute MSE between patch1 and patch2

        - Normalize patch1 and patch2 each to zero mean, unit variance
        - Compute mean square error between patch1 and patch2

    Args:
        patch1 - Grayscale image patch of shape (patch_size, patch_size)
        patch2 - Grayscale image patch of shape (patch_size, patch_size)
    Returns:
        error - Number representing mismatch between patch1 and patch2
    """
    assert patch1.shape == patch2.shape, "Different patch shapes"
    error = 0
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    patch1 = (patch1 - np.mean(patch1)) / np.std(patch1)
    patch2 = (patch2 - np.mean(patch2)) / np.std(patch2)
    error = np.mean((patch1 - patch2)**2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE
    return error


def track_features(
    frames,
    keypoints,
    error_thresh=1.5,
    optflow_fn=pyramid_lucas_kanade,
    exclude_border=5,
    **kwargs
):

    """Track keypoints over multiple frames

    Args:
        frames - List of grayscale images with the same shape.
        keypoints - Keypoints in frames[0] to start tracking. Numpy array of
            shape (N, 2).
        error_thresh - Threshold to determine lost tracks.
        optflow_fn(img1, img2, keypoints, **kwargs) - Optical flow function.
        kwargs - keyword arguments for optflow_fn.

    Returns:
        trajs - A list containing tracked keypoints in each frame. trajs[i]
            is a numpy array of keypoints in frames[i]. The shape of trajs[i]
            is (Ni, 2), where Ni is number of tracked points in frames[i].
    """

    kp_curr = keypoints
    trajs = [kp_curr]
    patch_size = 3  # Take 3x3 patches to compute error
    w = patch_size // 2  # patch_size//2 around a pixel

    for i in range(len(frames) - 1):
        I = frames[i]
        J = frames[i + 1]
        flow_vectors = optflow_fn(I, J, kp_curr, **kwargs)
        kp_next = kp_curr + flow_vectors

        new_keypoints = []
        for yi, xi, yj, xj in np.hstack((kp_curr, kp_next)):
            # Declare a keypoint to be 'lost' IF:
            # 1. the keypoint falls outside the image J
            # 2. the error between points in I and J is larger than threshold

            yi = int(round(yi))
            xi = int(round(xi))
            yj = int(round(yj))
            xj = int(round(xj))
            # Point falls outside the image
            if (
                yj > J.shape[0] - exclude_border - 1
                or yj < exclude_border
                or xj > J.shape[1] - exclude_border - 1
                or xj < exclude_border
            ):
                continue

            # Compute error between patches in image I and J
            patchI = I[yi - w : yi + w + 1, xi - w : xi + w + 1]
            patchJ = J[yj - w : yj + w + 1, xj - w : xj + w + 1]
            error = compute_error(patchI, patchJ)
            if error > error_thresh:
                continue

            new_keypoints.append([yj, xj])

        kp_curr = np.array(new_keypoints)
        trajs.append(kp_curr)

    return trajs


def IoU(bbox1, bbox2):
    """Compute IoU of two bounding boxes

    Args:
        bbox1 - 4-tuple (x, y, w, h) where (x, y) is the top left corner of
            the bounding box, and (w, h) are width and height of the box.
        bbox2 - 4-tuple (x, y, w, h) where (x, y) is the top left corner of
            the bounding box, and (w, h) are width and height of the box.
    Returns:
        score - IoU score
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    score = 0

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    score = (x * y) / (w1 * h1 + w2 * h2 - x * y)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return score
