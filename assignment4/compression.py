import numpy as np


def compress_image(image, num_values):
    """Compress an image using SVD and keeping the top `num_values` singular values.

    Args:
        image: numpy array of shape (H, W)
        num_values: number of singular values to keep

    Returns:
        compressed_image: numpy array of shape (H, W) containing the compressed image
        compressed_size: size of the compressed image
    """
    compressed_image = None
    compressed_size = 0

    # YOUR CODE HERE
    # Steps:
    #     1. Get SVD of the image
    #     2. Only keep the top `num_values` singular values, and compute `compressed_image`
    #     3. Compute the compressed size
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    U, s, Vt = np.linalg.svd(image, full_matrices=False)

    U_compressed = U[:, :num_values]
    s_compressed = s[:num_values]
    Vt_compressed = Vt[:num_values, :]
    
    compressed_image = U_compressed @ np.diag(s_compressed) @ Vt_compressed
    compressed_size = U_compressed.size + s_compressed.size + Vt_compressed.size

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # END YOUR CODE

    assert compressed_image.shape == image.shape, \
           "Compressed image and original image don't have the same shape"

    assert compressed_size > 0, "Don't forget to compute compressed_size"

    return compressed_image, compressed_size
