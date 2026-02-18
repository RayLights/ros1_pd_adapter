import numpy as np
import cv2
from io import BytesIO
from .AttentionMasks.raindrops_generator.raindrop.dropgenerator import generateDrops, generate_label
from .AttentionMasks.raindrops_generator.raindrop.config import cfg
import yaml 
import os 
import math 
import pickle
from scipy.constants import speed_of_light as c     # in m/s
from scipy.stats import linregress
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from copy import deepcopy
import itertools
from io import BytesIO
from typing import Optional, Tuple, List, Dict, Any

import copy 
from .kernels.kernels import (
    diamond_square,
    create_disk_kernel,
    create_motion_blur_kernel,
    clipped_zoom,
)
from .utils.utilFuncs import (
    round_to_nearest_odd,
    scramble_channel,
    equalise_power,
    simple_white_balance,
    clamp_values
)


def gaussian_noise(scale, img):
    """
    Adds unfirom distributed gausian noise to an image

    Parameters:
        - img (numpy array): The input image.
         - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array
    """
    factor = [0.03, 0.06, 0.12, 0.18, 0.22][scale]
    # scale to a number between 0 and 1
    x = np.array(img, dtype=np.float32) / 255.0
    # add random between 0 and 1
    return (
        np.clip(x + np.random.normal(size=x.shape, scale=factor), 0, 1).astype(
            np.float32
        )
        * 255
    )


def poisson_noise(scale, img):
    """
    Adds poisson noise to an image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array: Image with salt and pepper noise.
    """
    factor = [120, 105, 87, 55, 30][scale]
    x = np.array(img) / 255.0
    return np.clip(np.random.poisson(x * factor) / float(factor), 0, 1) * 255


def impulse_noise(scale, img):
    """
    Add salt and pepper noise to an image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array: Image with salt and pepper noise.
    """
    factor = [0.01, 0.02, 0.04, 0.065, 0.10][scale]
    # Number of salt noise pixels
    num_salt = np.ceil(factor * img.size * 0.5)
    # Add salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    img[tuple(coords)] = 255
    # Number of pepper noise pixels
    num_pepper = np.ceil(factor * img.size * 0.5)
    # Add pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    img[tuple(coords)] = 0
    return img


def defocus_blur(scale, image):
    """
    Applies a defocus blur to the given image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    factor = [2, 5, 6, 9, 12][scale]
    # Create the disk-shaped kernel.
    kernel = create_disk_kernel(factor)
    # Convolve the image with the kernel.
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image


def glass_blur(scale, image):
    """
    Applies glass blur effect to the given image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    factor = [2, 5, 6, 9, 12][scale]
    # Get the height and width of the image.
    height, width = image.shape[:2]
    # Generate random offsets for each pixel in the image.
    rand_x = np.random.randint(-factor, factor + 1, size=(height, width))
    rand_y = np.random.randint(-factor, factor + 1, size=(height, width))
    # Compute the new coordinates for each pixel after adding the random offsets.
    # Ensure that the new coordinates are within the image boundaries.
    coord_x = np.clip(np.arange(width) + rand_x, 0, width - 1)
    coord_y = np.clip(np.arange(height).reshape(-1, 1) + rand_y, 0, height - 1)
    # Create the glass-blurred image using the new coordinates.
    glass_blurred_image = image[coord_y, coord_x]
    return glass_blurred_image


def motion_blur(scale, image, size=10, angle=45):
    """
    Apply motion blur to the given image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    size, angle = [(2, 5), (4, 12), (6, 20), (10, 30), (15, 45)][scale]
    # Create the motion blur kernel.
    kernel = create_motion_blur_kernel(size, angle)
    # Convolve the image with the kernel.
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image


def zoom_blur(scale, img):
    """
    Applies a zoom blur effect on an image.\n
    This perturbation has an avereage duration of 36ms on an input image of 256*256*3

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    c = [
        np.arange(1, 1.01, 0.01),
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.15, 0.02),
        np.arange(1, 1.21, 0.02),
        np.arange(1, 1.31, 0.03),
    ][scale]
    img = (np.array(img) / 255.0).astype(np.float32)
    out = np.zeros_like(img)
    for zoom_factor in c:
        out += clipped_zoom(img, zoom_factor)
    img = (img + out) / (len(c) + 1)
    return np.clip(img, 0, 1) * 255


def increase_brightness(scale, image):
    """
    Increase the brightness of the image using HSV color space

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    factor = [1.1, 1.2, 1.3, 1.5, 1.7][scale]
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Adjust the V channel
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * factor, 0, 255)
    # Convert the image back to RGB color space
    brightened_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return brightened_image


def contrast(scale, img):
    """
    Increase or decrease the conrast of the image

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    factor = [1.1, 1.2, 1.3, 1.5, 1.7][scale]
    pivot = 127.5
    return np.clip(pivot + (img - pivot) * factor, 0, 255)


def elastic(scale, img):
    """
    Applies an elastic deformation on the image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    alpha, sigma = [(2, 0.4), (3, 0.75), (5, 0.9), (7, 1.2), (10, 1.5)][scale]
    # Generate random displacement fields
    dx = np.random.uniform(-1, 1, img.shape[:2]) * alpha
    dy = np.random.uniform(-1, 1, img.shape[:2]) * alpha
    # Smooth the displacement fields
    dx = cv2.GaussianBlur(dx, (0, 0), sigma)
    dy = cv2.GaussianBlur(dy, (0, 0), sigma)
    # Create a meshgrid of indices and apply the displacements
    x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    # Map the distorted image back using linear interpolation
    distorted_image = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return distorted_image


def pixelate(scale, img):
    """
    Pixelates the image

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    factor = [0.85, 0.55, 0.35, 0.2, 0.1][scale]
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * factor), int(h * factor)), cv2.INTER_AREA)
    return cv2.resize(img, (w, h), cv2.INTER_NEAREST)


def jpeg_filter(scale, image):
    """
    Introduce JPEG compression artifacts to the image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    factor = [30, 18, 15, 10, 5][scale]
    # Encode the image as JPEG with the specified quality
    _, jpeg_encoded_image = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), factor]
    )
    # Convert the JPEG encoded bytes to an in-memory binary stream
    jpeg_stream = BytesIO(jpeg_encoded_image.tobytes())
    # Decode the JPEG stream back to an image
    jpeg_artifact_image = cv2.imdecode(
        np.frombuffer(jpeg_stream.read(), np.uint8), cv2.IMREAD_COLOR
    )
    return jpeg_artifact_image


def shear_image(scale, image):
    """
    Apply horizontal shear to an image with different severities.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    shear_factor = [0.12, 0.2, 0.32, 0.45, 0.6][scale]
    # Load the image
    if image is None:
        raise ValueError("Image not found at the given path.")

    rows, cols, _ = image.shape

    # Define the shear matrix
    M = np.array([[1, shear_factor, 0], [0, 1, 0]])

    sheared = cv2.warpAffine(image, M, (cols, rows))

    return sheared


def translate_image(scale, image):
    """
    Apply translation to an image with different severities in both x and y directions.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    tx, ty = [(-0.1, 0.1), (25, -25), (40, -40), (65, -65), (90, -90)][scale]
    # Load the image
    if image is None:
        raise ValueError("Image not found at the given path.")

    rows, cols, _ = image.shape

    # Define the translation matrix
    M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)

    translated = cv2.warpAffine(image, M, (cols, rows))
    return translated


def scale_image(scale, image):
    """
    Apply scaling to an image with different severities while maintaining source dimensions.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    scale_factor = [0.96, 0.9, 0.8, 0.68, 0.5][scale]
    rows, cols, _ = image.shape

    # Resize the image
    new_dimensions = (int(cols * scale_factor), int(rows * scale_factor))
    scaled = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)

    # If scaled image is smaller, pad it
    if scale_factor < 1:
        top_pad = (rows - scaled.shape[0]) // 2
        bottom_pad = rows - scaled.shape[0] - top_pad
        left_pad = (cols - scaled.shape[1]) // 2
        right_pad = cols - scaled.shape[1] - left_pad
        scalled_image = cv2.copyMakeBorder(
            scaled,
            top_pad,
            bottom_pad,
            left_pad,
            right_pad,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
    # If scaled image is larger, crop it
    else:
        start_row = (scaled.shape[0] - rows) // 2
        start_col = (scaled.shape[1] - cols) // 2
        scalled_image = scaled[
            start_row : start_row + rows, start_col : start_col + cols
        ]

    return scalled_image


def rotate_image(scale, image):
    """
    Apply rotation to an image with different severities while maintaining source dimensions.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    angle = [10, 20, 45, 90, 180][scale]
    rows, cols, _ = image.shape
    center = (cols / 2, rows / 2)

    # Compute the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)

    # Apply the rotation
    rotated = cv2.warpAffine(image, M, (cols, rows), borderValue=(0, 0, 0))

    return rotated


def fog_mapping(scale, image):
    """
    Apply fog effect to an image with different severities using Diamond-Square algorithm.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.05, 0.12, 0.22, 0.35, 0.6][scale]
    rows, cols, _ = image.shape
    # Determine size for diamond-square algorithm (closest power of 2 plus 1)
    size = 2 ** int(np.ceil(np.log2(max(rows, cols)))) + 1

    # Generate fog pattern
    fog_pattern = diamond_square(size, 0.6)
    # Resize fog pattern to image size and normalize to [0, 255]
    fog_pattern_resized = cv2.resize(fog_pattern, (cols, rows))
    fog_pattern_resized = (
        (fog_pattern_resized - fog_pattern_resized.min())
        / (fog_pattern_resized.max() - fog_pattern_resized.min())
        * 255
    ).astype(np.uint8)
    fog_pattern_rgb = cv2.merge(
        [fog_pattern_resized, fog_pattern_resized, fog_pattern_resized]
    )  # Convert grayscale to RGB

    # Blend fog with image
    foggy = cv2.addWeighted(image, 1 - severity, fog_pattern_rgb, severity, 0)

    return foggy


def splatter_mapping(scale, image):
    """
    Apply splatter effect to an image with different severities.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.1, 0.2, 0.3, 0.4, 0.5][scale]
    rows, cols, _ = image.shape

    # Determine number and size of splatters based on severity
    num_splotches = int(severity * 50)
    max_splotch_size = max(6, int(severity * 50))

    splattered = image.copy()
    for _ in range(num_splotches):
        center_x = np.random.randint(0, cols)
        center_y = np.random.randint(0, rows)
        splotch_size = np.random.randint(5, max_splotch_size)

        # Generate a mask for splotch and apply to the image
        y, x = np.ogrid[-center_y : rows - center_y, -center_x : cols - center_x]
        mask = x * x + y * y <= splotch_size * splotch_size
        splattered[mask] = [0, 0, 0]  # Obscuring the region with black color

    return splattered


def dotted_lines_mapping(scale, image):
    """
    Apply dotted lines effect to an image with different severities.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.1, 0.2, 0.3, 0.4, 0.5][scale]

    rows, cols, _ = image.shape

    # Determine parameters based on severity
    num_lines = int(scale * 10)
    distance_between_dots = max(10, int(50 * (1 - severity)))
    dot_thickness = int(severity * 5)

    dotted = image.copy()
    for _ in range(num_lines):
        start_x = np.random.randint(0, cols)
        start_y = np.random.randint(0, rows)
        direction = np.random.rand(2) * 2 - 1  # Random direction
        direction /= np.linalg.norm(direction)  # Normalize to unit vector

        current_x, current_y = start_x, start_y
        while 0 <= current_x < cols and 0 <= current_y < rows:
            cv2.circle(
                dotted, (int(current_x), int(current_y)), dot_thickness, (0, 0, 0), -1
            )  # Draw dot
            current_x += direction[0] * distance_between_dots
            current_y += direction[1] * distance_between_dots

    return dotted


def zigzag_mapping(scale, image):
    """
    Apply zigzag effect to an image with different severities.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.1, 0.2, 0.3, 0.4, 0.6][scale]

    rows, cols, _ = image.shape

    # Determine parameters based on severity
    num_lines = int(severity * 10)
    amplitude = int(20 * severity)
    frequency = int(10 * severity)

    zigzag = image.copy()
    for _ in range(num_lines):
        start_x = np.random.randint(0, cols)
        start_y = np.random.randint(0, rows)
        direction = np.random.rand(2) * 2 - 1  # Random direction
        direction /= np.linalg.norm(direction)  # Normalize to unit vector

        current_x, current_y = start_x, start_y
        step = 0
        while 0 <= current_x < cols and 0 <= current_y < rows:
            # Calculate offset for zigzag
            offset = amplitude * np.sin(frequency * step)
            current_x += direction[0]
            current_y += direction[1] + offset
            if 0 <= current_x < cols and 0 <= current_y < rows:
                zigzag[int(current_y), int(current_x)] = [0, 0, 0]  # Draw on image
            step += 1
    return zigzag


def canny_edges_mapping(scale, image):
    """
    Apply Canny edge detection to an image with different severities.
    The detected edges are highlited and put on top of the input image

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.01, 0.1, 0.25, 0.4, 0.7][scale]
    edge_color = (255, 0, 0)

    # Convert the image to grayscale for edge detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Calculate low and high thresholds based on severity
    low_threshold = int(50 + severity * 100)
    high_threshold = int(150 + severity * 100)

    canny = cv2.Canny(gray_image, low_threshold, high_threshold)
    colored_edges = np.zeros_like(image)

    # Color the detected edges with the specified color
    colored_edges[canny > 0] = edge_color
    # Merge the colored edges with the original image
    merged_image = cv2.addWeighted(image, 0.7, colored_edges, 0.3, 0)

    return merged_image


def speckle_noise_filter(scale, image):
    """
    Apply speckle noise to an image with different severities.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.02, 0.05, 0.09, 0.14, 0.2][scale]

    rows, cols, _ = image.shape
    # Generate noise pattern
    noise = np.random.normal(1, severity, (rows, cols, 3))

    # Apply speckle noise by multiplying original image with noise pattern
    speckled = (image * noise).clip(0, 255).astype(np.uint8)
    return speckled


def false_color_filter(scale, image):
    """
    Apply false color effect to an image with different severities.
    Severity 1: The Red and Blue channels are swapped.
    Severity 2: The Red and Green channels are swapped.
    Severity 3: The Blue and Green channels are swapped.
    Severity 4: Each channel is inverted.
    Severity 5: Channels are averaged with their adjacent channels.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """

    false_color = image.copy()

    # Depending on the severity, we swap or mix channels in different ways
    if scale == 0:
        false_color[:, :, 0] = image[:, :, 1]
        false_color[:, :, 1] = image[:, :, 2]
        false_color[:, :, 2] = image[:, :, 0]
    elif scale == 1:
        false_color[:, :, 0] = image[:, :, 1]
        false_color[:, :, 1] = image[:, :, 0]
        false_color[:, :, 2] = image[:, :, 2]
    elif scale == 2:
        false_color[:, :, 0] = image[:, :, 2]
        false_color[:, :, 1] = image[:, :, 1]
        false_color[:, :, 2] = image[:, :, 0]
    elif scale == 3:
        false_color[:, :, 0] = 255 - image[:, :, 0]
        false_color[:, :, 1] = 255 - image[:, :, 1]
        false_color[:, :, 2] = 255 - image[:, :, 2]
    elif scale == 4:
        false_color[:, :, 0] = (image[:, :, 0] + image[:, :, 1]) // 2
        false_color[:, :, 1] = (image[:, :, 1] + image[:, :, 2]) // 2
        false_color[:, :, 2] = (image[:, :, 2] + image[:, :, 0]) // 2

    return false_color


def high_pass_filter(scale, image):
    """
    Apply high pass filter to an image with different severities.
    This filter highlightes edges and fine details in an image as well
    as darkens the input image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    kernel_size = [35, 59, 83, 107, 113][scale]

    image_float32 = np.float32(image)
    # Blur the image to get the low frequency components
    low_freq = cv2.GaussianBlur(image_float32, (kernel_size, kernel_size), 0)

    high_freq = image_float32 - low_freq
    sharpened = image_float32 + high_freq

    sharpened = np.clip(sharpened, 0, 255).astype("uint8")
    return sharpened


def low_pass_filter(scale, image):
    """
    Apply low pass filter to an image with different severities while preserving color.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """

    kernel_size = [15, 23, 30, 36, 40][scale]

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Split the channels
    H, S, V = cv2.split(hsv_image)

    # Apply Gaussian blur to the V channel (value/brightness)
    blurred_V = cv2.GaussianBlur(
        V,
        (
            round_to_nearest_odd(int(kernel_size)),
            round_to_nearest_odd(int(kernel_size)),
        ),
        0,
    )

    # Merge the blurred V channel with the original H and S channels
    merged_hsv = cv2.merge([H, S, blurred_V])

    # Convert back to RGB color space
    low_pass_rgb = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2RGB)

    return low_pass_rgb


def phase_scrambling(scale, image):
    """
    Apply power scrambling (phase scrambling) to an image with different severities.
    Phase scrambling involves manipulating the phase information of an image's
    Fourier transform while keeping the magnitude intact. This results in an
    image that retains its overall power spectrum but has its content scrambled.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.05, 0.15, 0.26, 0.38, 0.55][scale]
    # Scramble each channel
    R, G, B = cv2.split(image)
    scrambled_R = scramble_channel(R, severity)
    scrambled_G = scramble_channel(G, severity)
    scrambled_B = scramble_channel(B, severity)

    # Merging the scrambled channels
    scrambled_rgb = cv2.merge([scrambled_R, scrambled_G, scrambled_B])

    return scrambled_rgb


def histogram_equalisation(scale, image):
    """
    Apply histogram equalisation to an image with different severities while
    preserving color.
    We enhance the contrast of an image by effectively spreading out the
    pixel intensities in an image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    clip_limit = [1, 3, 5, 7, 10][scale]
    equalised_images = []

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv_image)

    # Apply CLAHE to the V channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    equalised_V = clahe.apply(V)

    # Merging the equalised V channel with original H and S
    equalised_hsv = cv2.merge([H, S, equalised_V])

    # Convert back to RGB color space
    equalised_rgb = cv2.cvtColor(equalised_hsv, cv2.COLOR_HSV2RGB)

    return equalised_rgb


def reflection_filter(scale, image):
    """
    Apply a reflection effect to an image with different intensity.
    Creates a mirror effect to the input image and appends the mirrored image to
    the bottom of the image. The intensity refers to the share of the image
    which is appended at the bottom (20%, 30%, 45%, 60% or 90%).

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.2, 0.3, 0.45, 0.6, 0.9][scale]
    # Calculate the portion of the image to reflect
    portion_to_reflect = int(image.shape[0] * severity)

    # Create the reflection by flipping vertically
    reflection = cv2.flip(image[-portion_to_reflect:], 0)

    # Stack the original image and its reflection
    reflected_img = np.vstack((image, reflection[:portion_to_reflect]))

    # Resize the image to maintain original dimensions
    reflected_img = cv2.resize(reflected_img, (image.shape[1], image.shape[0]))

    return reflected_img


def white_balance_filter(scale, image):
    """
    Apply a white balance effect to an image with different intensities.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.1, 0.25, 0.5, 0.75, 0.99][scale]
    return cv2.addWeighted(
        image, 1 - severity, simple_white_balance(image.copy()), severity, 0
    )


def sharpen_filter(scale, image):
    """
    Apply a sharpening effect to an image with different severities using a
    simple sharpen konvolution via a kernel matrix.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [1, 2, 3, 4, 5][scale]
    weight = [0.9, 0.8, 0.7, 0.6, 0.5][scale]

    # Base sharpening kernel
    kernel = np.array([[-1, -1, -1], [-1, 8 + severity, -1], [-1, -1, -1]])

    # Convolve the image with the sharpening kernel
    sharpened = cv2.filter2D(image, -1, kernel)
    return cv2.addWeighted(image, weight, sharpened, 1 - weight, 0)


def grayscale_filter(scale, image):
    """
    Apply a grayscale effect to an image with different intensities.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """

    severity = [0.1, 0.2, 0.35, 0.55, 0.85][scale]
    # Convert the image to grayscale
    grayscale_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grayscale_img_colored = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)

    # Interpolate between the original and grayscale image based on severity
    grayed_img = cv2.addWeighted(
        image, 1 - severity, grayscale_img_colored, severity, 0
    )

    return grayed_img


def posterize_filter(scale, image):
    """
    Reduces the number of distinct colors while mainting essential image features

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    scale = [128, 64, 32, 8, 4][scale]

    # Posterize the image
    indices = np.arange(0, 256)
    divider = np.linspace(0, 255, scale + 1)[1]
    quantiz = np.int0(np.linspace(0, 255, scale))
    color_levels = (indices / divider).astype(int) * (255 // (scale - 1))
    color_levels = np.clip(color_levels, 0, 255).astype(int)

    # Apply posterization for each channel
    posterized = np.zeros_like(image)
    for i in range(3):  # For each channel: B, G, R
        posterized[:, :, i] = color_levels[image[:, :, i]]

    return posterized


def cutout_filter(scale, image):
    """
    Creates random cutouts on the picture and makes the random cutouts black

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    scale = [1, 2, 4, 6, 10][scale]

    h, w, _ = image.shape

    # Apply patches to the image
    for _ in range(scale):
        # Determine patch size
        patch_size_x = np.random.randint(h * 0.05, h * 0.2)
        patch_size_y = np.random.randint(w * 0.05, w * 0.2)

        # Determine top-left corner of the patch
        x = np.random.randint(0, h - patch_size_x)
        y = np.random.randint(0, w - patch_size_y)

        # Apply the patch
        image[x : x + patch_size_x, y : y + patch_size_y, :] = 0  # set to black

    return image


def sample_pairing_filter(scale, image):
    """
    Randomly sample to regions of the image together

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """

    alpha = [0.9, 0.7, 0.5, 0.3, 0.1][scale]

    # Randomly select a section of the image
    h, w, _ = image.shape
    start_x = np.random.randint(0, w // 2)
    start_y = np.random.randint(0, h // 2)
    end_x = start_x + w // 2
    end_y = start_y + h // 2

    random_section = image[start_y:end_y, start_x:end_x]

    # Resize the section to the size of the original image
    random_section_resized = cv2.resize(random_section, (w, h))

    # Blend the image and the section
    blended = cv2.addWeighted(image, alpha, random_section_resized, 1 - alpha, 0)

    return blended


def gaussian_blur(scale, image):
    """
    Applies gaussian blur to the image

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """

    kernel_size = [(3, 3), (7, 7), (15, 15), (25, 25), (41, 41)][scale]

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(image, kernel_size, 0)

    return blurred


def saturation_filter(scale, image):
    """
    Increases the saturation of the image

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """

    multiplier = [1.05, 1.15, 1.4, 1.65, 1.9][scale]

    # Adjust the saturation channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * multiplier, 0, 255)

    # Convert the modified HSV image back to the RGB color space
    saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return saturated


def saturation_decrease_filter(scale, image):
    """
    Decreases the saturation of the image

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """

    multiplier = [0.9, 0.85, 0.6, 0.35, 0.1][scale]

    # Adjust the saturation channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * multiplier, 0, 255)

    # Convert the modified HSV image back to the RGB color space
    saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return saturated


def fog_filter(scale, image):
    """
    Apply a fog effect to the image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    intensity, noise_amount = [
        (0.1, 0.05),
        (0.2, 0.1),
        (0.3, 0.2),
        (0.45, 0.3),
        (0.65, 0.45),
    ][scale]
    # Create a white overlay of the same size as the image
    fog_overlay = np.ones_like(image) * 255
    # Optionally, introduce some noise to the fog overlay
    noise = np.random.normal(scale=noise_amount * 255, size=image.shape).astype(
        np.uint8
    )
    fog_overlay = cv2.addWeighted(fog_overlay, 1 - noise_amount, noise, noise_amount, 0)
    # Blend the fog overlay with the original image
    foggy_image = cv2.addWeighted(image, 1 - intensity, fog_overlay, intensity, 0)
    return foggy_image


def frost_filter(scale, image):
    """
    Apply a frost effect to the image using an overlay image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    intensity = [0.15, 0.19, 0.25, 0.32, 0.4][scale]
    frost_image_path = "./perturbationdrive/perturbationdrive/OverlayImages/frostImg.png"
    # Load the frost overlay image
    frost_overlay = cv2.imread(frost_image_path, cv2.IMREAD_UNCHANGED)
    assert (
        frost_overlay is not None
    ), "file could not be read, check with os.path.exists()"
    # Resize the frost overlay to match the input image dimensions
    frost_overlay_resized = cv2.resize(frost_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = frost_overlay_resized[:, :, :3]
    alpha = frost_overlay_resized[:, :, 3] / 255.0  # Normalize to [0, 1]
    # Blend the frost overlay with the original image using the alpha channel for transparency
    frosted_image = (1 - (intensity * alpha[:, :, np.newaxis])) * image + (
        intensity * bgr
    )
    frosted_image = np.clip(frosted_image, 0, 255).astype(np.uint8)
    # Decrease saturation to give a cold appearance
    hsv = cv2.cvtColor(frosted_image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 0.8
    frosted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frosted_image


def snow_filter(scale, image):
    """
    Apply a snow effect to the image using an overlay image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    intensity = [0.15, 0.22, 0.3, 0.45, 0.6][scale]
    frost_image_path = "./perturbationdrive/perturbationdrive//OverlayImages/snow.png"
    # Load the frost overlay image
    frost_overlay = cv2.imread(frost_image_path, cv2.IMREAD_UNCHANGED)
    assert (
        frost_overlay is not None
    ), "file could not be read, check with os.path.exists()"
    # Resize the frost overlay to match the input image dimensions
    frost_overlay_resized = cv2.resize(frost_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = frost_overlay_resized[:, :, :3]
    alpha = frost_overlay_resized[:, :, 3] / 255.0  # Normalize to [0, 1]
    # Blend the frost overlay with the original image using the alpha channel for transparency
    frosted_image = (1 - (intensity * alpha[:, :, np.newaxis])) * image + (
        intensity * bgr
    )
    frosted_image = np.clip(frosted_image, 0, 255).astype(np.uint8)
    # Decrease saturation to give a cold appearance
    hsv = cv2.cvtColor(frosted_image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 0.8
    frosted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frosted_image


def dynamic_snow_filter(scale, image, iterator):
    """
    Apply a dynamic snow effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    # Load the next frame from the iterator
    snow_overlay = next(iterator)
    snow_overlay = shift_color(snow_overlay, [71, 253, 135], [255, 255, 255])

    if (
        snow_overlay.shape[0] != image.shape[0]
        or snow_overlay.shape[1] != image.shape[1]
    ):
        snow_overlay = cv2.resize(snow_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = snow_overlay[:, :, :3]
    mask = snow_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def static_snow_filter(scale, image, snow_overlay):
    """
    Apply a static snow effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    snow_overlay = shift_color(snow_overlay, [71, 253, 135], [255, 255, 255])

    if (
        snow_overlay.shape[0] != image.shape[0]
        or snow_overlay.shape[1] != image.shape[1]
    ):
        snow_overlay = cv2.resize(snow_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = snow_overlay[:, :, :3]
    mask = snow_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def dynamic_rain_filter(scale, image, iterator):
    """
    Apply a dynamic rain effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    rain_overlay = next(iterator)
    rain_overlay = shift_color(rain_overlay, [31, 146, 59], [191, 35, 0])

    # Load the next frame from the iterator
    if (
        rain_overlay.shape[0] != image.shape[0]
        or rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1.0 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def dynamic_raindrop_filter(scale, image, iterator):
    """
    Apply a dynamic rain dropeffect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    # Load the next frame from the iterator
    overlay = next(iterator)
    overlay = shift_color(overlay, [71, 253, 135], [255, 255, 255])


    # Load the next frame from the iterator
    if (
        overlay.shape[0] != image.shape[0]
        or overlay.shape[1] != image.shape[1]
    ):
        overlay = cv2.resize(overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = overlay[:, :, :3]
    mask = overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1.0 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def static_rain_filter(scale, image, rain_overlay):
    """
    Apply a dynamic rain effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    rain_overlay = shift_color(rain_overlay, [31, 146, 59], [191, 35, 0])

    if (
        rain_overlay.shape[0] != image.shape[0]
        or rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1.0 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def new_rain_filter(scale, image):
    """
    Generate procedural raindrops on the input image using the raindrops_generator.

    Interface: func(intensity, image) -> image

    Parameters:
        - scale (int): Severity in [0..4]. Controls drop count and radius.
        - image (numpy array): Input image (H, W, 3) uint8.

    Returns: numpy array with rendered raindrops.
    """
    if not (0 <= scale <= 4):
        raise ValueError("Scale must be within [0, 4].")

    h, w = image.shape[:2]
    # Derive a local config from the base cfg without mutating the global one
    local_cfg = copy.deepcopy(cfg)

    # Severity schedules: increase number and size of drops with scale
    # Scales chosen to be subtle at low levels and heavy at high levels
    drop_scales = [0.4, 0.7, 1.0, 1.3, 1.7]
    radius_scales = [0.6, 0.8, 1.0, 1.2, 1.35]
    edge_darkratio = [0.85, 0.75, 0.65, 0.6, 0.55][scale]  # darker edges with severity

    # Apply schedules to min/max values with sane lower/upper bounds
    local_cfg["minDrops"] = max(5, int(round(local_cfg["minDrops"] * drop_scales[scale])))
    local_cfg["maxDrops"] = max(local_cfg["minDrops"] + 1, int(round(local_cfg["maxDrops"] * drop_scales[scale])))

    local_cfg["minR"] = max(6, int(round(local_cfg["minR"] * radius_scales[scale])))
    local_cfg["maxR"] = max(local_cfg["minR"] + 1, int(round(local_cfg["maxR"] * radius_scales[scale])))

    local_cfg["edge_darkratio"] = float(edge_darkratio)
    # No label output needed here
    local_cfg["return_label"] = False

    # Generate drops and render them onto the image
    drops, _, _ = generate_label(h, w,None, local_cfg)
    output = generateDrops(image, local_cfg, drops)

    return np.asarray(output, dtype=np.uint8)


def new_dynamic_rain_filter(scale, image):
    """
    Generate dynamic (frame-varying) raindrops using the raindrops_generator.

    Each call creates a fresh set of drops, optionally with motion streaks to
    simulate movement. Interface: func(intensity, image) -> image.

    Parameters:
        - scale (int): Severity in [0..4]. Controls count/size and streak length.
        - image (numpy array): Input image (H, W, 3) uint8.

    Returns: numpy array with rendered dynamic raindrops.
    """
    if not (0 <= scale <= 4):
        raise ValueError("Scale must be within [0, 4].")

    h, w = image.shape[:2]

    local_cfg = copy.deepcopy(cfg)

    # Dynamic severity schedules
    drop_scales = [0.5, 0.8, 1.1, 1.5, 2.0]
    radius_scales = [0.55, 0.8, 1.0, 1.2, 1.35]
    edge_darkratio = [0.85, 0.75, 0.65, 0.6, 0.55][scale]
    streak_lengths = [0, 2, 4, 6, 8]  # motion blur length per drop

    local_cfg["minDrops"] = max(8, int(round(local_cfg["minDrops"] * drop_scales[scale])))
    local_cfg["maxDrops"] = max(local_cfg["minDrops"] + 2, int(round(local_cfg["maxDrops"] * drop_scales[scale])))
    local_cfg["minR"] = max(6, int(round(local_cfg["minR"] * radius_scales[scale])))
    local_cfg["maxR"] = max(local_cfg["minR"] + 1, int(round(local_cfg["maxR"] * radius_scales[scale])))
    local_cfg["edge_darkratio"] = float(edge_darkratio)
    local_cfg["return_label"] = False

    # Create drops for this frame
    drops, _, _ = generate_label(h, w, None, local_cfg)

    # Add per-drop motion parameter for generateDrops to render simple streaks
    ml = streak_lengths[scale]
    if ml > 0:
        for d in drops:
            try:
                setattr(d, "motion_length", int(ml))
            except Exception:
                pass

    output = generateDrops(image, local_cfg, drops)
    return np.asarray(output, dtype=np.uint8)
# check maps

def object_overlay(scale, img1):
    c = [10, 5, 3, 2, 1.5]
    overlay_path = "./perturbationdrive/perturbationdrive/OverlayImages/Logo_of_the_Technical_University_of_Munichpng.png"
    img2 = cv2.imread(overlay_path)
    assert img2 is not None, "file could not be read, check with os.path.exists()"
    img1_shape0_div_c_scale = int(img1.shape[0] / c[scale])
    img1_shape1_div_2 = int(img1.shape[1] / 2)
    img1_shape0_div_2 = int(img1.shape[0] / 2)

    # Calculate scale factor and target image width directly without extra division
    targetImageWidth = int(
        img1.shape[1] * (img1_shape0_div_c_scale * 100.0 / img2.shape[0]) / 100
    )

    # Resize img2 in a more efficient manner
    img2 = cv2.resize(
        img2,
        (img1_shape0_div_c_scale, targetImageWidth),
        interpolation=cv2.INTER_NEAREST,
    )

    # Precompute reused expressions
    img2_shape0_div_2 = int(img2.shape[0] / 2)
    img2_shape1_div_2 = int(img2.shape[1] / 2)

    # Calculate the start of the ROI
    height_roi = img1_shape0_div_2 - img2_shape0_div_2
    width_roi = img1_shape1_div_2 - img2_shape1_div_2

    rows, cols, _ = img2.shape
    roi = img1[height_roi : height_roi + rows, width_roi : width_roi + cols]

    # Now create a mask of the logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of the logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only the region of the logo from the logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    # Put the logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[height_roi : height_roi + rows, width_roi : width_roi + cols] = dst

    return img1


def dynamic_object_overlay(scale, image, iterator):
    """
    Apply a dynamic bird flying effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    # Load the next frame from the iterator
    rain_overlay = next(iterator)
    rain_overlay = shift_color(rain_overlay, [175, 221, 202], [0, 0, 0])

    # Resize the frost overlay to match the input image dimensions
    if (
        rain_overlay.shape[0] != image.shape[0]
        or rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def static_object_overlay(scale, image, rain_overlay):
    """
    Apply a dynamic bird flying effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    rain_overlay = shift_color(rain_overlay, [175, 221, 202], [0, 0, 0])

    # Resize the frost overlay to match the input image dimensions
    if (
        rain_overlay.shape[0] != image.shape[0]
        or rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def dynamic_sun_filter(scale, image, iterator):
    """
    Apply a dynamic sun effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    # Load the next frame from the iterator
    rain_overlay = next(iterator)
    rain_overlay = shift_color(rain_overlay, [223, 234, 212], [28, 202, 255])

    # Resize the frost overlay to match the input image dimensions
    if (
        rain_overlay.shape[0] != image.shape[0]
        or rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def static_sun_filter(scale, image, rain_overlay):
    """
    Apply a dynamic sun effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    # Load the next frame from the iterator
    rain_overlay = shift_color(rain_overlay, [223, 234, 212], [28, 202, 255])

    if (
        rain_overlay.shape[0] != image.shape[0]
        or rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def dynamic_lightning_filter(scale, image, iterator):
    """
    Apply a dynamic lightning effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    # Load the next frame from the iterator
    rain_overlay = next(iterator)
    rain_overlay = shift_color(rain_overlay, [5, 122, 101], [8, 152, 188])

    # Resize the frost overlay to match the input image dimensions
    if (
        rain_overlay.shape[0] != image.shape[0]
        or rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def static_lightning_filter(scale, image, rain_overlay):
    """
    Apply a dynamic lightning effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    rain_overlay = shift_color(rain_overlay, [5, 122, 101], [8, 152, 188])

    if (
        rain_overlay.shape[0] != image.shape[0]
        or rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def dynamic_smoke_filter(scale, image, iterator):
    """
    Apply a dynamic smoke effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    # Load the next frame from the iterator
    rain_overlay = next(iterator)
    rain_overlay = shift_color(rain_overlay, [30, 112, 65], [132, 132, 132])

    # Resize the frost overlay to match the input image dimensions
    if (
        rain_overlay.shape[0] != image.shape[0]
        or rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def static_smoke_filter(scale, image, rain_overlay):
    """
    Apply a dynamic smoke effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    rain_overlay = shift_color(rain_overlay, [30, 112, 65], [132, 132, 132])
    if (
        rain_overlay.shape[0] != image.shape[0]
        or rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def perturb_high_attention_regions(
    saliency_map, image, perturbation, boundary=0.5, scale=0
):
    """
    Perturbs the regions of an image where the saliency map has an value greater than boundary
    Can be used with either vanilla saliency map or grad-cam map

    Parameters:
        - saliency_map (numpy array): Two dimensional saliency map
        - img (numpy array): The input image. Needs to have the same dimensions as the image
        - perturbation func: The perturbation to apply to the image
        - boundary float=0.5: The boundary value above which to perturb the image regions. Needs to be in the range of [0, 1]
        - scale int=0: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    if boundary < 0 or boundary > 1:
        raise ValueError("The boundary value needs to be in the range of [0, 1]")
    # Create a binary mask from the array
    mask = saliency_map > boundary
    # Apply the gaussian noise to the whole image
    noise_img = perturbation(scale, image)
    # Now apply the mask: replace the original image pixels with noisy pixels where mask is True
    image[mask] = noise_img[mask]
    return image


def perturb_highest_n_attention_regions(
    saliency_map, image, perturbation, n=30, scale=0
):
    """
    Perturbs the highest n% of the regions of an image where the saliency map has an value greater than threshold
    """
    if n < 0 or n > 100:
        raise ValueError("The threshold value needs to be in the range of [0, 100]")
    # Create a binary mask from the array
    mask = saliency_map > np.percentile(saliency_map, n)
    # Apply the gaussian noise to the whole image
    noise_img = perturbation(scale, image)
    # Now apply the mask: replace the original image pixels with noisy pixels where mask is True
    image[mask] = noise_img[mask]
    return image


def perturb_lowest_n_attention_regions(
    saliency_map, image, perturbation, n=30, scale=0
):
    """
    Perturbs the lowest n% of the regions of an image where the saliency map has an value greater than threshold
    """
    if n < 0 or n > 100:
        raise ValueError("The threshold value needs to be in the range of [0, 100]")
    # Create a binary mask from the array
    thres = np.percentile(saliency_map, n)
    if thres == 0:
        mask = saliency_map <= thres
    else:
        mask = saliency_map < thres
    # Apply the gaussian noise to the whole image
    noise_img = perturbation(scale, image)
    # Now apply the mask: replace the original image pixels with noisy pixels where mask is True
    image[mask] = noise_img[mask]
    return image


def perturb_random_n_attention_regions(
    saliency_map, image, perturbation, n=30, scale=0
):
    """
    Perturbs n% of the regions of an image where the saliency map has an value greater than threshold
    """
    if n < 0 or n > 100:
        raise ValueError("The n value needs to be in the range of [0, 100]")
    # Create a binary mask from the array
    mask = np.random.choice([True, False], size=saliency_map.shape, p=[n / 100, 1 - n / 100])
    # Apply the gaussian noise to the whole image
    noise_img = perturbation(scale, image)
    # Now apply the mask: replace the original image pixels with noisy pixels where mask is True
    image[mask] = noise_img[mask]
    return image


def effects_attention_regions(
    saliency_map,scale, image,type
):
    mask = saliency_map > np.percentile(saliency_map, 90)
    coordinates = np.argwhere(mask)
    selected_coords = coordinates[np.random.choice(coordinates.shape[0], scale+1, replace=False)]
    selected_coords_tuples = [tuple(row) for row in selected_coords]
    selected_coords_tuples=clamp_values(selected_coords_tuples, 5, image.shape[1]-5, 5, image.shape[0]-5)
    List_of_Drops, _,_  = generate_label(image.shape[1], image.shape[0], selected_coords_tuples,cfg)
    output_image = generateDrops(image, cfg, List_of_Drops)
    return output_image


def shift_color(image, source_color, target_color):
    # Check if the image has an alpha channel
    has_alpha = image.shape[2] == 4

    if has_alpha:
        # Split the image into BGR and alpha channels
        bgr, alpha = image[:, :, :3], image[:, :, 3]
    else:
        bgr = image

    # Convert source and target colors to numpy arrays
    source_color = np.array(source_color, dtype=np.int16)
    target_color = np.array(target_color, dtype=np.int16)

    # Calculate the difference
    color_diff = target_color - source_color

    # Apply the difference to each pixel in the BGR channels
    shifted_bgr = np.clip(bgr.astype(np.int16) + color_diff, 0, 255).astype(np.uint8)

    if has_alpha:
        # Recombine the shifted BGR channels with the untouched alpha channel
        shifted_image = cv2.merge((shifted_bgr, alpha))
    else:
        shifted_image = shifted_bgr

    return shifted_image


# ===============================================
#                LIDAR                           
# ===============================================

def _lidar_severity_value(scale: int, schedule: Tuple[float, ...]) -> float:
    if not 0 <= scale < len(schedule):
        raise ValueError("Scale must be within [0, 4].")
    return schedule[scale]


def _ensure_generator(rng: Optional[np.random.Generator]) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    return rng


def lidar_inject_ghost_points(
    scale: int,
    point_cloud: np.ndarray,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Adds ghost points to the LiDAR point cloud to simulate multi-path reflections.

    Parameters:
        - scale int: The severity of the perturbation on a scale from 0 to 4.
        - point_cloud (numpy array): Array shaped (N, C) representing LiDAR points.
        - bounds tuple: Optional explicit (min, max) bounds for ghost placement.
        - rng (numpy.random.Generator | None): Optional RNG for reproducibility.

    Returns: numpy array: Point cloud augmented with ghost points.
    """
    pc = np.asarray(point_cloud)
    if pc.size == 0:
        return pc.copy()

    generator = _ensure_generator(rng)
    ghost_ratio = _lidar_severity_value(scale, (0.02, 0.05, 0.1, 0.18, 0.26))
    num_ghosts = max(1, int(np.ceil(pc.shape[0] * ghost_ratio)))

    position_dims = min(3, pc.shape[1])

    if bounds is None:
        mins = pc[:, :position_dims].min(axis=0)
        maxs = pc[:, :position_dims].max(axis=0)
    else:
        mins = np.asarray(bounds[0])[:position_dims]
        maxs = np.asarray(bounds[1])[:position_dims]

    ghost_positions = generator.uniform(mins, maxs, size=(num_ghosts, position_dims))

    residual_dims = pc.shape[1] - position_dims
    if residual_dims > 0:
        max_intensity = _lidar_severity_value(scale, (0.3, 0.35, 0.4, 0.45, 0.5))
        ghost_rest = generator.uniform(0.0, max_intensity, size=(num_ghosts, residual_dims))
        ghosts = np.concatenate((ghost_positions, ghost_rest), axis=1)
    else:
        ghosts = ghost_positions

    return np.vstack((pc, ghosts.astype(pc.dtype))).copy()


def lidar_reduce_reflectivity(scale: int, point_cloud: np.ndarray) -> np.ndarray:
    """
    Lowers the intensity channels to mimic low-reflectivity surfaces.

    Parameters:
        - scale int: The severity of the perturbation on a scale from 0 to 4.
        - point_cloud (numpy array): Array shaped (N, C) representing LiDAR points.

    Returns: numpy array: Point cloud with reduced intensity values.
    """
    pc = np.asarray(point_cloud)
    if pc.size == 0:
        return pc.copy()

    if pc.shape[1] <= 3:
        return pc.copy()

    atten_factor = _lidar_severity_value(scale, (0.85, 0.7, 0.5, 0.35, 0.2))
    result = pc.copy()

    intensities = result[:, 3:]
    max_intensity = np.max(intensities) if intensities.size > 0 else 1.0

    result[:, 3:] = np.clip(intensities * atten_factor, 0.0, max_intensity)

    return result


def lidar_simulate_adverse_weather(
    scale: int,
    point_cloud: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Applies combined dropout, positional jitter, and reflectivity damping to mimic adverse weather.

    Parameters:
        - scale int: The severity of the perturbation on a scale from 0 to 4.
        - point_cloud (numpy array): Array shaped (N, C) representing LiDAR points.
        - rng (numpy.random.Generator | None): Optional RNG for reproducibility.

    Returns: numpy array: Weather-perturbed point cloud.
    """
    pc = np.asarray(point_cloud)
    if pc.size == 0:
        return pc.copy()

    generator = _ensure_generator(rng)

    drop_rate = _lidar_severity_value(scale, (0.08, 0.16, 0.28, 0.42, 0.6))
    positional_noise = _lidar_severity_value(scale, (0.01, 0.02, 0.05, 0.08, 0.12))
    reflectivity_factor = _lidar_severity_value(scale, (0.8, 0.65, 0.45, 0.3, 0.15))

    keep_mask = generator.random(pc.shape[0]) > drop_rate
    if not keep_mask.any():
        keep_mask[generator.integers(0, pc.shape[0])] = True

    weather_pc = pc[keep_mask].copy()

    position_dims = min(3, weather_pc.shape[1])
    weather_pc[:, :position_dims] += generator.normal(
        loc=0.0, scale=positional_noise, size=(weather_pc.shape[0], position_dims)
    )

    if weather_pc.shape[1] > position_dims:
        intensities = weather_pc[:, position_dims:]
        max_intensity = np.max(intensities) if intensities.size > 0 else 1.0
        noise = generator.normal(
            loc=0.0,
            scale=positional_noise / 2,
            size=intensities.shape,
        )
        weather_pc[:, position_dims:] = np.clip(
            intensities * reflectivity_factor + noise,
            0.0,
            max_intensity,
        )

    return weather_pc

# ===============================================
#        LIDAR (MultiCorrupt-style)      
# #      Combined perturbations for multi-corruption scenarios
#       https://github.com/ika-rwth-aachen/MultiCorrupt/tree/main            
# ===============================================
_FOG_LOOKUP_DIR = os.path.join(os.path.dirname(__file__), "utils", "fog_lookup_tables")
_SNOW_NPY_DIR = os.path.join(os.path.dirname(__file__), "utils", "npy")
_SNOW_LABEL_YAML = os.path.join(os.path.dirname(__file__), "utils", "nuscenes_snow.yaml")
_SNOW_LASER_YAML = os.path.join(os.path.dirname(__file__), "utils", "snow_laser.yaml")

PI = np.pi

seed = 1000
np.random.seed(seed)
RNG = np.random.default_rng(seed)

""" motion blur """
def pts_motion(severity, points):
    # Mapping 5 severity levels 
    # Level 1: 0.06, Level 3: 0.1, Level 5: 0.13
    s_vals = [0.06,  0.1, 0.13]
    safe_severity = min(severity, 2)
    s = s_vals[safe_severity]
    
    trans_std = [s, s, s]
    noise_translate = np.array([
    np.random.normal(0, trans_std[0], 1),
    np.random.normal(0, trans_std[1], 1),
    np.random.normal(0, trans_std[2], 1),
    ]).T
    
    points[:, 0:3] += noise_translate
    num_points = points.shape[0]
    jitters_x = np.clip(np.random.normal(loc=0.0, scale=trans_std[0]*0.15, size=num_points), -3 * trans_std[0], 3 * trans_std[0])
    jitters_y = np.clip(np.random.normal(loc=0.0, scale=trans_std[1]*0.2, size=num_points), -3 * trans_std[1], 3 * trans_std[1])
    jitters_z = np.clip(np.random.normal(loc=0.0, scale=trans_std[2]*0.12, size=num_points), -3 * trans_std[2], 3 * trans_std[2])

    points[:, 0] += jitters_x
    points[:, 1] += jitters_y
    points[:, 2] += jitters_z
    return points


""" spatial misalignment """
def transform_points(severity, points):
    """
    Rotate and translate a set of points.
    
    Parameters:
    severity (int): Severity level (1-3)
    points (numpy.ndarray): A 2D array where each row represents a point (x, y, z, ...).
    
    Returns:
    numpy.ndarray: The transformed points.
    """
    # Mapping 5 severity levels. 
    # Format: (probability, degrees)
    s_vals = [(0.2, 1),  (0.4, 2),  (0.6, 3)]
    safe_severity = min(severity, 2)
    s = s_vals[safe_severity]
    
    
    # Convert the angle from degrees to radians
    rand_num = np.random.rand()
    
    # Check if the random number is less than the given probability
    if rand_num < s[0]:
        
        # Convert the angle from degrees to radians
        theta = np.radians(s[1])

        # Define the rotation matrix for rotation around the x-axis
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta), np.sin(theta)],
            [0, -np.sin(theta), np.cos(theta)]
        ])

        # Define the rotation matrix for rotation around the y-axis
        rotation_matrix_y = np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)]
        ])

        # Define the rotation matrix for rotation around the z-axis
        rotation_matrix_z = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        # Combine the three rotations
        rotation_matrix = rotation_matrix_x @ rotation_matrix_y @ rotation_matrix_z

               
        # Extract the x, y, z coordinates of the points
        xyz_points = points[:, :3]
       
        # Apply the rotation matrix to the x, y, z coordinates
        rotated_xyz = np.dot(xyz_points, rotation_matrix.T)
       
        # Define the translation vector (2 units along the x-axis)
        translation_vector = np.array([2, 0, 0])
       
        # Apply the translation to the rotated x, y, z coordinates
        translated_xyz = rotated_xyz + translation_vector
       
        # Concatenate the translated x, y, z coordinates with the other properties of the points
        transformed_points = np.hstack((translated_xyz, points[:, 3:]))
       
        return transformed_points
    
    else:
        # If the random number is not less than the given probability, return the original points
        return points


""" beam reduce """
def reduce_LiDAR_beamsV2(severity, pts):
    # Mapping 3 levels: 1=Less severe (16 beams), 3=Most severe (4 beams)
    
    s_vals = [16, 8, 4]
    safe_severity = min(severity, 2)
    s = s_vals[safe_severity]
    
    allowed_beams = []
    
    if s == 16:
        #  level 1 behavior
        allowed_beams = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    elif s == 8:
        #  level 2 behavior
        allowed_beams = [1, 5, 9, 13, 17, 21, 25, 29]
    elif s == 4:
        #  level 3 behavior
        allowed_beams = [1, 9, 17, 25]
    
    mask = np.full(pts.shape[0], False)
    
    for beam in allowed_beams:
        beam_mask = pts[:, 4] == beam
        mask = np.logical_or(beam_mask, mask)
    return pts[mask, :]


""" points missing """
def pointsreducing(severity, pts):
    """
    Simulates missing lidar points based on a given severity level.

    Args:
    severity: An integer between 1 and 3.
    pts: A numpy array of lidar points.

    Returns:
    A numpy array of lidar points with missing points.
    """
    
    s_vals = [70, 80, 90]
    safe_severity = min(severity, 2)
    s = s_vals[safe_severity]

    size = pts.shape[0]
    nr_of_samps = int(round(size * ((100 - s) / 100)))  # Calculate number of points to keep

    permutations = np.random.permutation(size)  # Generate random permutation of indices
    ind = permutations[:nr_of_samps]  # Select indices for points to keep

    pts = pts[ind]  # Extract points based on selected indices

    return pts


'''  fog function'''
# Use the config variable instead of relative path
INTEGRAL_PATH = _FOG_LOOKUP_DIR

class ParameterSet:

    def __init__(self, **kwargs) -> None:

        self.n = 500
        self.n_min = 100
        self.n_max = 1000

        self.r_range = 100
        self.r_range_min = 50
        self.r_range_max = 250

        ##########################
        # soft target a.k.a. fog #
        ##########################

        # attenuation coefficient => amount of fog
        self.alpha = 0.06
        self.alpha_min = 0.003
        self.alpha_max = 0.5
        self.alpha_scale = 1000

        # meteorological optical range (in m)
        self.mor = np.log(20) / self.alpha

        # backscattering coefficient (in 1/sr) [sr = steradian]
        self.beta = 0.046 / self.mor
        self.beta_min = 0.023 / self.mor
        self.beta_max = 0.092 / self.mor
        self.beta_scale = 1000 * self.mor

        ##########
        # sensor #
        ##########

        # pulse peak power (in W)
        self.p_0 = 80
        self.p_0_min = 60
        self.p_0_max = 100

        # half-power pulse width (in s)
        self.tau_h = 5e-9
        self.tau_h_min = 3e-9
        self.tau_h_max = 1e-8
        self.tau_h_scale = 1e9

        # total pulse energy (in J)
        self.e_p = self.p_0 * self.tau_h  # equation (7) in [1]

        # aperture area of the receiver (in in m²)
        self.a_r = 0.25
        self.a_r_min = 0.01
        self.a_r_max = 0.1
        self.a_r_scale = 1000

        # loss of the receiver's optics
        self.l_r = 0.05
        self.l_r_min = 0.01
        self.l_r_max = 0.10
        self.l_r_scale = 100

        self.c_a = c * self.l_r * self.a_r / 2

        self.linear_xsi = True

        self.D = 0.0                              # in m              (displacement of transmitter and receiver)
        self.ROH_T = 0.01                         # in m              (radius of the transmitter aperture)
        self.ROH_R = 0.01                         # in m              (radius of the receiver aperture)
        self.GAMMA_T_DEG = 0.12                      # in deg            (opening angle of the transmitter's FOV)
        self.GAMMA_R_DEG = 0.20                    # in deg            (opening angle of the receiver's FOV)
        self.GAMMA_T = math.radians(self.GAMMA_T_DEG)
        self.GAMMA_R = math.radians(self.GAMMA_R_DEG)


        # range at which receiver FOV starts to cover transmitted beam (in m)
        self.r_1 = 0.9
        self.r_1_min = 0
        self.r_1_max = 10
        self.r_1_scale = 10

        # range at which receiver FOV fully covers transmitted beam (in m)
        self.r_2 = 1.0
        self.r_2_min = 0
        self.r_2_max = 10
        self.r_2_scale = 10

        ###############
        # hard target #
        ###############

        # distance to hard target (in m)
        self.r_0 = 30
        self.r_0_min = 1
        self.r_0_max = 200

        # reflectivity of the hard target [0.07, 0.2, > 4 => low, normal, high]
        self.gamma = 0.000001
        self.gamma_min = 0.0000001
        self.gamma_max = 0.00001
        self.gamma_scale = 10000000

        # differential reflectivity of the target
        self.beta_0 = self.gamma / np.pi

        self.__dict__.update(kwargs)


def get_integral_dict(p: ParameterSet) -> Dict:
    alpha =p.alpha
    beta=p.beta
    # Using the global INTEGRAL_PATH derived from top config
    filename = Path(INTEGRAL_PATH) / f'integral_0m_to_200m_stepsize_0.1m_alpha_{alpha}_beta_{beta}.pickle'

    with open(filename, 'rb') as handle:
        integral_dict = pickle.load(handle)

    return integral_dict


def P_R_fog_hard(p: ParameterSet, pc: np.ndarray) -> np.ndarray:
    r_0 = np.linalg.norm(pc[:, 0:3], axis=1)
    pc[:, 3] = np.round(np.exp(-2 * p.alpha * r_0) * pc[:, 3])
    return pc


def P_R_fog_soft(p: ParameterSet, pc: np.ndarray, original_intesity: np.ndarray,  noise: int, gain: bool = False,
                 noise_variant: str = 'v1') -> Tuple[np.ndarray, np.ndarray, Dict]:

    augmented_pc = np.zeros(pc.shape)
    fog_mask = np.zeros(len(pc), dtype=bool)

    r_zeros = np.linalg.norm(pc[:, 0:3], axis=1)

    min_fog_response = np.inf
    max_fog_response = 0
    num_fog_responses = 0

    integral_dict = get_integral_dict(p)

    r_noise = RNG.integers(low=1, high=20, size=1)[0]
    r_noise = 10
    for i, r_0 in enumerate(r_zeros):

        # load integral values from precomputed dict
        key = float(str(round(r_0, 1)))
        # limit key to a maximum of 50 m
        fog_distance, fog_response = integral_dict[min(key, 50)]
        fog_response = fog_response * original_intesity[i] * (r_0 ** 2) * p.beta / p.beta_0

        # limit to 255
        # fog_response = min(fog_response, 255)

        if fog_response > pc[i, 3]:

            fog_mask[i] = 1

            num_fog_responses += 1

            scaling_factor = fog_distance / r_0

            augmented_pc[i, 0] = pc[i, 0] * scaling_factor
            augmented_pc[i, 1] = pc[i, 1] * scaling_factor
            augmented_pc[i, 2] = pc[i, 2] * scaling_factor
            augmented_pc[i, 3] = fog_response

            # keep 5th feature if it exists
            if pc.shape[1] > 4:
                augmented_pc[i, 4] = pc[i, 4]

            if noise > 0:

                if noise_variant == 'v1':

                    # add uniform noise based on initial distance
                    distance_noise = RNG.uniform(low=r_0 - noise, high=r_0 + noise, size=1)[0]
                    noise_factor = r_0 / distance_noise

                elif noise_variant == 'v2':

                    # add noise in the power domain
                    power = RNG.uniform(low=-1, high=1, size=1)[0]
                    noise_factor = max(1.0, noise/5) ** power       # noise=10 => noise_factor ranges from 1/2 to 2

                elif noise_variant == 'v3':

                    # add noise in the power domain
                    power = RNG.uniform(low=-0.5, high=1, size=1)[0]
                    noise_factor = max(1.0, noise*4/10) ** power    # noise=10 => ranges from 1/2 to 4

                elif noise_variant == 'v4':

                    additive = r_noise * RNG.beta(a=2, b=20, size=1)[0]
                    new_dist = fog_distance + additive
                    noise_factor = new_dist / fog_distance

                else:

                    raise NotImplementedError(f"noise variant '{noise_variant}' is not implemented (yet)")

                augmented_pc[i, 0] = augmented_pc[i, 0] * noise_factor
                augmented_pc[i, 1] = augmented_pc[i, 1] * noise_factor
                augmented_pc[i, 2] = augmented_pc[i, 2] * noise_factor

            if fog_response > max_fog_response:
                max_fog_response = fog_response

            if fog_response < min_fog_response:
                min_fog_response = fog_response

        else:
            augmented_pc[i] = pc[i]

    if gain:
        max_intensity = np.ceil(max(augmented_pc[:, 3]))
        gain_factor = 255 / max_intensity
        augmented_pc[:, 3] *= gain_factor

    simulated_fog_pc = None
    num_fog = 0
    if num_fog_responses > 0:
        fog_points = augmented_pc[fog_mask]
        simulated_fog_pc = fog_points
        num_fog = len(fog_points)


    info_dict = {'min_fog_response': min_fog_response,
                 'max_fog_response': max_fog_response,
                 'num_fog_responses': num_fog_responses,}

    return augmented_pc, simulated_fog_pc,  num_fog, info_dict


def simulate_fog(severity, pc: np.ndarray, noise: int=0, gain: bool = False, noise_variant: str = 'v1',
                 hard: bool = True, soft: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    
    # 3 levels of severity mapping
    s_vals = [
        (0.02, 0.008), 
        (0.03, 0.008),
        (0.06, 0.05)   
    ]
    safe_severity = min(severity, 2)
    s = s_vals[safe_severity]
    
    p = ParameterSet(alpha=s[0],beta=s[1]) 
    augmented_pc = copy.deepcopy(pc)
    original_intensity = copy.deepcopy(pc[:, 3])

    info_dict = None
    simulated_fog_pc = None

    if hard:
        augmented_pc = P_R_fog_hard(p, augmented_pc)
    if soft:
        augmented_pc, simulated_fog_pc,  num_fog, info_dict = P_R_fog_soft(p, augmented_pc, original_intensity,  noise, gain,
                                                                             noise_variant)

    return augmented_pc, simulated_fog_pc, num_fog, info_dict


'''  snow function'''
def estimate_ground_plane(point_cloud):
    min_ground = -1.3 
    max_ground = -2.1
    # Assuming point_cloud is a numpy array with shape (N, 3) representing (x, y, z) coordinates

    valid_loc = (point_cloud[:, 2] < min_ground) & (point_cloud[:, 2] > max_ground)
    point_cloud = point_cloud[valid_loc]

    if len(point_cloud) < 25:
        w = [0, 0, -1]
        h = -1.85
        print("Not enought points. Use default flat world assumption!!")
    else:
        # Create RANSACRegressor model
        model = make_pipeline(StandardScaler(), RANSACRegressor())

        # Fit the model to the data
        model.fit(point_cloud[:, :2], point_cloud[:, 2])

        # Extract the estimated coefficients (slope and intercept)
        w = np.zeros(3)
        w[0] = model.named_steps['ransacregressor'].estimator_.coef_[0]
        w[1] = model.named_steps['ransacregressor'].estimator_.coef_[1]
        w[2] = -1.0
        w = w / np.linalg.norm(w)
        h = model.named_steps['ransacregressor'].estimator_.intercept_
    
    if h < max_ground or h > min_ground:
        w = [0, 0, -1]
        h = -1.85
        print("Bad RANSAC Parameters. Use default flat world assumption!")
       
    return w, h

def calculate_plane(pointcloud, standart_height=-1.55):
    """
    caluclates plane from loaded pointcloud
    returns the plane normal w and lidar height h.
    :param pointcloud: binary with x,y,z, coordinates
    :return:           w, h
    """

    # Filter points which are close to ground based on mounting position
    valid_loc = (pointcloud[:, 2] < -1.55) & \
                (pointcloud[:, 2] > -1.86 - 0.01 * pointcloud[:, 0]) & \
                (pointcloud[:, 0] > 10) & \
                (pointcloud[:, 0] < 70) & \
                (pointcloud[:, 1] > -3) & \
                (pointcloud[:, 1] < 3)
    pc_rect = pointcloud[valid_loc]

    if pc_rect.shape[0] <= pc_rect.shape[1]:
        w = [0, 0, -1]
        # Standard height from vehicle mounting position
        h = standart_height
    else:
        try:
            reg = RANSACRegressor(loss='squared_loss', max_trials=1000).fit(pc_rect[:, [0, 1]], pc_rect[:, 2])
            w = np.zeros(3)
            w[0] = reg.estimator_.coef_[0]
            w[1] = reg.estimator_.coef_[1]
            w[2] = -1.0
            h = reg.estimator_.intercept_
            w = w / np.linalg.norm(w)

        except:
            # If error occurs fall back to flat earth assumption
            print('Was not able to estimate a ground plane. Using default flat earth assumption')
            w = [0.0, 0.0, -1.0]
            # Standard height from vehicle mounting position
            h = standart_height
    
    # if estimated h is not reasonable fall back to flat earth assumption
    if abs(h - standart_height) > 1.5:
        print('Estimated h is not reasonable. Using default flat earth assumption')
        w = [0.0, 0.0, -1.0]
        h = standart_height
    
    return w, h

# Load labels from config file
with open(_SNOW_LABEL_YAML, 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)
learning_map = semkittiyaml['learning_map']

def compute_occupancy(snowfall_rate: float, terminal_velocity: float, snow_density: float=0.1) -> float:
    """
    :param snowfall_rate:           Typically ranges from 0 to 2.5                          [mm/h]
    :param terminal_velocity:       Varies from 0.2 to 2                                    [m/s]
    :param snow_density:            Varies from 0.01 to 0.2 depending on snow wetness       [g/cm³]
    :return:                        Occupancy ratio.
    """
    water_density = 1.0

    return (water_density * snowfall_rate) / ((3.6 * 10 ** 6) * (snow_density * terminal_velocity))

def snowfall_rate_to_rainfall_rate(snowfall_rate: float, terminal_velocity: float,
                                   snowflake_density: float = 0.1, snowflake_diameter: float = 0.003) -> float:
    """
    :param snowfall_rate:       Typically ranges from 0 to 2.5                          [mm/h]
    :param terminal_velocity:   Varies from 0.2 to 2                                    [m/s]
    :param snowflake_density:   Varies from 0.01 to 0.2 depending on snow wetness       [g/cm^3]
    :param snowflake_diameter:  Varies from 1 to 10                                     [m]

    :return:
    rainfall_rate:              Varies from 0.5 (slight rain) to 50 (violent shower)    [mm/h]
    """

    rainfall_rate = np.sqrt((snowfall_rate / (487 * snowflake_density * snowflake_diameter * terminal_velocity))**3)

    return rainfall_rate

def ransac_polyfit(x, y, order=3, n=15, k=100, t=0.1, d=15, f=0.8):
    # Applied https://en.wikipedia.org/wiki/Random_sample_consensus
    # Taken from https://gist.github.com/geohot/9743ad59598daf61155bf0d43a10838c
    # n – minimum number of data points required to fit the model
    # k – maximum number of iterations allowed in the algorithm
    # t – threshold value to determine when a data point fits a model
    # d – number of close data points required to assert that a model fits well to data
    # f – fraction of close data points required

    bestfit = np.polyfit(x, y, order)
    besterr = np.sum(np.abs(np.polyval(bestfit, x) - y))
    for kk in range(k):
        maybeinliers = np.random.randint(len(x), size=n)
        maybemodel = np.polyfit(x[maybeinliers], y[maybeinliers], order)
        alsoinliers = np.abs(np.polyval(maybemodel, x) - y) < t
        if sum(alsoinliers) > d and sum(alsoinliers) > len(x) * f:
            bettermodel = np.polyfit(x[alsoinliers], y[alsoinliers], order)
            thiserr = np.sum(np.abs(np.polyval(bettermodel, x[alsoinliers]) - y[alsoinliers]))
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
    return bestfit

def estimate_laser_parameters(pointcloud_planes, calculated_indicent_angle, power_factor=15, noise_floor=0.7,
                              debug=True, estimation_method='linear'):
    """
    :param pointcloud_planes: Get all points which correspond to the ground
    :param calculated_indicent_angle: The calculated incident angle for each individual point
    :param power_factor: Determines, how much more Power is available compared to a groundplane reflection.
    :param noise_floor: What are the minimum intensities that could be registered
    :param debug: Show additional Method
    :param estimation_method: Method to fit to outputted laser power.
    :return: Fits the laser outputted power level and noiselevel for each point based on the assumed ground floor reflectivities.
    """
    # normalize intensitities
    normalized_intensitites = pointcloud_planes[:, 3] / np.cos(calculated_indicent_angle)
    distance = np.linalg.norm(pointcloud_planes[:, :3], axis=1)

    # linear model
    p = None
    stat_values = None
    if len(normalized_intensitites) < 3:
        return None, None, None, None
    if estimation_method == 'linear':
        reg = linregress(distance, normalized_intensitites)
        w = reg[0]
        h = reg[1]
        p = [w, h]
        stat_values = reg[2:]
        relative_output_intensity = power_factor * (p[0] * distance + p[1])

    elif estimation_method == 'poly':
        # polynomial 2degre fit
        p = np.polyfit(np.linalg.norm(pointcloud_planes[:, :3], axis=1),
                       normalized_intensitites, 2)
        relative_output_intensity = power_factor * (
                p[0] * distance ** 2 + p[1] * distance + p[2])


    # estimate minimum noise level therefore get minimum reflected intensitites
    hist, xedges, yedges = np.histogram2d(distance, normalized_intensitites, bins=(50, 2555),
                                          range=((10, 70), (5, np.abs(np.max(normalized_intensitites)))))
    idx = np.where(hist == 0)
    hist[idx] = len(pointcloud_planes)
    ymins = np.argpartition(hist, 2, axis=1)[:, 0]
    min_vals = yedges[ymins]
    idx = np.where(min_vals > 5)
    min_vals = min_vals[idx]
    idx1 = [i + 1 for i in idx]
    x = (xedges[idx] + xedges[idx1]) / 2

    if estimation_method == 'poly':
        pmin = ransac_polyfit(x, min_vals, order=2)
        adaptive_noise_threshold = noise_floor * (
                pmin[0] * distance ** 2 + pmin[1] * distance + pmin[2])
    elif estimation_method == 'linear':
        if len(min_vals) > 3:
            pmin = linregress(x, min_vals)
        else:
            pmin = p
        adaptive_noise_threshold = noise_floor * (
                pmin[0] * distance + pmin[1])
    # Guess that noise level should be half the road relfection


    return relative_output_intensity, adaptive_noise_threshold, p, stat_values

def process_single_channel(root_path: str, particle_file_prefix: str, orig_pc: np.ndarray, beam_divergence: float,
                           order: List[int], channel_infos: List, channel: int) -> Tuple:
    """
    :param root_path:               Needed for training on GPU cluster.
    :param particle_file_prefix:    Path to file where sampled particles are stored (x, y, r).
    :param orig_pc:                 N-by-5 array containing original pointcloud (x, y, z, intensity, channel).
    :param beam_divergence:         Equivalent to the total beam opening angle (in degree).
    :param order:                   Order of the particle disks.
    :param channel_infos            List of Dicts containing sensor calibration info.

    :param channel:                 Number of the LiDAR channel [0, 31].

    :return:                        Tuple of
                                    - intensity_diff_sum,
                                    - idx,
                                    - the augmented points of the current LiDAR channel.
    """
    
    intensity_diff_sum = 0

    index = order[channel]

    min_intensity = 0  #channel_infos[channel].get('min_intensity', 0)  # not all channels contain this info

    focal_distance = channel_infos[channel]['focal_distance'] * 100
    focal_slope = channel_infos[channel]['focal_slope']
    focal_offset = (1 - focal_distance / 13100) ** 2                # from velodyne manual

    particle_file = f'{particle_file_prefix}_{index + 1}.npy'

    channel_mask = orig_pc[:, 4] == channel
    idx = np.where(channel_mask == True)[0]

    pc = orig_pc[channel_mask]
    # print(pc.shape)

    N = pc.shape[0]

    x, y, z, intensity = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3]
    distance = np.linalg.norm([x, y, z], axis=0)

    center_angles = np.arctan2(y, x)  # in range [-PI, PI]
    center_angles[center_angles < 0] = center_angles[center_angles < 0] + 2 * PI  # in range [0, 2*PI]

    beam_angles = -np.ones((N, 2))

    beam_angles[:, 0] = center_angles - np.radians(beam_divergence / 2)  # could lead to negative values
    beam_angles[:, 1] = center_angles + np.radians(beam_divergence / 2)  # could lead to values above 2*PI

    # put beam_angles back in range [0, 2*PI]
    beam_angles[beam_angles < 0] = beam_angles[beam_angles < 0] + 2 * PI
    beam_angles[beam_angles > 2 * PI] = beam_angles[beam_angles > 2 * PI] - 2 * PI

    occlusion_list = get_occlusions(beam_angles=beam_angles, ranges_orig=distance, beam_divergence=beam_divergence,
                                    root_path=root_path, particle_file=particle_file)

    lidar_range = 100                       # in meter, real sensor is 200 but 100 is sufficient
    intervals_per_meter = 10                # => 10cm discretization
    beta_0 = 1 * 10 ** -6 / PI
    tau_h = 5e-9                           #  value 5ns taken from VLP-32C specsheet

    M = lidar_range * intervals_per_meter

    M_extended = int(np.ceil(M + c * tau_h * intervals_per_meter))
    lidar_range_extended = lidar_range + c * tau_h

    R = np.round(np.linspace(0, lidar_range_extended, M_extended), len(str(intervals_per_meter)))

    for j, beam_dict in enumerate(occlusion_list):

        d_orig = distance[j]
        i_orig = intensity[j]

     
        max_intensity = 255

        i_adjusted = i_orig - 255 * focal_slope * np.abs(focal_offset - (1 - d_orig/120)**2)
        i_adjusted = np.clip(i_adjusted, 0, max_intensity)      # to make sure we don't get negative values

        CA_P0 = i_adjusted * d_orig ** 2 / beta_0

        if len(beam_dict.keys()) > 1:                               # otherwise there is no snowflake in the current beam

            i = np.zeros(M_extended)

            for key, tuple_value in beam_dict.items():

                if key != -1:                                       # if snowflake
                    i_orig = 0.9 * max_intensity                # set i to 90% reflectivity
                    CA_P0 = i_orig / beta_0                     # and do NOT normalize with original range

                r_j, ratio = tuple_value

                start_index = int(np.ceil(r_j * intervals_per_meter))
                end_index = int(np.floor((r_j + c * tau_h) * intervals_per_meter) + 1)

                for k in range(start_index, end_index):
                    i[k] += received_power(CA_P0, beta_0, ratio, R[k], r_j, tau_h)

            max_index = np.argmax(i)
            i_max = i[max_index]
            d_max = (max_index / intervals_per_meter) - (c * tau_h / 2)

            i_max += max_intensity * focal_slope * np.abs(focal_offset - (1 - d_max/120)**2)
            i_max = np.clip(i_max, min_intensity, max_intensity)

            if abs(d_max - d_orig) < 2 * (1 / intervals_per_meter):  # only alter intensity

                pc[j, 4] = 1

                new_i = int(i_max)

                intensity_diff_sum += i_orig - new_i

            else:  # replace point of hard target with snowflake

                pc[j, 4] = 2

                d_scaling_factor = d_max / d_orig

                pc[j, 0] = pc[j, 0] * d_scaling_factor
                pc[j, 1] = pc[j, 1] * d_scaling_factor
                pc[j, 2] = pc[j, 2] * d_scaling_factor

                new_i = int(i_max)

            assert new_i >= 0, f'new intensity is negative ({new_i})'

            clipped_i = np.clip(new_i, min_intensity, max_intensity)

            pc[j, 3] = clipped_i

        else:
            pc[j, 4] = 0

    return intensity_diff_sum, idx, pc


def binary_angle_search(angles: List[float], low: int, high: int, angle: float) -> int:
    """
    Adapted from https://www.geeksforgeeks.org/python-program-for-binary-search

    :param angles:                  List of individual endpoint angles.
    :param low:                     Start index.
    :param high:                    End index.
    :param angle:                   Query angle.

    :return:                        Index of angle if present in list of angles, else -1
    """

    # Check base case
    if high >= low:

        mid = (high + low) // 2

        # If angle is present at the middle itself
        if angles[mid] == angle:
            return mid

        # If angle is smaller than mid, then it can only be present in left sublist
        elif angles[mid] > angle:
            return binary_angle_search(angles, low, mid - 1, angle)

        # Else the angle can only be present in right sublist
        else:
            return binary_angle_search(angles, mid + 1, high, angle)

    else:
        # Angle is not present in the list
        return -1


def compute_occlusion_dict(beam_angles: Tuple[float, float], intervals: np.ndarray,
                           current_range: float, beam_divergence: float) -> Dict:
    """
    :param beam_angles:         Tuple of angles (left, right).
    :param intervals:           N-by-3 array of particle tangent angles and particle distance from origin.
    :param current_range:       Range to the original hard target.
    :param beam_divergence:     Equivalent to the total beam opening angle (in degree).

    :return:
    occlusion_dict:             Dict containing a tuple of the distance and the occluded angle by respective particle.
                                e.g.
                                0: (distance to particle, occlusion ratio [occluded angle / total angle])
                                1: (distance to particle, occlusion ratio [occluded angle / total angle])
                                3: (distance to particle, occlusion ratio [occluded angle / total angle])
                                7: (distance to particle, occlusion ratio [occluded angle / total angle])
                                ...
                                -1: (distance to original target, unocclusion ratio [unoccluded angle / total angle])

                                all (un)occlusion ratios always sum up to the value 1
    """

    try:
        N = intervals.shape[0]
    except IndexError:
        N = 1

    right_angle, left_angle = beam_angles

    # Make everything properly sorted in the corner case of phase discontinuity.
    if right_angle > left_angle:
        right_angle = right_angle - 2*PI
        right_left_order_violated = intervals[:, 0] > intervals[:, 1]
        intervals[right_left_order_violated, 0] = intervals[right_left_order_violated, 0] - 2*PI

    endpoints = sorted(set([right_angle] + list(itertools.chain(*intervals[:, :2])) + [left_angle]))
    diffs = np.diff(endpoints)
    n_intervals = diffs.shape[0]

    assignment = -np.ones(n_intervals)

    occlusion_dict = {}

    for j in range(N):

        a1, a2, distance = intervals[j]

        i1 = binary_angle_search(endpoints, 0, len(endpoints), a1)
        i2 = binary_angle_search(endpoints, 0, len(endpoints), a2)

        assignment_made = False

        for k in range(i1, i2):

            if assignment[k] == -1:
                assignment[k] = j
                assignment_made = True

        if assignment_made:
            ratio = diffs[assignment == j].sum() / np.radians(beam_divergence)
            occlusion_dict[j] = (distance, np.clip(ratio, 0, 1))

    ratio = diffs[assignment == -1].sum() / np.radians(beam_divergence)
    occlusion_dict[-1] = (current_range, np.clip(ratio, 0, 1))

    return occlusion_dict

def tangent_angles_to_interval_angles(angles: np.ndarray, right_angle: float, left_angle: float,
                                      right_angle_hit: np.ndarray, left_angle_hit: np.ndarray) -> np.ndarray:
    """
    :param angles:              N-by-2 array containing tangent angles.
    :param right_angle:         Right beam angle.
    :param left_angle:          Left beam angle.
    :param right_angle_hit:     Flag if right beam angle has been exceeded.
    :param left_angle_hit:      Flag if left beam angle has been exceeded.

    :return:                    N-by-2 array of corrected tangent angles that do not exceed beam limits.
    """

    angles[right_angle_hit, 0] = right_angle
    angles[left_angle_hit, 1] = left_angle

    return angles

def do_angles_intersect_particles(angles: np.ndarray, particle_centers: np.ndarray) -> np.ndarray:
    """
    Assumption: either the ray that corresponds to an angle or its opposite ray intersects with all particles.

    :param angles:              (M,) array of angles in the range [0, 2*PI).
    :param particle_centers:    (N, 2) array of particle centers (abscissa, ordinate).

    :return:
    """
    try:
        M = angles.shape[0]
    except IndexError:
        M = 1

    try:
        N = particle_centers.shape[0]
    except IndexError:
        N = 1

    x, y = particle_centers[:, 0], particle_centers[:, 1]

    angle_to_centers = np.arctan2(y, x)
    angle_to_centers[angle_to_centers < 0] = angle_to_centers[angle_to_centers < 0] + 2*PI                      # (N, 1)

    angle_differences = np.tile(angles, (1, N)) - np.tile(angle_to_centers.T, (M, 1))                           # (M, N)

    answer = np.logical_or.reduce((np.abs(angle_differences) < PI/2,
                                   np.abs(angle_differences - 2*PI) < PI/2,
                                   np.abs(angle_differences + 2*PI) < PI/2))                                    # (M, N)

    return answer

def angles_to_lines(angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param angles:              M-by-2 array of angles, where for each row, the value in the first column
                                is lower than the value in the second column.
    :return:
    a_s:                        N-by-2 array holding the $a$ coefficients of the tangent rays of particles passing from
                                the origin. Each row corresponds to a_s particle.
    b_s:                        N-by-2 array holding the $b$ coefficients of the tangent rays of particles passing from
                                the origin. Each row corresponds to a_s particle.
    """

    tan_directions = np.tan(angles)                                                                                 # (M, 2)

    directions_vertical = np.logical_or(angles == PI/2, angles == 3 * PI/2)
    directions_not_vertical = np.logical_not(directions_vertical)

    a_s = np.zeros_like(angles)
    b_s = np.zeros_like(angles)

    a_s[np.where(directions_vertical)] = 1
    b_s[np.where(directions_vertical)] = 0

    a_s[np.where(directions_not_vertical)] = -tan_directions[np.where(directions_not_vertical)]
    b_s[np.where(directions_not_vertical)] = 1

    # a_s[np.abs(a_s) < EPSILON] = 0              # to prevent -0 value

    return a_s, b_s




def tangent_lines_to_tangent_angles(lines: Tuple[np.ndarray, np.ndarray], center_angles: np.ndarray) -> np.ndarray:
    """
    :param lines:               Tuple of two N-by-2 arrays holding the $a$ and $b$ coefficients of the tangents.
    :param center_angles:       N-by-1 array containing the angle to the particle center.

    :return:
    angles:                     N-by-2 array of tangent angles (right angle first, left angle second).
    """

    a_s, b_s = lines

    try:
        N = center_angles.shape[0]
    except IndexError:
        N = 1

    angles = -np.ones((N, 2))                                                                                       # (N, 2)

    ray_1_angles = np.arctan(-a_s/b_s)                                      # in range [-PI/2, PI/2]            # (N, 2)
    ray_2_angles = deepcopy(ray_1_angles) + PI                              # in range [PI/2, 3*PI/2]           # (N, 2)

    # correct value range
    ray_1_angles[ray_1_angles < 0] = ray_1_angles[ray_1_angles < 0] + 2*PI  # in range [0, 2*PI]                # (N, 2)
    ray_1_angles = np.abs(ray_1_angles)                                     # to prevent -0 value

    # catch special case if line is vertical
    ray_1_angles[b_s == 0] = PI/2
    ray_2_angles[b_s == 0] = 3*PI/2

    tangent_1_angles = np.column_stack((ray_1_angles[:, 0], ray_2_angles[:, 0]))                                   # (N, 2)
    tangent_2_angles = np.column_stack((ray_1_angles[:, 1], ray_2_angles[:, 1]))                                   # (N, 2)

    for i, tangent_angles in enumerate([tangent_1_angles, tangent_2_angles]):

        tangent_difference = tangent_angles - np.column_stack((center_angles, center_angles))                   # (N, 2)

        correct_ray = np.logical_or.reduce((np.abs(tangent_difference) < PI/2,
                                            np.abs(tangent_difference - 2*PI) < PI/2,
                                            np.abs(tangent_difference + 2*PI) < PI/2))                          # (N, 2)

        angles[:, i] = tangent_angles[np.where(correct_ray)]                                                    # (N, 2)

    angles.sort(axis=1)

    # swap order where discontinuity is crossed
    swap = angles[:, 1] - angles[:, 0] > PI
    angles[swap, 0], angles[swap, 1] = angles[swap, 1], angles[swap, 0]

    return angles

def tangents_from_origin(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param samples:             N-by-3 array of sampled particles as disks, where each row contains abscissa and
                                ordinate of disk center and disk radius (in meters).
    :return:
    a:                          N-by-2 array holding the $a$ coefficients of the tangent rays of particles passing from
                                the origin. Each row corresponds to a particle.
    b:                          N-by-2 array holding the $b$ coefficients of the tangent rays of particles passing from
                                the origin. Each row corresponds to a particle.
    """

    # Solve systems of equations that encode the following information:
    # 1) rays include origin,
    # 2) rays are tangent to the circles corresponding to the particles, i.e., they intersect with the circles at
    # exactly one point.

    x_s, y_s, r_s = samples[:, 0], samples[:, 1], samples[:, 2]

    try:
        N = samples.shape[0]
    except IndexError:
        N = 1

    discriminants = r_s * np.sqrt(x_s ** 2 + y_s ** 2 - r_s ** 2)

    case_1 = np.abs(x_s) - r_s == 0  # One of the two lines is vertical.
    case_2 = np.logical_not(case_1)  # Both lines are not vertical.

    a_1_case_1, b_1_case_1 = np.ones(N), np.zeros(N)
    a_2_case_1, b_2_case_1 = (y_s ** 2 - x_s ** 2) / (2 * x_s * y_s), - np.ones(N)

    a_1_case_2 = (-x_s * y_s + discriminants) / (r_s ** 2 - x_s ** 2)
    a_2_case_2 = (-x_s * y_s - discriminants) / (r_s ** 2 - x_s ** 2)
    b_1_case_2 = -np.ones(N)
    b_2_case_2 = -np.ones(N)

    # Compute the coefficients by distinguishing the two cases.
    a_1, a_2, b_1, b_2 = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)

    a_1[case_1] = a_1_case_1[case_1]
    a_2[case_1] = a_2_case_1[case_1]
    b_1[case_1] = b_1_case_1[case_1]
    b_2[case_1] = b_2_case_1[case_1]

    a_1[case_2] = a_1_case_2[case_2]
    a_2[case_2] = a_2_case_2[case_2]
    b_1[case_2] = b_1_case_2[case_2]
    b_2[case_2] = b_2_case_2[case_2]

    a = np.column_stack((a_1, a_2))
    b = np.column_stack((b_1, b_2))

    return a, b

def distances_of_points_to_lines(points: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    :param points:      N-by-2 array of points, where each row contains the coordinates (abscissa, ordinate) of a point
    :param a:           M-by-1 array of $a$ coefficients of lines
    :param b:           M-by-1 array of $b$ coefficients of lines
    :param c:           M-by-1 array of $c$ coefficients of lines
                        where ax + by = c

    :return:            N-by-M array containing distances of points to lines
    """

    try:
        N = points.shape[0]
    except IndexError:
        N = 1

    abscissa, ordinate = points[:, 0, np.newaxis], points[:, 1, np.newaxis]

    numerators = np.dot(abscissa, a.T) + np.dot(ordinate, b.T) + np.dot(np.ones((N, 1)), c.T)

    denominators = np.dot(np.ones((N, 1)), np.sqrt(a ** 2 + b ** 2).T)

    return np.abs(numerators / denominators)


def get_occlusions(beam_angles: np.ndarray, ranges_orig: np.ndarray, root_path: str, particle_file: str,
                   beam_divergence: float) -> List:
    """
    :param beam_angles:         M-by-2 array of beam endpoint angles, where for each row, the value in the first column
                                is lower than the value in the second column.
    :param ranges_orig:         M-by-1 array of original ranges corresponding to beams (in m).
    :param root_path:           Needed for training on GPU cluster.

    :param particle_file:       Path to N-by-3 array of all sampled particles as disks,
                                where each row contains abscissa and ordinate of the disk center and disk radius (in m).
    :param beam_divergence:     Equivalent to the opening angle of an individual LiDAR beam (in degree).

    :return:
    occlusion_list:             List of M Dicts.
                                Each Dict contains a Tuple of
                                If key == -1:
                                - distance to the original hard target
                                - angle that is not occluded by any particle
                                Else:
                                - the distance to an occluding particle
                                - the occluded angle by this particle

    """

    M = np.shape(beam_angles)[0]
    # print("M shape is :()".format(M))

    if root_path:
        path = Path(root_path) / 'training' / 'snowflakes' / 'npy' / particle_file
    else:
        # Use the global config path for snow NPY
        path = Path(_SNOW_NPY_DIR) / particle_file

    all_particles = np.load(str(path))
    x, y, _ = all_particles[:, 0], all_particles[:, 1], all_particles[:, 2]

    all_particle_ranges = np.linalg.norm([x, y], axis=0)                                                                # (N,)
    all_beam_limits_a, all_beam_limits_b = angles_to_lines(beam_angles)                                                 # (M, 2)

    occlusion_list = []

    # Main loop over beams.
    for i in range(M):

        current_range = ranges_orig[i]                                                                                  # (K,)

        right_angle = beam_angles[i, 0]
        left_angle = beam_angles[i, 1]

        in_range = np.where(all_particle_ranges < current_range)

        particles = all_particles[in_range]                                                                             # (K, 3)

        x, y, particle_radii = particles[:, 0], particles[:, 1], particles[:, 2]

        particle_angles = np.arctan2(y, x)                                                                              # (K,)
        particle_angles[particle_angles < 0] = particle_angles[particle_angles < 0] + 2 * PI

        tangents_a, tangents_b = tangents_from_origin(particles)                                                      # (K, 2)

        ################################################################################################################
        # Determine whether centers of the particles lie inside the current beam,
        # which is first sufficient condition for intersection.
        standard_case = np.logical_and(right_angle <= particle_angles, particle_angles <= left_angle)
        seldom_case = np.logical_and.reduce((right_angle - 2 * PI <= particle_angles, particle_angles <= left_angle,
                                             np.full_like(particle_angles, right_angle > left_angle, dtype=bool)))
        seldom_case_2 = np.logical_and.reduce((right_angle <= particle_angles, particle_angles <= left_angle + 2 * PI,
                                               np.full_like(particle_angles, right_angle > left_angle, dtype=bool)))

        center_in_beam = np.logical_or.reduce((standard_case, seldom_case, seldom_case_2))  # (K,)
        ################################################################################################################

        ################################################################################################################
        # Determine whether distances from particle centers to beam rays are smaller than the radii of the particles,
        # which is second sufficient condition for intersection.
        beam_limits_a = all_beam_limits_a[i, np.newaxis].T                                                              # (2, 1)
        beam_limits_b = all_beam_limits_b[i, np.newaxis].T                                                              # (2, 1)
        beam_limits_c = np.zeros((2, 1))  # origin                                                                      # (2, 1)

        # Get particle distances to right and left beam limit.
        distances = distances_of_points_to_lines(particles[:, :2],
                                                 beam_limits_a, beam_limits_b, beam_limits_c)                         # (K, 2)

        radii_intersecting = distances < np.column_stack((particle_radii, particle_radii))                              # (K, 2)

        intersect_right_ray = do_angles_intersect_particles(right_angle, particles[:, 0:2]).T                         # (K, 1)
        intersect_left_ray = do_angles_intersect_particles(left_angle, particles[:, 0:2]).T                           # (K, 1)

        right_beam_limit_hit = np.logical_and(radii_intersecting[:, 0], intersect_right_ray[:, 0])
        left_beam_limit_hit = np.logical_and(radii_intersecting[:, 1], intersect_left_ray[:, 0])

        ################################################################################################################
        # Determine whether particles intersect the current beam by taking the disjunction of the above conditions.
        particles_intersect_beam = np.logical_or.reduce((center_in_beam,
                                                         right_beam_limit_hit, left_beam_limit_hit))            # (K,)

        ################################################################################################################

        intersecting_beam = np.where(particles_intersect_beam)

        particles = particles[intersecting_beam]  # (L, 3)
        particle_angles = particle_angles[intersecting_beam]
        tangents_a = tangents_a[intersecting_beam]
        tangents_b = tangents_b[intersecting_beam]
        tangents = (tangents_a, tangents_b)
        right_beam_limit_hit = right_beam_limit_hit[intersecting_beam]
        left_beam_limit_hit = left_beam_limit_hit[intersecting_beam]

        # Get the interval angles from the tangents.
        tangent_angles = tangent_lines_to_tangent_angles(tangents, particle_angles)                                   # (L, 2)

        # Correct tangent angles that do exceed beam limits.
        interval_angles = tangent_angles_to_interval_angles(tangent_angles, right_angle, left_angle,
                                                            right_beam_limit_hit, left_beam_limit_hit)        # (L, 2)

        ################################################################################################################
        # Sort interval angles by increasing distance from origin.
        distances_to_origin = np.linalg.norm(particles[:, :2], axis=1)                                                  # (L,)

        intervals = np.column_stack((interval_angles, distances_to_origin))                                             # (L, 3)
        ind = np.argsort(intervals[:, -1])
        intervals = intervals[ind]                                                                                      # (L, 3)

        occlusion_list.append(compute_occlusion_dict((right_angle, left_angle),
                                                     intervals,
                                                     current_range,
                                                     beam_divergence))

    return occlusion_list

# only works with ground labels
def simulate_snow(severity: int,
                  pc: np.ndarray,
                  label: np.ndarray,
                  beam_divergence: float,
                  shuffle: bool=True,
                  noise_floor: float=0.7,
                  root_path: str=None) -> np.ndarray:
    """
    :param severity:                Integer 0-2
    :param pc:                      N-by-5 array containing original pointcloud (x, y, z, intensity, channel).
    :param label:                   Semantic labels.
    :param beam_divergence:         Beam divergence in degrees.
    :param shuffle:                 Flag if order of sampled snowflakes should be shuffled.
    :param noise_floor:             Noise floor threshold.
    :param root_path:               Optional root path, needed for training on GPU cluster.

    :return:                        numpy array of pointcloud
    """

    assert pc.shape[1] == 5

    labels = copy.deepcopy(label).reshape(-1)
    labels = np.vectorize(learning_map.__getitem__)(labels)
    driveable_surface = labels == 11
    other_flat = labels == 12
    sidewalk = labels == 13

    mask1 = np.logical_or(driveable_surface, other_flat)
    ground = np.logical_or(mask1, sidewalk)

    pc_ground = pc[ground, :]
    calculated_indicent_angle = np.arccos(-np.divide(np.matmul(pc_ground[:, :3], np.asarray([0, 0, 1])),
                                                     np.linalg.norm(pc_ground[:, :3],
                                                                    axis=1) * np.linalg.norm([0, 0, 1])))
    
    relative_output_intensity, adaptive_noise_threshold, _, _ = estimate_laser_parameters(pc_ground,
                                                                                          calculated_indicent_angle,
                                                                                          noise_floor=noise_floor,
                                                                                          debug=False)

    adaptive_noise_threshold *= np.cos(calculated_indicent_angle)

    ground_distances = np.linalg.norm(pc_ground[:, :3], axis=1)
    distances = np.linalg.norm(pc[:, :3], axis=1)

    p = np.polyfit(ground_distances, adaptive_noise_threshold, 2)

    relative_output_intensity = p[0] * distances ** 2 + p[1] * distances + p[2]

    orig_pc = copy.deepcopy(pc)
    aug_pc = copy.deepcopy(pc)

    # Use constant config path for laser yaml
    sensor_info = Path(_SNOW_LASER_YAML)

    with open(sensor_info, 'r') as stream:
        sensor_dict = yaml.safe_load(stream)

    channel_infos = sensor_dict['lasers']
    num_channels = sensor_dict['num_lasers']
    # num_channels = 32

    channels = range(num_channels)
    order = list(range(num_channels))
    
    if shuffle:
        np.random.shuffle(order)

    channel_list = [None] * num_channels

    # Mapping 3 severity levels
  
    s_vals = [
        (0.5, 1.2),  
        (2.5, 1.6),  
        (1.5, 0.4)   
    ]
    safe_severity = min(severity, 2)
    s = s_vals[safe_severity]

    rain_rate = snowfall_rate_to_rainfall_rate(float(s[0]), float(s[1]))
    occupancy = compute_occupancy(float(s[0]), float(s[1]))
    particle_file_prefix = f'gunn_{rain_rate}_{occupancy}' 
    
    channel_list = []

    for channel in channels:
        print(f"Processing channel {channel}/{num_channels}...", end='\r')
        result = process_single_channel(root_path, particle_file_prefix,
                                        orig_pc, beam_divergence, order,
                                        channel_infos, channel)
        channel_list.append(result)


    intensity_diff_sum = 0
    # import pdb; pdb.set_trace()
    for item in channel_list:

        tmp_intensity_diff_sum, idx, pc_ = item

        intensity_diff_sum += tmp_intensity_diff_sum

        aug_pc[idx] = pc_

    aug_pc[:, 3] = aug_pc[:, 3] 
    scattered = aug_pc[:, 4] == 2
    above_threshold = aug_pc[:, 3] > relative_output_intensity[:]
    scattered_or_above_threshold = np.logical_or(scattered, above_threshold)
    num_removed = np.logical_not(scattered_or_above_threshold).sum()

    aug_pc = aug_pc[np.where(scattered_or_above_threshold)]

    num_attenuated = (aug_pc[:, 4] == 1).sum()

    if num_attenuated > 0:
        avg_intensity_diff = int(intensity_diff_sum / num_attenuated)
    else:
        avg_intensity_diff = 0

    stats = num_attenuated, num_removed, avg_intensity_diff

    return aug_pc


# works with no ground labels but long runtime. hasnt worked yet.
def simulate_snow_sweep(severity: int,
                        pc: np.ndarray,
                        beam_divergence: float=0.1, #assumption for fortuna car
                        shuffle: bool=True,
                        noise_floor: float=0.7,
                        root_path: str=None) -> np.ndarray:
    """
    :param severity:                Integer 1-3
    :param pc:                      N-by-5 array containing original pointcloud (x, y, z, intensity, channel).
    :param beam_divergence:         Beam divergence in degrees.
    :param shuffle:                 Flag if order of sampled snowflakes should be shuffled.
    :param noise_floor:             Noise floor threshold.
    :param root_path:               Optional root path, needed for training on GPU cluster.

    :return:                        N-by-4 array of the augmented pointcloud.
    """

    assert pc.shape[1] == 5

    w, h = estimate_ground_plane(pc)
    ground = np.logical_and(np.matmul(pc[:, :3], np.asarray(w)) + h < 0.5,
                            np.matmul(pc[:, :3], np.asarray(w)) + h > -0.5)

    pc_ground = pc[ground, :]
    calculated_indicent_angle = np.arccos(-np.divide(np.matmul(pc_ground[:, :3], np.asarray([0, 0, 1])),
                                                     np.linalg.norm(pc_ground[:, :3],
                                                                    axis=1) * np.linalg.norm([0, 0, 1])))
    
    relative_output_intensity, adaptive_noise_threshold, _, _ = estimate_laser_parameters(pc_ground,
                                                                                          calculated_indicent_angle,
                                                                                          noise_floor=noise_floor,
                                                                                          debug=False)

    adaptive_noise_threshold *= np.cos(calculated_indicent_angle)

    ground_distances = np.linalg.norm(pc_ground[:, :3], axis=1)
    distances = np.linalg.norm(pc[:, :3], axis=1)

    p = np.polyfit(ground_distances, adaptive_noise_threshold, 2)

    relative_output_intensity = p[0] * distances ** 2 + p[1] * distances + p[2]

    orig_pc = copy.deepcopy(pc)
    aug_pc = copy.deepcopy(pc)

    sensor_info = Path(_SNOW_LASER_YAML)

    with open(sensor_info, 'r') as stream:
        sensor_dict = yaml.safe_load(stream)

    channel_infos = sensor_dict['lasers']
    num_channels = sensor_dict['num_lasers']
    # num_channels = 32

    channels = range(num_channels)
    order = list(range(num_channels))
    
    if shuffle:
        np.random.shuffle(order)

    channel_list = [None] * num_channels

    # Mapping 3 severity levels
    s_vals = [
        (0.5, 1.2),  
        (2.5, 1.6),  
        (1.5, 0.4)   
    ]
    safe_severity = min(severity, 2)
    s = s_vals[safe_severity]

    rain_rate = snowfall_rate_to_rainfall_rate(float(s[0]), float(s[1]))
    occupancy = compute_occupancy(float(s[0]), float(s[1]))
    particle_file_prefix = f'gunn_{rain_rate}_{occupancy}' 
    
    channel_list = []

    for channel in channels:
        result = process_single_channel(root_path, particle_file_prefix,
                                        orig_pc, beam_divergence, order,
                                        channel_infos, channel)
        channel_list.append(result)


    intensity_diff_sum = 0

    for item in channel_list:

        tmp_intensity_diff_sum, idx, pc_ = item

        intensity_diff_sum += tmp_intensity_diff_sum

        aug_pc[idx] = pc_

    aug_pc[:, 3] = aug_pc[:, 3] 
    scattered = aug_pc[:, 4] == 2
    above_threshold = aug_pc[:, 3] > relative_output_intensity[:]
    scattered_or_above_threshold = np.logical_or(scattered, above_threshold)
    num_removed = np.logical_not(scattered_or_above_threshold).sum()

    aug_pc = aug_pc[np.where(scattered_or_above_threshold)]

    num_attenuated = (aug_pc[:, 4] == 1).sum()

    if num_attenuated > 0:
        avg_intensity_diff = int(intensity_diff_sum / num_attenuated)
    else:
        avg_intensity_diff = 0

    stats = num_attenuated, num_removed, avg_intensity_diff

    return aug_pc


def received_power(CA_P0: float, beta_0: float, ratio: float, r: float, r_j: float, tau_h: float) -> float:
    answer = ((CA_P0 * beta_0 * ratio * xsi(r_j)) / (r_j ** 2)) * np.sin((PI * (r - r_j)) / (c * tau_h)) ** 2
    return answer

def xsi(R: float, R_1: float = 0.9, R_2: float = 1.0) -> float:
    if R <= R_1:    # emitted ligth beam from the tansmitter is not captured by the receiver
        return 0
    elif R >= R_2:  # emitted ligth beam from the tansmitter is fully captured by the receiver
        return 1
    else:           # emitted ligth beam from the tansmitter is partly captured by the receiver
        m = (1 - 0) / (R_2 - R_1)
        b = 0 - (m * R_1)
        y = m * R + b
        return y
    
########################### 3D_Corruptions_AD ##############################
#               https://github.com/thu-ml/3D_Corruptions_AD 
#               We adapted the code from 3D_Corruptions_AD for our use case.
#               We made some modifications to the code to fit our use case, such as changing the input and output formats, and removing some functionalities that are not relevant to our use case.
######################################################################################

# Weather Corruptions

'''
Rain
'''
def rain_sim(severity, pointcloud   ):
    from .utils import lisa
    rain_sim = lisa.LISA(show_progressbar=True)
    c = [0.20, 0.73, 1.5625, 3.125, 7.29, 10.42][severity]
    
    # Enforce KITTI format (N x 4) to prevent LISA shape broadcasting crashes
    pc_4 = pointcloud[:, :4] if pointcloud.shape[1] > 4 else pointcloud
    
    points = rain_sim.augment(pc_4, c)
    return points
    

'''
Snow
'''
def snow_sim(severity, pointcloud):
    from .utils import lisa
    from .utils.wet_ground.augmentation import ground_water_augmentation
    snow_sim = lisa.LISA(mode='gunn', show_progressbar=True) 
    c = [0.20, 0.73, 1.5625, 3.125, 7.29, 10.42][severity]
    
    # Enforce KITTI format (N x 4) to prevent LISA shape broadcasting crashes
    pc_4 = pointcloud[:, :4] if pointcloud.shape[1] > 4 else pointcloud
    
    points = snow_sim.augment(pc_4, c)
    return points
#TODO: DROP BECAUSE SAME AS BELOW AND DIFF gunn files.
'''
Fog: 
'''
def fog_sim(severity, pointcloud):
    from .utils.fog_sim import simulate_fog
    from .utils.fog_sim import ParameterSet
    c = [0.005, 0.01, 0.02, 0.03, 0.06][severity] # form original paper
    parameter_set = ParameterSet(alpha=c, gamma=0.000001)
    points, _, _ = simulate_fog(parameter_set, pointcloud, 1)
    return points

'''
Sunlight
'''
def scene_glare_noise(severity, pointcloud):
    N, C = pointcloud.shape
    c = [int(0.010*N), int(0.020*N),int(0.030*N),int(0.040*N), int(0.050*N)][severity]
    index = np.random.choice(N, c, replace=False)
    pointcloud[index] += np.random.normal(size=(c, C)) * 2.0
    return pointcloud

# Sensor Corruptions


'''
Crosstalk
'''
def lidar_crosstalk_noise(severity, pointcloud):
    N, C = pointcloud.shape
    c = [int(0.004*N), int(0.008*N),int(0.012*N),int(0.016*N), int(0.020*N)][severity]
    index = np.random.choice(N, c, replace=False)
    pointcloud[index] += np.random.normal(size=(c, C)) * 3.0
    return pointcloud


'''
Density
'''
def density_dec_global(severity, pointcloud):
    N, C = pointcloud.shape
    num = int(N * 0.3)
    c = [int(0.2*num), int(0.4*num), int(0.6*num), int(0.8*num), num][severity]
    idx = np.random.choice(N, c, replace=False)
    pointcloud = np.delete(pointcloud, idx, axis=0)
    return pointcloud

'''
Cutout
'''
def cutout_local(severity, pointcloud):
    N, C = pointcloud.shape
    num = int(N*0.02)
    c = [(2,num), (3,num), (5,num), (7,num), (10,num)][severity]
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0],1)
        picked = pointcloud[i]
        dist = np.sum((pointcloud - picked)**2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
    return pointcloud


'''
Gaussian (L)
'''
def gaussian_noise_lidar(severity, pointcloud):
    N, C = pointcloud.shape # N*3
    c = [0.02, 0.04, 0.06, 0.08, 0.10][severity]
    jitter = np.random.normal(size=(N, C)) * c
    new_pc = (pointcloud + jitter).astype('float32')
    return new_pc

'''
Uniform (L)
'''
def uniform_noise(severity, pointcloud):
    # TODO
    N, C = pointcloud.shape
    c = [0.02, 0.04, 0.06, 0.08, 0.10][severity]
    jitter = np.random.uniform(-c, c, (N, C))
    new_pc = (pointcloud + jitter).astype('float32')
    return new_pc

'''
Impulse (L)
'''

def impulse_noise_lidar(severity, pointcloud):
    N, C = pointcloud.shape
    c = [N // 30, N // 25, N // 20, N // 15, N // 10][severity]
    index = np.random.choice(N, c, replace=False)
    pointcloud[index] += np.random.choice([-1, 1], size=(c, C)) * 0.1
    return pointcloud

'''
Fov lost
'''

def fov_filter(severity, pointcloud):
    # Extract the angle bounds based on severity
    angle1 = [-105, -90, -75, -60, -45][severity]
    angle2 = [105, 90, 75, 60, 45][severity]
    
    # Handle both raw Numpy arrays and OpenPCDet BasePoints
    if isinstance(pointcloud, np.ndarray):
        pts_npy = pointcloud
    else:
        # If it's a BasePoints object (common in 3D_Corruptions_AD), extract the tensor
        pts_npy = pointcloud.tensor.numpy()

    # --- THE FIX ---
    # np.arctan2(X, Y) completely eliminates the ZeroDivisionError 
    # and natively outputs the exact [-pi, pi] angles we need.
    pts_p = np.arctan2(pts_npy[:, 0], pts_npy[:, 1])
    
    # Convert from Radians to Degrees for the filter check
    pts_p = np.rad2deg(pts_p)
    # ---------------

    # Keep only the points that fall inside the FOV angles
    filt = np.logical_and(pts_p >= angle1, pts_p <= angle2)

    return pointcloud[filt]


# Motion corruptions

'''
Moving Obj. 
'''
def moving_noise_bbox(severity, pointcloud, bbox):
    cor = 'move_bbox'
    from .utils import bbox_util
    pointcloud = bbox_util.pick_bbox(cor,severity,bbox,pointcloud)
    return pointcloud

'''
Motion Compensation
'''
def fulltrajectory_noise(severity, pointcloud, pc_pose):
    from .utils.lidar_split import lidar_split, reconstruct_pc
    ct = [0.02, 0.04, 0.06, 0.08, 0.10][severity]
    cr = [0.002, 0.004, 0.006, 0.008, 0.010][severity]
    new_pose_list, new_lidar_list = lidar_split(pointcloud, pc_pose)
    r_noise = np.random.normal(size=(100, 3, 3)) * cr
    t_noise = np.random.normal(size=(100, 3)) * ct
    new_pose_list[:, :3, :3] += r_noise
    new_pose_list[:, :3, 3] += t_noise
    f_pc = reconstruct_pc(new_lidar_list, new_pose_list)
    return f_pc




# Object corruptions

'''
Local Density
'''

def density_dec_bbox(severity, pointcloud, bbox):
    cor = 'density_dec_bbox'
    from .utils import bbox_util
    pointcloud = bbox_util.pick_bbox(cor,severity,bbox,pointcloud)
    return pointcloud

'''
Local Cutout
'''
def cutout_bbox(severity, pointcloud, bbox):
    cor = 'cutout_bbox'
    from .utils import bbox_util
    pointcloud = bbox_util.pick_bbox(cor,severity,bbox,pointcloud)
    return pointcloud


'''
Local Gaussian
'''
def gaussian_noise_bbox(severity, pointcloud, bbox):
    cor = 'gaussian_noise_bbox'
    from .utils import bbox_util
    pointcloud = bbox_util.pick_bbox(cor,severity,bbox,pointcloud)
    return pointcloud

'''
Local Uniform
'''

def uniform_noise_bbox(severity, pointcloud, bbox):
    cor = 'uniform_noise_bbox'
    from .utils import bbox_util
    pointcloud = bbox_util.pick_bbox(cor,severity,bbox,pointcloud)
    return pointcloud

'''
Local Impulse
'''

def impulse_noise_bbox(severity, pointcloud, bbox):
    cor = 'impulse_noise_bbox'
    from .utils import bbox_util
    pointcloud = bbox_util.pick_bbox(cor,severity,bbox,pointcloud)
    return pointcloud

'''
Scale
'''
def scale_bbox(severity, pointcloud, bbox):
    cor = 'scale_bbox'
    from .utils import bbox_util
    pointcloud = bbox_util.pick_bbox(cor,severity,bbox,pointcloud)
    return pointcloud

'''
Shear
'''
def shear_bbox(severity, pointcloud, bbox):
    cor = 'shear_bbox'
    from .utils import bbox_util
    pointcloud = bbox_util.pick_bbox(cor,severity,bbox,pointcloud)
    return pointcloud

'''
Rotation
'''
def rotation_bbox(severity, pointcloud, bbox):
    cor = 'rotation_bbox'
    from .utils import bbox_util
    pointcloud = bbox_util.pick_bbox(cor,severity,bbox,pointcloud)
    return pointcloud


# Alignment

'''
Spatial
'''

def spatial_alignment_noise(severity, ori_pose):
    '''
    input: ori_pose 4*4
    output: noise_pose 4*4
    '''
    ct = [0.02, 0.04, 0.06, 0.08, 0.10][severity]*2
    cr = [0.002, 0.004, 0.006, 0.008, 0.010][severity]*2
    r_noise = np.random.normal(size=(3, 3)) * cr
    t_noise = np.random.normal(size=(3)) * ct
    ori_pose[:3, :3] += r_noise
    ori_pose[:3, 3] += t_noise
    return ori_pose


'''
Temporal
'''
def temporal_alignment_noise(severity,pointcloud):
    return pointcloud


###### Approximation Methods for Fast Simulation of Weather Effects ######

'''
Fast Real-Time Rain Approximation
Since both multiCorrupt and 3d_corruptions_ad's rain simulation are computationally expensive,
we propose a fast approximation that captures the key effects of rain on LiDAR point clouds:
attenuation (point dropout) and backscatter (ghost points).
This method is designed for real-time applications and can be used as a quick way to simulate rain effects
without the overhead of more complex physical models.
'''
def fast_rain(severity, pointcloud):
    
    # 1. Define severity scales (0 to 4 mapping)
    # drop_rates: 5%, 10%, 15%, 20%, 30% of points disappear
    drop_rates = [0.05, 0.10, 0.15, 0.20, 0.30]
    # ghost_rates: add 1%, 2%, 5%, 8%, 12% backscatter noise near the sensor
    ghost_rates = [0.01, 0.02, 0.05, 0.08, 0.12]
    
    drop_rate = drop_rates[severity]
    ghost_rate = ghost_rates[severity]
    
    N = pointcloud.shape[0]
    
    # --- Step 1: Attenuation (Fast Point Dropout) ---
    # Create a random mask to keep points
    keep_mask = np.random.rand(N) > drop_rate
    pc_dropped = pointcloud[keep_mask]
    
    # --- Step 2: Backscatter (Fast Ghost Points) ---
    num_ghosts = int(N * ghost_rate)
    ghost_points = np.zeros((num_ghosts, pointcloud.shape[1]), dtype=np.float32)
    
    # Generate random points within a sphere of radius R (mostly close to sensor, e.g., 0.5m to 4.0m)
    r = np.random.uniform(0.5, 4.0, num_ghosts)
    theta = np.random.uniform(0, 2 * np.pi, num_ghosts)
    
    # Convert polar to Cartesian (X, Y)
    ghost_points[:, 0] = r * np.cos(theta) # X
    ghost_points[:, 1] = r * np.sin(theta) # Y
    # Z usually stays within the vertical spread of the sensor (-2m to 2m)
    ghost_points[:, 2] = np.random.uniform(-2.0, 2.0, num_ghosts)
    
    # Rain backscatter has very low intensity (if intensity column exists)
    if ghost_points.shape[1] > 3:
        ghost_points[:, 3] = np.random.uniform(0, 15, num_ghosts) 
    # Match the ring column if it exists (arbitrary ring assignment)
    if ghost_points.shape[1] > 4:
        ghost_points[:, 4] = np.random.randint(0, 32, num_ghosts)
        
    # Combine the surviving points with the ghost points
    fast_rain_pc = np.vstack((pc_dropped, ghost_points))
    
    return fast_rain_pc

def fast_fog(severity, points_np):
    # Ensure severity stays strictly within 0 to 4
    idx = max(0, min(int(severity), 4))
    
    # Define severity scales (Index 0 to 4)
    # max_visible_ranges: At level 4, the LiDAR can only see 15 meters.
    max_visible_ranges = [80.0, 60.0, 40.0, 25.0, 15.0]
    # ghost_counts: Number of backscatter points injected near the sensor
    ghost_counts = [200, 500, 1000, 2000, 3000]
    
    max_visible_range = max_visible_ranges[idx]
    num_ghost_points = ghost_counts[idx]
    
    N = points_np.shape[0]
    if N == 0:
        return points_np
    
    # --- Step 1: Attenuation (Drop distant points) ---
    distances = np.sqrt(points_np[:, 0]**2 + points_np[:, 1]**2 + points_np[:, 2]**2)
    drop_probabilities = np.clip(distances / max_visible_range, 0.0, 1.0)
    keep_mask = np.random.rand(N) > drop_probabilities
    attenuated_points = points_np[keep_mask]
    
    # --- Step 2: Backscatter (Ghost points near car) ---
    ghost_points = np.zeros((num_ghost_points, points_np.shape[1]), dtype=np.float32)
    
    # Fog backscatter happens very close to the ego-vehicle (-3m to +3m)
    ghost_points[:, 0] = np.random.uniform(-3.0, 3.0, num_ghost_points) # X
    ghost_points[:, 1] = np.random.uniform(-3.0, 3.0, num_ghost_points) # Y
    ghost_points[:, 2] = np.random.uniform(-2.0, 2.0, num_ghost_points) # Z
    
    # Low intensity for fog
    if ghost_points.shape[1] > 3:
        ghost_points[:, 3] = np.random.uniform(0.0, 10.0, num_ghost_points)
    if ghost_points.shape[1] > 4:
        ghost_points[:, 4] = np.random.randint(0, 32, num_ghost_points)
        
    fast_fog_cloud = np.vstack((attenuated_points, ghost_points))
    
    return fast_fog_cloud


def fast_snow(severity, points_np):
    # Ensure severity stays strictly within 0 to 4
    idx = max(0, min(int(severity), 4))
    
    # Define severity scales (Index 0 to 4)
    # drop_rates: 2%, 5%, 10%, 15%, 20% random beam occlusion
    drop_rates = [0.02, 0.05, 0.10, 0.15, 0.20]
    # flake_counts: Number of highly reflective volumetric flakes
    flake_counts = [500, 2000, 4000, 6000, 8000]
    
    drop_ratio = drop_rates[idx]
    num_flakes = flake_counts[idx]
    
    N = points_np.shape[0]
    if N == 0:
        return points_np
    
    # --- Step 1: Beam Occlusion (Random Attenuation) ---
    keep_mask = np.random.rand(N) > drop_ratio
    attenuated_points = points_np[keep_mask]
    
    # --- Step 2: High-Intensity Ghost Flakes ---
    ghost_flakes = np.zeros((num_flakes, points_np.shape[1]), dtype=np.float32)
    
    # Snowflakes fall everywhere (-40m to +40m around the car)
    ghost_flakes[:, 0] = np.random.uniform(-40.0, 40.0, num_flakes) # X
    ghost_flakes[:, 1] = np.random.uniform(-40.0, 40.0, num_flakes) # Y
    ghost_flakes[:, 2] = np.random.uniform(-2.0, 10.0, num_flakes)  # Z (falling from sky)
    
    # High intensity for snow (highly reflective)
    if ghost_flakes.shape[1] > 3:
        ghost_flakes[:, 3] = np.random.uniform(50.0, 150.0, num_flakes)
    if ghost_flakes.shape[1] > 4:
        ghost_flakes[:, 4] = np.random.randint(0, 32, num_flakes)
        
    fast_snow_cloud = np.vstack((attenuated_points, ghost_flakes))
    
    return fast_snow_cloud