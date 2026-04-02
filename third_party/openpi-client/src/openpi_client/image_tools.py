import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def resize_with_pad_depth(images: np.ndarray, height: int, width: int, method=Image.NEAREST) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple depth images using PIL. 
    
    Resizes a batch of depth images to a target height and width without distortion by padding with zeros.
    This function is specifically designed for float32 depth images with shape (H, W, 1).
    Args:
        images: A batch of depth images in [..., height, width, 1] format with float32 dtype.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.
    Returns:
        The resized depth images in [..., height, width, 1] with float32 dtype.
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_depth_pil(im, height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image

def _resize_with_pad_depth_pil(depth_image: np.ndarray, height: int, width: int, method: int) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for one depth image using PIL. 
    
    Resizes a depth image to a target height and width without distortion by padding with zeros.
    This function handles float32 depth images with shape (H, W, 1).
    Args:
        depth_image: A depth image with shape (H, W, 1) and float32 dtype.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use.
    Returns:
        The resized depth image with shape (H, W, 1) and float32 dtype.
    """
    # Remove the channel dimension for PIL processing
    depth_2d = depth_image.squeeze(axis=-1)

    # Check if already the correct size
    if depth_2d.shape == (height, width):
        return depth_image

    # Convert to PIL Image for processing
    # Note: PIL expects uint8 for images, but we'll use float32 directly
    # We need to normalize the depth values to 0-255 range for PIL processing
    depth_min = np.min(depth_2d)
    depth_max = np.max(depth_2d)

    if depth_max > depth_min:
        # Normalize to 0-255 range
        depth_normalized = ((depth_2d - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    else:
        # If all values are the same, create a zero image
        depth_normalized = np.zeros_like(depth_2d, dtype=np.uint8)

    pil_image = Image.fromarray(depth_normalized)

    # Calculate resize dimensions
    cur_height, cur_width = depth_2d.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize the image
    resized_pil = pil_image.resize((resized_width, resized_height), resample=method)

    # Create padded image
    padded_pil = Image.new('L', (width, height), 0)  # 'L' mode for grayscale
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    padded_pil.paste(resized_pil, (pad_width, pad_height))

    # Convert back to numpy
    padded_array = np.array(padded_pil, dtype=np.float32)

    # Denormalize back to original depth range
    if depth_max > depth_min:
        padded_array = (padded_array / 255.0) * (depth_max - depth_min) + depth_min

    # Add channel dimension back
    return padded_array[..., np.newaxis]
