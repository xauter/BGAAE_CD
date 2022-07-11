import tensorflow as tf
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral
from pydensecrf.utils import create_pairwise_gaussian

def dense_gaussian_filtering(x, y, difference_img):
    """
        Concerning the filtering, the method proposed in
        krahenbuhl2011efficient is used. It exploits spatial context to
        filter $d$ with fully connected conditional random field models.
        It defines the pairwise edge potentials between all pairs of pixels
        in the image by a linear combination of Gaussian kernels in an
        arbitrary feature space. The main downside of the iterative
        optimisation of the random field lies in the fact that it requires
        the propagation of all the potentials across the image.
        However, this highly efficient algorithm reduces the computational
        complexity from quadratic to linear in the number of pixels by
        approximating the random field with a mean field whose iterative
        update can be computed using Gaussian filtering in the feature
        space. The number of iterations and the kernel width of the
        Gaussian kernels are the only hyper-parameters manually set,
        and we opted to tune them according to luppino2019unsupervised:
        $5$ iterations and a kernel width of $0.1$.
    """

    d = np.array(difference_img[0])
    d = np.concatenate((d, 1.0 - d), axis=2)
    W, H = d.shape[:2]
    stack = np.concatenate((x[0], y[0]), axis=-1)

    CD = dcrf.DenseCRF2D(W, H, 2)
    d[d == 0] = 10e-20
    U = -(np.log(d))
    U = U.transpose(2, 0, 1).reshape((2, -1))
    U = U.copy(order="C")
    CD.setUnaryEnergy(U.astype(np.float32))
    pairwise_energy_gaussian = create_pairwise_gaussian((10, 10), (W, H))
    CD.addPairwiseEnergy(pairwise_energy_gaussian, compat=1)
    pairwise_energy_bilateral = create_pairwise_bilateral(
        sdims=(10, 10), schan=(0.1,), img=stack, chdim=2
    )
    CD.addPairwiseEnergy(pairwise_energy_bilateral, compat=1)
    Q = CD.inference(3)
    heatmap = np.array(Q, dtype=np.float32)
    heatmap = np.reshape(heatmap[0, ...], (1, W, H, 1))
    return tf.convert_to_tensor(heatmap)

def get_difference_img(sx, sy, bandwidth=tf.constant(3, dtype=tf.float32)):
    """
        Compute difference image.
        Bandwidth governs the norm difference clipping threshold
    """
    d = tf.norm(sx - sy, ord=2, axis=-1)
    threshold = tf.math.reduce_mean(d) + bandwidth * tf.math.reduce_std(d)
    d = tf.where(d < threshold, d, threshold)

    return tf.expand_dims(d / tf.reduce_max(d), -1)

def threshold_otsu(image):
    """Return threshold value based on Otsu's method. Adapted to tf from sklearn
    Parameters
    ----------
    image : (N, M) ndarray
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.
    Raises
    ------
    ValueError
         If ``image`` only contains a single grayscale value.
    References
    ----------
    .. [1] Wikipedia, https://en.wikipedia.org/wiki/Otsu's_Method
    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_otsu(image)
    >>> binary = image <= thresh
    Notes
    -----
    The input image must be grayscale.
    """
    if len(image.shape) > 2 and image.shape[-1] in (3, 4):
        msg = (
            "threshold_otsu is expected to work correctly only for "
            "grayscale images; image shape {0} looks like an RGB image"
        )
        warn(msg.format(image.shape))

    # Check if the image is multi-colored or not
    tf.debugging.assert_none_equal(
        tf.math.reduce_min(image),
        tf.math.reduce_max(image),
        summarize=1,
        message="expects more than one image value",
    )

    hist = tf.histogram_fixed_width(image, tf.constant([0, 255]), 256)
    hist = tf.cast(hist, tf.float32)
    bin_centers = tf.range(0.5, 256, dtype=tf.float32)

    # class probabilities for all possible thresholds
    weight1 = tf.cumsum(hist)
    weight2 = tf.cumsum(hist, reverse=True)
    # class means for all possible thresholds
    mean = tf.math.multiply(hist, bin_centers)
    mean1 = tf.math.divide(tf.cumsum(mean), weight1)
    # mean2 = (tf.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
    mean2 = tf.math.divide(tf.cumsum(mean, reverse=True), weight2)

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    tmp1 = tf.math.multiply(weight1[:-1], weight2[1:])
    tmp2 = (mean1[:-1] - mean2[1:]) ** 2
    variance12 = tf.math.multiply(tmp1, tmp2)

    idx = tf.math.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold

def get_change_map(difference_img):
    """
        Input:
            difference_img - tensor of shape (h, w), (1, h, w)
                             or (1, h, w, 1) in [0,1]
        Output:
            change_map - tensor with same shape as input, bool
    """
    tmp = tf.cast(difference_img * 255, tf.int32)
    threshold = threshold_otsu(tmp) / 255

    return difference_img >= threshold

def clip(image):
    """
        Normalize image from R_+ to [-1, 1].

        For each channel, clip any value larger than mu + 3sigma,
        where mu and sigma are the channel mean and standard deviation.
        Scale to [-1, 1] by (2*pixel value)/(max(channel)) - 1

        Input:
            image - (h, w, c) image array in R_+
        Output:
            image - (h, w, c) image array normalized within [-1, 1]
    """
    temp = np.reshape(image, (-1, image.shape[-1]))
    limits = tf.reduce_mean(temp, 0) + 3.0 * tf.math.reduce_std(temp, 0)
    for i, limit in enumerate(limits):
        channel = temp[:, i]
        channel = tf.clip_by_value(channel, 0, limit)
        ma, mi = tf.reduce_max(channel), tf.reduce_min(channel)
        channel = 2.0 * ((channel) / (ma)) - 1
        temp[:, i] = channel

    return tf.reshape(tf.convert_to_tensor(temp, dtype=tf.float32), image.shape)