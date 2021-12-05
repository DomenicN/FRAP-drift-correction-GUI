"""
ProcessingUtilities.py - Script that contains image processing utilities
such as background detection, photobleaching, etc.
"""
import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import scipy.stats as stats
import cv2
import warnings
from scipy.optimize import curve_fit
from skimage.filters import threshold_multiotsu


def subtractBackground(img, method='segmentation', coords=None,
                       radius=0, plot=False):
    """
    Takes an image and performs background subtraction.
    :param img: image matrix
    :param method: (str) indicates method for subtraction. Histfit
    fits a gaussian to the first peak and uses the mean as background.
    Segmentation segments background, and subtracts.
    :param coords: (tuple) coordinates of ROI
    :param radius: (int) radius of ROI
    :param plot: (bool) plot?
    :return: background subtracted image
    """
    METHODS = ['histfit', 'segmentation']
    if method == 'histfit':
        gm = fitGMM(img)
        # get fitted parameters
        weights, means, covars = gm.weights_, gm.means_, gm.covariances_
        if plot:
            flattened_data = img[:, :, 0].ravel().reshape(-1, 1)
            plt.hist(flattened_data, bins=100, histtype='bar', density=True, ec='red', alpha=0.5)

            f_axis = flattened_data.copy().ravel()
            f_axis.sort()
            for i in range(len(weights)):
                plt.plot(f_axis, weights[i] * stats.norm.pdf(f_axis, means[i], np.sqrt(covars[i])).ravel(), c='red')

            plt.rcParams['agg.path.chunksize'] = 10000
            plt.show()

        # subtract background
        img -= min(means)
        img[img < 0] = 0
        return img
    elif method == 'segmentation':
        if coords is None:
            coords = [(0, 0) for i in range(img.shape[-1])]

        img = np.array([img[:, :, i] - np.mean(segmentCell(img[:, :, i],
                                    tuple(reversed(coords[i])),radius, inverted=True))
                                    for i in range(img.shape[-1])])

        img[img < 0] = 0
        img = np.moveaxis(img, [0], [-1])
        if plot:
            flattened_data = img[:, :, 0].ravel().reshape(-1, 1)
            plt.hist(flattened_data, bins=100, histtype='bar', density=True, ec='red', alpha=0.5)

            plt.rcParams['agg.path.chunksize'] = 10000
            plt.show()
        return img
    else:
        error_str = 'Method must be one of: ' + ' '.join(METHODS)
        raise ValueError(error_str)

def correctPhotobleaching(mov):
    """
    Estimates photbleaching rate and corrects photobleaching
    :param mov: FRAPImage
    :return: Corrected movie, photobleaching rate, variance of tau
    """
    # get intensity data
    print('Getting intensities...')
    frame = mov.get_image_data()
    radius = int(np.ceil(mov.get_roi_radii()[0]))
    coords = [(pos[0] + radius, pos[1] + radius) for pos in mov.get_viewer_coords()]
    coords = np.around(coords, decimals=0).astype(int)
    start_frame = mov.get_start_frame()
    mean_intensities = [np.mean(segmentCell(frame[:, :, i], tuple(reversed(coords[i])), radius)) \
                        for i in range(mov.get_tdim())]
    intensity_ratios = [intensity / mean_intensities[start_frame] for intensity in mean_intensities]

    # fit exponential decay
    print('Fitting decay...')
    def func(x, tau):
        return np.exp(-abs(tau) * x)
    x_data = mov.get_frame_metadata()[:, 1].copy()
    x_data -= x_data[start_frame]
    popt, pcov = curve_fit(func, x_data, intensity_ratios, bounds=(0, np.inf))

    # correct image
    print('Correcting photobleaching...')
    correction_factors = func(x_data, *popt)
    for i in range(mov.get_tdim()):
        frame[:, :, i] /= correction_factors[i]
    corrected_mean_intensities = [mean_intensities[i] / correction_factors[i] for i in range(len(mean_intensities))]
    return frame, popt, pcov, mean_intensities, corrected_mean_intensities

def segmentCell(frame, roi_center, roi_radius, inverted=False, plot=False, return_mask=False):
    """
    Segments the cell region of interest and returns
    values as a vector.
    :param frame: A single frame
    :param roi_center: center coordinates of the ROI
    :param roi_radius: radius of the ROI
    :param inverted: (bool) Should data be inverted from mask?
    :param plot: (bool) Should the mask be plotted?
    :return: Vector of values within region, binary mask
    """

    # apply gaussian blur
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    blur = cv2.circle(blur, roi_center, int(roi_radius + 4), (0,0,0), -1)

    # compute threshold
    threshold = min(threshold_multiotsu(blur))

    binary_mask = blur > threshold
    kernel = np.ones((10, 10), np.uint8)
    binary_mask = binary_mask.astype('float64')
    # close gaps
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    # expand slightly
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    # remove unwanted noise
    kernel = np.ones((50, 50), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.circle(binary_mask, roi_center, roi_radius, (0,0,0), -1)
    if plot:
        plt.imshow(binary_mask)
        plt.show()
    if inverted:
        return frame[np.logical_not(binary_mask.astype('bool'))]
    if return_mask:
        return binary_mask
    return frame[binary_mask.astype('bool')]

def fitGMM(frame, components=3):
    """
    Fits Gaussian mixture model to the image histogram
    :param frame: image frame
    :param components: number of components for GMM
    :return: gm object
    """
    print('Fitting GMM...')
    gm = GaussianMixture(n_components=components, warm_start=True)
    flattened_data = frame.ravel().reshape(-1, 1)
    gm.fit(flattened_data)
    return gm

def processImage(frap_image, subtract_bg = False, prebleach_ss=True):
    """
    Function which processes frap image
    :param frap_image: FRAP Image
    """
    # background subtraction
    frap_image.reset_image_data()
    frame_data = frap_image.get_image_data()
    if subtract_bg:
        print('Subtracting background...')
        radius = int(np.ceil(frap_image.get_roi_radii()[0]))
        coords = [(pos[0] + radius, pos[1] + radius) for pos in frap_image.get_viewer_coords()]
        coords = np.around(coords, decimals=0).astype(int)
        frame_data = subtractBackground(frame_data, coords=coords, radius=radius)
        frap_image.set_image_data(frame_data)

    # photobleaching correction
    frame_data, popt, pcov, mi, cmi = correctPhotobleaching(frap_image)
    frap_image.set_image_data(frame_data)
    frap_image.update_viewer_coords()
    frap_image.set_nonbleach_intensities(mi)
    frap_image.set_corrected_nonbleach_intensities(cmi)
    gap_ratio, bleaching_depth = computeQCMetrics(frap_image, prebleach_ss)
    frap_image.set_gap_ratio(gap_ratio)
    frap_image.set_bleaching_depth(bleaching_depth)
    return popt, pcov

def normalizeFRAPCurve(data, start_frame, method='Double', prebleach_ss = True):
    """
    Normalized FRAP curve to be from 0 to 1
    :param data: (series) FRAP curve data
    :param start_frame: (int) Index of bleaching start frame
    :param method: (str) Method of normalization
    :return: (series) Normalized data
    """
    METHODS = ['Fullscale', 'Double']
    if prebleach_ss:
        ss = np.mean(data[:start_frame])

    else:
        ss = np.mean(data[-5:])

    if method == 'Fullscale':
        data = (data - data[start_frame]) / (ss - data[start_frame])
        return data

    if method == 'Double':
        data = data / ss
        return data

    else:
        error_str = 'Method must be one of: ' + ' '.join(METHODS)
        raise ValueError(error_str)

def estimateRadius(frap_image):
    """
    Estimates the nuclear radius
    :param frap_image: an object of class FRAPImage
    :return: radius in um
    """
    try:
        # initilize frame info
        frame = frap_image.get_image_data()
        frame = frame[:, :, 0]
        radius = int(np.ceil(frap_image.get_roi_radii()[0]))
        coords = [(pos[0] + radius, pos[1] + radius) for pos in frap_image.get_viewer_coords()]
        coords = np.around(coords, decimals=0).astype(int)
        coord = tuple(reversed(coords[0]))

        # generate contours
        binary_mask = segmentCell(frame, (0, 0), 0, return_mask=True)
        binary_mask = binary_mask.astype(np.uint8)
        # cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
        contours, im2 = cv2.findContours(binary_mask, 1, 2)

        cnt = contours[0]
        for contour in contours:
            # test to see if FRAP roi is inside the polygon
            inside = cv2.pointPolygonTest(contour,coord,False)
            if inside == 1:
                cnt = contour
                break

        # compute minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(cnt)

        # convert radius to um
        xdim = frap_image.xdim
        physical_size = frap_image.get_physical_size()

        return radius / (float(physical_size[0]) * float(xdim))
    except:
        warnings.warn("Unable to estimate radius.")
        return -1.0

def computeGapRatio(frap_img, start_frame):
    """
    Computes gap ratio from frap_img
    :param frap_img: an object of class FRAPImage
    :param start_frame: the bleaching frame
    :return: gap ratio
    """
    nonbleach_intensities = frap_img.get_corrected_nonbleach_intensities()
    if nonbleach_intensities is not None:
        return 1 - (np.mean(nonbleach_intensities[-10:]) / np.mean(nonbleach_intensities[:start_frame]))

def computeBleachingDepth(data, start_frame, prebleach_ss):
    """
    Computes bleaching depth from frap_img
    :param data: an object of class FRAPImage
    :param prebleach_ss: use the prebleach period as steady state?
    :return:
    """
    if prebleach_ss:
        ss = np.mean(data[:start_frame])

    else:
        ss = np.mean(data[-5:])

    return data[start_frame] / ss

def computeBleachingProfile(data):
    """
    Computes the profile of the bleach as a function of radial distance
    :param data: an object of class FRAPImage
    :return: a vector of intensities
    """
    # get start frame
    start_frame = data.get_start_frame()

    # if start_frame is 0, this will give an error later, so don't allow this
    if start_frame == 0:
        start_frame = 10 # arbitrary number

    # get conversion factors
    xdim = float(data.xdim)
    physical_size = float(data.get_physical_size()[0])
    ss = np.mean(data.get_mean_intensity_data()[:start_frame])

    # get coordinates and radius
    r_coord = data.get_roi_coords()
    radius = data.get_roi_radii()[0]

    X, Y = np.ogrid[0:data.xdim, 0:data.ydim]
    dist_from_center = np.sqrt((X - r_coord[0]) ** 2 + (Y - r_coord[1]) ** 2)
    image_data = data.get_frame(start_frame)

    # subtract bg
    inv_data = np.mean(segmentCell(image_data, (0, 0), 0, inverted=True))
    new_img = np.zeros(image_data.shape)
    for i in range(image_data.shape[-1]):
        new_img = image_data - inv_data
    new_img[new_img < 0] = 0
    image_data = new_img
    distances = np.array([r / (xdim * physical_size) for r in range(3, 4 * int(radius))])
    mean_intensities = np.array([np.mean(image_data[np.logical_and(dist_from_center <= r,dist_from_center >= r-2)]) / ss
                                 for r in range(3, 4 * int(radius))])
    mean_intensities = mean_intensities / np.mean(mean_intensities[-8:])

    return distances, mean_intensities

def photobleachingProfile(x, r_c, sigma, theta):
    """
    Function for photobleaching profile
    :param x: x vals
    :param r_c: uniform radius
    :param theta: bleaching depth
    :param sigma: sd of normal
    :return: function evaluated on x
    """
    data = np.zeros(x.shape)
    data[x <= r_c] = theta
    data[x > r_c] = 1 - (1 - theta) * np.exp(-(x[x > r_c] - r_c)**2 / (2 * sigma**2))
    return data

def fitBleachingProfile(data):
    """
    Fits bleaching profile
    :param data: FRAPImage data
    :return:
    """
    xdim = float(data.xdim)
    physical_size = float(data.get_physical_size()[0])
    radius = data.get_roi_radii()[0] / (xdim * physical_size)
    p0 = (radius, 3 * radius, 0.5)
    distances, intensities = data.get_bleaching_profile()
    popt, pcov = curve_fit(photobleachingProfile, distances, intensities, p0,
                           bounds=([0, 0, 0], [radius, 10 * radius, 1]))
    return popt, pcov


def computeQCMetrics(frap_img, prebleach_ss=True):
    """
    Computes QC metrics gap ratio and bleaching depth on an image
    :param frap_img: an object of class FRAPImage
    :param prebleach_ss: use prebleach as steady state
    :return: gap ratio and bleaching depth
    """
    start_frame = frap_img.get_start_frame()
    return computeGapRatio(frap_img, start_frame), \
           computeBleachingDepth(frap_img.get_mean_intensity_data(), start_frame, prebleach_ss)

def computeBestStartFrame(frap_image):
    """
    Computes the best start frame guess by finding the frame which maximizes the difference
    between itself and the prior frame
    :param frap_image: an object of class FRAPImage
    :return: best start frame guess
    """
    intensities = frap_image.get_mean_intensity_data()
    delta_intensity = [intensities[i+1] - intensities[i] for i in range(len(intensities)-1)]
    return np.argmax(delta_intensity)
