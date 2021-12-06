"""
ImageReader.py - Script that reads in images and relevant metadata
"""
import numpy as np
import pandas as pd
from .ProcessingUtilities import estimateRadius, computeBleachingProfile, \
    fitBleachingProfile, normalizeFRAPCurve, processImage, computeBestStartFrame
from .ReaderUtilities import make_metadata_tree, make_frame_metadata, \
    make_roi, make_dimension_metadata, read_image
from .frapFile import FRAPFile

class FRAPImage:
    """
    This is a class for storing FRAP image data and metadata.
    """

    def __init__(self, path):
        """
        Constructor for FRAPImage class

        :param path: (str) path to image
        """
        # Path and name
        self.path = path
        self.name = path.split('/')[-1]
        print('Reading metadata...')

        # Metadata
        self.metadata_tree = make_metadata_tree(self.path) # contains the original metadata for the image

        self.frame_data = make_frame_metadata(self.path)  # frame data array [index, DeltaT]
        self.shape, self.physical_size = make_dimension_metadata(self.path)
        self.xdim, self.ydim, self.tdim = self.shape
        self.roi_coords, self.roi_radii = make_roi(self.path)
        self.roi_viewer_coords = [(self.roi_coords[0] - self.roi_radii[0],
                                   self.roi_coords[1] - self.roi_radii[1]) for i in range(self.tdim)]
        self.keyframes = {0:self.roi_viewer_coords[0], self.tdim - 1:self.roi_viewer_coords[-1]}

        print('Reading image data...')
        self.image_data = read_image(self.path)
        self.raw_image_data = self.image_data.copy()

        self.set_mean_intensity_data(update=False)

        # Set photobleaching params
        self.photobleaching_params = None

        # Initialize normalization information
        self.normal_method = None
        self.prebleach_ss = None

        # Initialize cell intensity information
        self.nonbleach_intensities = None
        self.corrected_nonbleach_intensities = None

        # Estimate nuclear radius
        self.nuclear_radius = estimateRadius(self)

        # Set initial values for gap ratio and bleaching depth
        self.gap_ratio = -1
        self.bleaching_depth = -1

        # Fix start time
        self.start_frame = computeBestStartFrame(self)
        x_data = self.get_frame_metadata()[:, 1]
        x_data -= x_data[self.start_frame - 1]

        # Compute bleaching profile
        self.bleach_distances, self.bleach_profile = computeBleachingProfile(self)
        self.bleach_profile_popt, self.bleach_profile_pcov = fitBleachingProfile(self)

        # Attach model
        self.Model = None

        # FRAP File
        self.file = FRAPFile(self)
        self.file.update()

    def reset_image_data(self):
        """
        Resets image data
        :return:
        """
        print('\nReading image data...')
        self.image_data = self.raw_image_data

        self.set_mean_intensity_data()
        # Set photobleaching params
        self.photobleaching_params = None

        # Initialize normalization information
        self.normal_method = None
        self.prebleach_ss = None

        # Initialize cell intensity information
        self.nonbleach_intensities = None
        self.corrected_nonbleach_intensities = None
        self.file.update()

    def get_tdim(self):
        """
        Getter for tdim size

        :return: (int) size along t dimension
        """
        return self.tdim

    def get_frame_metadata(self):
        """
        Getter for frame metadata

        :return: (ndarray) has frame index and DeltaT
        """
        return self.frame_data

    def get_frame(self, idx):
        """
        Getter for an individual frame's data

        :param idx: (int) the index of the desired frame
        :return: (ndarray) image data for that frame
        """
        return self.image_data[:, :, idx]

    def get_physical_size(self):
        """
        Getter for image physical size

        :return: (tuple) real X dimension, real Y dimension
        """
        return self.physical_size

    def get_viewer_coords(self):
        """
        Getter for the ROI coordinates for the viewer

        :return: Array of tuples. X coordinate for ROI, Y coordinate for ROI
        """
        return self.roi_viewer_coords

    def get_roi_coords(self):
        """
        Getter for the ROI coordinates

        :return: (tuple) X coordinate for ROI, Y coordinate for ROI
        """
        return self.roi_coords

    def get_roi_radii(self):
        """
        Getter for ROI radii

        :return: (tuple) X radius, Y radius
        """
        return self.roi_radii

    def set_image_data(self, img):
        """
        Setter for image data

        :param img: new image data
        """
        self.image_data = img
        self.set_mean_intensity_data()
        self.file.update()

    def get_image_data(self):
        """
        Getter for image data

        :return: (ndarray) array containing image data
        """
        return self.image_data

    def set_keyframe(self, idx, viewer_coords):
        """
        Sets a new keyframe

        :param idx: index of new keyframe
        :param viewer_coords: coordinates of new keyframe
        :return:
        """
        self.keyframes[idx] = viewer_coords
        self.update_viewer_coords()
        self.file.update()

    def del_keyframe(self, idx):
        """
        Deletes a keyframe

        :param idx: index of keyframe to be deleted
        :return:
        """
        if idx in self.keyframes.keys():
            self.keyframes.pop(idx)
            self.update_viewer_coords()

    def get_keyframes(self):
        """
        Returns a dictionary of keyframes

        :return: (dict) of keyframes
        """
        return self.keyframes

    def update_viewer_coords(self):
        """
        Updates viewer coords and mean intensity data based on current keyframe dictionary
        """
        keyvals = sorted(self.keyframes.items())
        new_coords = []
        next_pos = None
        for i in range(len(keyvals) - 1):
            cur_frame, cur_pos = keyvals[i]
            next_frame, next_pos = keyvals[i+1]
            frame_range = next_frame-cur_frame
            new_coords += list(zip(np.linspace(cur_pos[0], next_pos[0], num=frame_range, endpoint=False),
                                   np.linspace(cur_pos[1], next_pos[1], num=frame_range, endpoint=False)))
        new_coords += [(next_pos[0], next_pos[1])]
        self.roi_viewer_coords = new_coords

        # update mean intensity data
        r_coords = [(pos[0] + self.roi_radii[0], pos[1] + self.roi_radii[1]) for pos in new_coords]
        self.set_mean_intensity_data(r_coords)
        self.file.update()

    def get_mean_intensity_data(self):
        """
        Getter for intensity data

        :return: (list) of mean ROI intensities
        """
        return self.mean_intensity_data

    def set_mean_intensity_data(self, r_coords = None, update=True):
        """
        Setter for mean intensities

        :param r_coords: (list) of coordinates for ROI
        """
        if r_coords is None:
            r_coords = [(pos[0] + self.roi_radii[0], pos[1] + self.roi_radii[1]) for pos in self.roi_viewer_coords]
        mean_intensities = np.empty(self.tdim)
        raw_mean_intensities = np.empty(self.tdim)
        for frame in range(self.tdim):
            X, Y = np.ogrid[0:self.xdim, 0:self.ydim]
            dist_from_center = np.sqrt((X - r_coords[frame][0])**2 + \
                                       (Y - r_coords[frame][1])**2)
            image_data = self.get_frame(frame)
            raw_image_data = self.raw_image_data[:, :, frame]
            mean_intensities[frame] = np.mean(image_data[dist_from_center <= self.roi_radii[0]])
            raw_mean_intensities[frame] = np.mean(raw_image_data[dist_from_center <= self.roi_radii[0]])

        self.raw_mean_intensity_data = raw_mean_intensities
        self.mean_intensity_data = mean_intensities
        if update:
            self.file.update()

    def normalize_frap_curve(self, method, prebleach_ss):
        """
        Normalizes FRAP curve based on the method given
        :param method: method to perform normalization
        :return:
        """
        self.normal_method = method
        self.prebleach_ss = prebleach_ss
        # Fix time 0
        if method == 'Fullscale':
            x_data = self.get_frame_metadata()[:, 1]
            x_data -= x_data[self.start_frame]
        else:
            x_data = self.get_frame_metadata()[:, 1]
            x_data -= x_data[self.start_frame - 1]

        # Do normalization
        self.set_mean_intensity_data()

        self.mean_intensity_data = normalizeFRAPCurve(self.mean_intensity_data, self.start_frame,
                                                      method, prebleach_ss)
        self.file.update()

    def correct_photobleaching(self, subtract_bg):
        """
        Corrects photobleaching
        :param subtract_bg: whether to perform background subtraction as well
        :return:
        """
        print('\nProcessing image...')
        self.photobleaching_params = processImage(self, subtract_bg)
        self.file.update()

    def get_time_intensity_pt(self, idx):
        """
        Get a time intensity value pair

        :param idx: (int) desired frame
        :return: (tuple of floats) time intensity value pair
        """
        return self.frame_data[idx,1], self.mean_intensity_data[idx]

    def attach_model(self, class_name, start_frame = None):
        """
        Attaches a FRAP recovery model to the FRAPImage class instance

        :param class_name: (class) a class name
        :param start_frame: (int or none) frame index of photobleaching
        """
        if start_frame is None:
            start_frame = self.start_frame
        self.Model = class_name(self, start_frame)
        self.Model.fit()
        self.ModelParams = (self.Model.get_parameters(), self.Model.get_cov())
        self.ModelFun = self.Model.func
        self.ModelData = self.Model.get_fit_pts()

    def get_start_frame(self):
        """
        Getter for start_frame

        :return: start_frame
        """
        return self.start_frame

    def set_start_frame(self, new_start_frame):
        """
        Setter for start_frame

        :return:
        """
        self.start_frame = new_start_frame
        # Recompute bleaching profile
        self.bleach_distances, self.bleach_profile = computeBleachingProfile(self)
        self.bleach_profile_popt, self.bleach_profile_pcov = fitBleachingProfile(self)

    def reset_start_frame(self):
        """
        Resets start frame to argmin

        :return: new start_frame
        """
        self.start_frame = np.argmin(self.get_mean_intensity_data())
        return self.start_frame


    def get_model_params(self):
        """
        Getter for model parameters

        :return: model parameters
        """
        return self.ModelParams

    def get_model_fun(self):
        """
        Getter for model function

        :return: model function
        """
        return self.ModelFun

    def get_model_data(self):
        """
        Getter for model data

        :return: Model evaluated at timepoints
        """
        return self.ModelData

    def get_photobleaching_params(self):
        """
        Getter for photobleaching fit

        :return: optimal parameters for photobleaching fit
        """
        return self.photobleaching_params

    def get_gap_ratio(self):
        """
        Getter for the gap ratio

        :return: gap ratio of the experiment
        """
        return self.gap_ratio

    def set_gap_ratio(self, gr):
        """
        Setter for the gap ratio

        :param gr: new gap ratio
        :return:
        """
        self.gap_ratio = gr
        self.file.set_other_metrics()

    def get_bleaching_depth(self):
        """
        Getter for the bleaching depth

        :return: bleaching depth of the experiment
        """
        return self.bleaching_depth

    def set_bleaching_depth(self, bd):
        """
        Setter for the bleaching depth

        :param bd: new bleaching depth value
        :return:
        """
        self.bleaching_depth = bd
        self.file.set_other_metrics()

    def detach_model(self):
        """
        Detaches the FRAP recovery model
        :return:
        """
        self.Model = None
        self.ModelParams = None
        self.ModelFun = None
        self.ModelData = None

    def get_nonbleach_intensities(self):
        """
        Getter for nonbleach intensities
        :return:
        """
        return self.nonbleach_intensities

    def set_nonbleach_intensities(self, intensities):
        """
        Sets the intensity values for the nonbleached areas
        :param intensities: a list of intensity values
        :return:
        """
        self.nonbleach_intensities = intensities
        self.file.update()

    def get_corrected_nonbleach_intensities(self):
        """
        Getter for corrected nonbleach intensities
        :return:
        """
        return self.corrected_nonbleach_intensities

    def set_corrected_nonbleach_intensities(self, intensities):
        """
        Sets the intensity values for the corrected nonbleached areas
        :param intensities: a list of intensity values
        :return:
        """
        self.corrected_nonbleach_intensities = intensities
        self.file.update()

    def get_nuclear_radius(self):
        """
        Getter for the nuclear radius
        :return: nuclear radius in um
        """
        return self.nuclear_radius

    def get_real_roi_radius(self):
        """
        Returns ROI radius in real units
        :return:
        """
        return self.roi_radii[0] / (float(self.physical_size[0]) * float(self.xdim))

    def get_bleaching_profile(self):
        """
        Returns bleaching distances and values
        :return:
        """
        return self.bleach_distances, self.bleach_profile

    def save_data(self, path):
        """
        Saves data in custom xml format for later use/analysis
        :param path: (str) location of file save
        :return:
        """
        self.file.save_file(path)

    def export_to_csv(self, path):
        """
        Exports intensity data as a csv without metdata
        :param path: (str) location of file save
        :return:
        """
        full_data = pd.DataFrame({'time': self.frame_data[:, 1],
                                  'intensity': self.mean_intensity_data,
                                  'intensity_raw':self.raw_mean_intensity_data,
                                  'nonbleach':self.nonbleach_intensities,
                                  'corrected_nonbleach':self.corrected_nonbleach_intensities})
        individual_params = pd.DataFrame({'nuclear_radius':self.nuclear_radius,
                                          'roi_radius':self.get_real_roi_radius(),
                                          'gap_ratio':self.gap_ratio,
                                          'bleaching_depth':self.bleaching_depth,
                                          'radius_uniform':self.bleach_profile_popt[0]}, index=[0])
        bleach_profile = pd.DataFrame({'bleach_distances':self.bleach_distances,
                                       'bleach_profile':self.bleach_profile})
        full_data = pd.concat([full_data, individual_params, bleach_profile], axis=1)
        full_data.to_csv(path)

class UnknownFiletypeError(Exception):
    """
    User defined error for unrecognized filetype
    """

    def __init__(self, filetype):
        self.message = f"Extension {filetype} is not recognized."
        super().__init__(self.message)
