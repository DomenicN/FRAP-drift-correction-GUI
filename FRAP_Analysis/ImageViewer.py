"""
ImageViewer.py - Simple image viewer for FRAP data
Originally Written by: Alec Heckert for package quot
https://github.com/alecheckert/quot
Modified by Domenic Narducci for FRAP
"""

import sys

# Image Reader
from ImageReader import FRAPImage

# Model fitter
import ModelFitter as mf

# Core GUI utilities
from PySide2.QtCore import Qt, QLocale
from PySide2.QtWidgets import QWidget, QGridLayout, \
    QPushButton, QDialog, QLabel, QLineEdit, QShortcut,\
    QApplication, QListWidget, QListWidgetItem, QHBoxLayout, \
    QVBoxLayout, QComboBox, QAction, QMenuBar, QFileDialog, \
    QCheckBox
from PySide2.QtGui import QKeySequence, QFont
from PySide2.QtGui import Qt as QtGui_Qt
from guiUtils import IntSlider, SingleImageWindow, LabeledQComboBox, \
    coerce_type, set_dark_app

# pyqtgraph utilities for showing images
import pyqtgraph
from pyqtgraph import ImageView, CircleROI, PlotWidget

# numpy
import numpy as np

# change list widget item comparison
def _new_lt_(self, other):
    return int(self.text()) < int(other.text())

QListWidgetItem.__lt__ = _new_lt_

MODELS = ["None", "Basic Exponential", "Pure Circular Diffusion",
          "Single Reaction Dominant", "Double Reaction Dominant",
          "Full Single Reaction (Avg)", "Full Double Reaction (Avg)"]

class ImageViewer(QWidget):
    """
    Show a single frame from a movie with a slider
    to change the frame. This essentially harnesses
    pyqtgraph.ImageView for a simple image viewer that is
    occasionally easier than Fiji.
    init
    ----
    """
    def __init__(self, path, parent=None):
        super(ImageViewer, self).__init__(parent=parent)
        self.path = path
        self.initData()
        self.initUI()
        self.createMenu()

        # Resize main window
        self.win.resize(1000, 800)

        # Show the main window
        self.win.show()

    def initData(self):
        """
        Try to read the image data at the target path.
        """
        self.ImageReader = FRAPImage(self.path)

    def initUI(self):
        """
        Initialize the user interface.
        """
        # Main window
        self.win = QWidget()
        self.win.setWindowTitle(self.path)
        layout = QGridLayout(self.win)

        # Image Viewer
        self.initImageViewer(layout)

        # Keyframes
        self.initKeyframeBox(layout)

        # Pre-processing
        self.initProcessingBox(layout)

        # Modeling
        self.initModelingBox(layout)

        # ROI intensity plotting
        self.IntensityPlot = PlotWidget(parent=self.win)
        self.IntensityMarker = self.make_intensity_plot()
        self.IntensityMarker.setZValue(10)
        self.ModelPlot = self.IntensityPlot.plot([],[])
        layout.addWidget(self.IntensityPlot, 3, 0, 1, 1)
        layout.setColumnStretch(0, 2)


    def initImageViewer(self, layout):
        """
        Initializes the image viewer for the UI
        :param layout: layout to initialize image viewer into
        :return:
        """
        # ImageView
        self.ImageView = ImageView(parent=self.win)
        layout.addWidget(self.ImageView, 0, 0, 2, -1)

        # ROI
        coords = self.ImageReader.get_roi_coords()
        radii = self.ImageReader.get_roi_radii()
        coords = (coords[0] - radii[0], coords[1] - radii[1])

        self.ROI = CircleROI(pos=coords,
                             radius=radii[0], resizable=False, rotatable=False)
        self.ImageView.getView().addItem(self.ROI)

        # Frame slider
        self.frame_slider = IntSlider(minimum=0, interval=1,
                                      maximum=self.ImageReader.tdim - 1, init_value=0,
                                      name='Frame', parent=self.win)
        layout.addWidget(self.frame_slider, 2, 0, 1, 2, alignment=Qt.AlignTop)
        self.frame_slider.assign_callback(self.frame_slider_callback)

        # Use the right/left keys to tab through frames
        self.left_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Left), self.win)
        self.right_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Right), self.win)
        self.left_shortcut.activated.connect(self.prev_frame)
        self.right_shortcut.activated.connect(self.next_frame)

        # Update the frame
        self.load_frame(0, reset=True)

    def initKeyframeBox(self, layout):
        """
        Initializes the keyframe box
        :param layout: layout to initialize the keyframe box into
        :return:
        """
        # Make button hbox
        keyframe_hbox = QHBoxLayout()

        # Add keyframe button
        self.KeyframeButton = QPushButton("Add keyframe", self.win)
        keyframe_hbox.addWidget(self.KeyframeButton)
        self.KeyframeButton.clicked.connect(self.keyframe_callback)

        # Add export keyframes button
        self.BleachButton = QPushButton("Set bleach frame", self.win)
        keyframe_hbox.addWidget(self.BleachButton)
        self.BleachButton.clicked.connect(self.bleach_button_callback)
        layout.addLayout(keyframe_hbox, 2, 2, alignment=Qt.AlignRight)

        # Add keyframe list
        self.KeyframeList = QListWidget(parent=self.win)
        keyframes = self.ImageReader.get_keyframes().keys()
        self.KeyframeList.addItems([str(key) for key in keyframes])

        # TODO: Figure out how to sort by number instead of alphabetically
        self.KeyframeList.setSortingEnabled(False)
        # self.KeyframeList.sortItems(Qt.AscendingOrder)
        layout.addWidget(self.KeyframeList, 3, 2, 1, 1)

        # make bleach frame label
        self.startFrameLabel = QLabel()
        self.startFrameLabel.setWordWrap(True)
        self.startFrameLabel.setText("Bleach Frame: {}".format(self.ImageReader.get_start_frame()))
        keyframe_hbox.addWidget(self.startFrameLabel)

        # Use the delete key to remove the selected keyframe
        self.delete_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Backspace), self.win)
        self.delete_shortcut.activated.connect(self.delete_keyframe)

    def createMenu(self):
        """
        Creates menu with options
        :return:
        """
        # init menu bar options
        self.mainMenu = QMenuBar(parent=self.win)
        self.fileMenu = self.mainMenu.addMenu("File")

        # create actions
        openAction = QAction("Open File...", self.win)
        openAction.setShortcut("Ctrl+O")
        openAction.triggered.connect(self.openImage)
        saveAction = QAction("Save Data...", self.win)
        saveAction.setShortcut("Ctrl+S")
        saveAction.triggered.connect(self.saveData)
        exportCsvAction = QAction("Export as csv...", self.win)
        exportCsvAction.triggered.connect(self.exportAsCsv)

        # attach actions
        self.fileMenu.addAction(openAction)
        self.fileMenu.addAction(saveAction)
        self.fileMenu.addAction(exportCsvAction)

    def initProcessingBox(self, layout):
        """
        Initializes the processing box to customize processing features
        :param layout: layout to initialize the processing box into
        :return:
        """

        # make processing hbox
        processing_vbox = QVBoxLayout()

        # make processing header
        processingLabel = QLabel()
        processingLabel.setWordWrap(True)
        processingLabel.setText("Processing")
        processingLabel.setFont(QFont('Arial', 16))
        processing_vbox.addWidget(processingLabel)

        # make processing sub-sections
        processing_hbox = QHBoxLayout()

        ### PHOTOBLEACHING ###
        # make photobleaching subsection
        photobleaching_vbox = QVBoxLayout()
        photobleachingLabel = QLabel()
        photobleachingLabel.setWordWrap(True)
        photobleachingLabel.setText("Photobleaching Correction")
        photobleachingLabel.setFont(QFont('Arial', 14))
        photobleaching_vbox.addWidget(photobleachingLabel)

        # subtract background option
        self.backgroundCheckbox = QCheckBox("Subtract background")
        self.backgroundCheckbox.setChecked(True)
        photobleaching_vbox.addWidget(self.backgroundCheckbox)

        # make photobleaching button
        self.photobleachingButton = QPushButton("Correct photobleaching")
        photobleaching_vbox.addWidget(self.photobleachingButton)
        self.photobleachingButton.clicked.connect(self.photobleaching_callback)
        photobleaching_vbox.addWidget(self.photobleachingButton)

        # make photobleaching label
        self.photobleachingLabel = QLabel()
        self.photobleachingLabel.setWordWrap(True)
        self.photobleachingLabel.setText("Photobleaching t1/2 = ")
        photobleaching_vbox.addWidget(self.photobleachingLabel)

        # make photobleaching variance label
        self.photobleachingVarLabel = QLabel()
        self.photobleachingVarLabel.setWordWrap(True)
        self.photobleachingVarLabel.setText("Fit Variance = ")
        photobleaching_vbox.addWidget(self.photobleachingVarLabel)

        ### NORMALIZATION ###
        # make normalization subsection
        normalization_vbox = QVBoxLayout()
        normalizationLabel = QLabel()
        normalizationLabel.setWordWrap(True)
        normalizationLabel.setText("Normalization")
        normalizationLabel.setFont(QFont('Arial', 14))
        normalization_vbox.addWidget(normalizationLabel)

        # make normalization combobox
        self.NormalBox = QComboBox()
        self.NormalBox.addItem('Fullscale')
        self.NormalBox.addItem('Double')
        normalization_vbox.addWidget(self.NormalBox)

        # steady state option
        self.steadyStateCheckbox = QCheckBox("Use pre-bleach as steady-state")
        self.steadyStateCheckbox.setChecked(True)
        normalization_vbox.addWidget(self.steadyStateCheckbox)

        # make photobleaching button
        self.normalizationButton = QPushButton("Normalize")
        normalization_vbox.addWidget(self.normalizationButton)
        self.normalizationButton.clicked.connect(self.normalization_callback)
        normalization_vbox.addWidget(self.normalizationButton)

        photobleaching_vbox.addStretch()
        normalization_vbox.addStretch()
        processing_hbox.addLayout(photobleaching_vbox)
        processing_hbox.addLayout(normalization_vbox)
        processing_vbox.addLayout(processing_hbox)
        processing_vbox.addStretch()
        layout.addLayout(processing_vbox, 4, 0, 1, 1)

    def initModelingBox(self, layout):
        """
        Initializes the modeling box to customize processing features
        :param layout: layout to initialize the modeling box into
        :return:
        """

        # make processing hbox
        modeling_vbox = QVBoxLayout()

        # make processing header
        modelingLabel = QLabel()
        modelingLabel.setWordWrap(True)
        modelingLabel.setText("Modeling")
        modelingLabel.setFont(QFont('Arial', 16))

        modeling_vbox.addWidget(modelingLabel)

        # Model fit combo box
        self.ModelBox = QComboBox(self.win)
        for model in MODELS:
            self.ModelBox.addItem(model)
        modeling_vbox.addWidget(self.ModelBox)

        # Fit hbox
        fit_hbox = QHBoxLayout()

        # Fit label
        self.ModelParamLabel = QLabel()
        self.ModelParamLabel.setWordWrap(True)
        fit_hbox.addWidget(self.ModelParamLabel)

        # Fit button
        self.ModelFitButton = QPushButton("Fit Model", self.win)
        self.ModelFitButton.clicked.connect(self.fit_model)
        fit_hbox.addWidget(self.ModelFitButton)
        modeling_vbox.addLayout(fit_hbox)
        modeling_vbox.addStretch()
        layout.addLayout(modeling_vbox, 4, 2, 1, 1)

    def openImage(self):
        """
        Opens a file browser
        :return:
        """
        filename = QFileDialog.getOpenFileName(parent=self.win, caption="Open file",
                                               filter="Image files (*.czi)")
        self.path = filename[0]
        self.refresh()

    def saveData(self):
        """
        Opens a file browser to save data
        :return:
        """
        filename = QFileDialog.getSaveFileName(parent=self.win, caption="Save file",
                                               filter="(*.frap)")
        self.ImageReader.save_data(filename[0])

    def exportAsCsv(self):
        """
        Opens a file browser to export as csv
        :return:
        """
        filename = QFileDialog.getSaveFileName(parent=self.win, caption="Save file",
                                               filter="(*.csv)")
        self.ImageReader.export_to_csv(filename[0])

    def refresh(self):
        """
        Refreshes imageviewer, keyframes, model plot, and slider
        :return:
        """
        self.win.setWindowTitle(self.path)
        self.initData()
        self.load_frame(0, reset=True)
        keyframes = self.ImageReader.get_keyframes().keys()
        self.KeyframeList.clear()
        self.KeyframeList.addItems([str(key) for key in keyframes])
        # update plot
        self.update_plot()

        # delete label
        self.ModelParamLabel.setText("")

        # reset int slider
        self.frame_slider.setMax(self.ImageReader.tdim - 1)

        # reset bleach frame
        self.startFrameLabel.setText("Bleach Frame: {}".format(self.ImageReader.get_start_frame()))

    def update_plot(self):
        """
        Updates the plot
        :return:
        """

        # update plot
        self.IntensityMarker = self.make_intensity_plot()
        self.ModelPlot.clear()
        self.ModelPlot = self.IntensityPlot.plot([], [])

    def make_intensity_plot(self):
        """
        Sets all parameters to initialize ROI intensity plot
        :return: Returns plot object for current frame marker
        """

        time = self.ImageReader.get_frame_metadata()[:, 1]
        time -= time[self.ImageReader.get_start_frame()]

        intensity = self.ImageReader.get_mean_intensity_data()

        self.IntensityPlot.clear()
        self.IntensityPlot.plot(time, intensity)
        self.IntensityPlot.setLabel('left', 'Mean ROI Intesity')
        self.IntensityPlot.setLabel('bottom', 'Time', units='s')
        self.IntensityPlot.setXRange(min(time), max(time))
        self.IntensityPlot.setYRange(min(intensity), max(intensity))

        # current frame marker
        current_frame = self.frame_slider.value()
        intensityMarker = self.IntensityPlot.plot([time[current_frame]],
                                       [intensity[current_frame]], pen = None, symbol = '+')
        return intensityMarker

    def load_frame(self, frame_index, reset=False):
        """
        Change the current frame.
        args
        ----
            frame_index     :   int
            reset           :   bool, reset the LUTs and ROI
        """
        self.image = self.ImageReader.get_frame(frame_index)
        self.ImageView.setImage(self.image, autoRange=reset, autoLevels=reset,
            autoHistogramRange=reset)
        self.ROI.setPos(self.ImageReader.get_viewer_coords()[frame_index])

    def load_plot_marker(self, frame_index):
        """
        Changes the location of the marker on the plot.
        :param frame_index: (int) frame index
        """
        tpt, ipt = self.ImageReader.get_time_intensity_pt(frame_index)
        self.IntensityMarker.setData([tpt],[ipt])

    def next_frame(self):
        """
        Go to the frame after the current one.
        """
        next_idx = int(self.frame_slider.value())
        if next_idx < self.frame_slider.maximum:
            next_idx += 1
        self.frame_slider.setValue(next_idx)

    def prev_frame(self):
        """
        Go the frame before the current one.
        """
        prev_idx = int(self.frame_slider.value())
        if prev_idx > self.frame_slider.minimum:
            prev_idx -= 1
        self.frame_slider.setValue(prev_idx)

    def delete_keyframe(self):
        """
        Delete the selected keyframe
        """
        if self.KeyframeList.currentItem() is not None:
            currentIdx = int(self.KeyframeList.currentItem().text())
            if currentIdx != 0 and currentIdx != self.ImageReader.get_tdim()-1:
                # delete keyframe
                self.ImageReader.del_keyframe(currentIdx)
                self.KeyframeList.takeItem(self.KeyframeList.currentRow())

                # update plot
                self.IntensityMarker = self.make_intensity_plot()

                # update image
                self.load_frame(self.frame_slider.value())

    def frame_slider_callback(self):
        """
        Change the current frame.
        """
        self.load_frame(self.frame_slider.value())
        self.load_plot_marker(self.frame_slider.value())

    def keyframe_callback(self):
        """
        Add a keyframe for the ROI at the current frame and ROI position
        """
        # update keyframes in image reader
        self.ImageReader.set_keyframe(self.frame_slider.value(),
                                      tuple(self.ROI.pos()))

        # update keyframe list
        keyframe_items = [int(self.KeyframeList.item(i).text()) \
                          for i in range(self.KeyframeList.count())]
        if self.frame_slider.value() not in keyframe_items:
            self.KeyframeList.addItem(str(self.frame_slider.value()))
            #self.KeyframeList.sortItems(Qt.AscendingOrder)

        # detach model
        self.ImageReader.detach_model()

        # update plot
        self.update_plot()

        # delete label
        self.ModelParamLabel.setText("")

    def bleach_button_callback(self):
        """
        Set's the start frame for bleaching
        """
        # TODO: Check this
        bleach_frame = self.frame_slider.value()
        self.ImageReader.set_start_frame(bleach_frame)
        self.startFrameLabel.setText("Bleach Frame: {}".format(self.ImageReader.get_start_frame()))
        # update plot
        self.update_plot()


    def photobleaching_callback(self):
        """
        Corrects photobleaching/handles pressing of button
        :return:
        """
        self.ImageReader.correct_photobleaching(self.backgroundCheckbox.isChecked())

        # reload frames
        self.load_frame(0, reset=True)

        # update plot
        self.update_plot()

        # delete label
        self.ModelParamLabel.setText("")

        # get photobleaching params tau and tau variance
        tau, tau_var = self.ImageReader.get_photobleaching_params()
        t_half = 0.693147 / tau[0]
        self.photobleachingLabel.setText("Photobleaching t1/2 = {:.3g} [1/s]".format(t_half))
        self.photobleachingVarLabel.setText("Fit Variance = {:.3g}".format(*tau_var[0]))

    def normalization_callback(self):
        """
        Normalizes the FRAP curve
        :return:
        """
        current_selection = str(self.NormalBox.currentText())
        prebleach_ss = self.steadyStateCheckbox.isChecked()
        self.ImageReader.normalize_frap_curve(current_selection, prebleach_ss)

        # reload frames
        self.load_frame(0, reset=True)

        # update plot
        self.update_plot()

        # delete label
        self.ModelParamLabel.setText("")

    def fit_model(self):
        """
        Attaches model, fits it and displays it
        """
        SELECTIONS = {"Basic Exponential":mf.BasicExponential,
                      "Pure Circular Diffusion":mf.PureDiffusion,
                      "Single Reaction Dominant":mf.OneReaction,
                      "Double Reaction Dominant":mf.TwoReaction,
                      "Full Single Reaction (Avg)":mf.FullOneReactionAverage,
                      "Full Double Reaction (Avg)":mf.FullTwoReactionAverage}
        LABELS = {"Basic Exponential":"A = {}\nTau = {}",
                  "Pure Circular Diffusion":"t_d = {}\nD = {}",
                  "Single Reaction Dominant":"k_off = {}\nk_on_* = {}",
                  "Double Reaction Dominant":"k_off1 = {}\nk_off2 = {}\nk_on1_* = {}\n k_on2_* = {}",
                  "Full Single Reaction (Avg)":"k_on = {}\nk_off = {}\nD_f = {}",
                  "Full Double Reaction (Avg)":"k_on_1 = {}\nk_on_2 = {}\nk_off_1 = {}\nk_off_2 = {}\nD_f = {}"}
        current_selection = str(self.ModelBox.currentText())
        if current_selection == "None":
            self.ImageReader.detach_model()
            self.ModelPlot.clear()
            self.ModelPlot = self.IntensityPlot.plot([],[])
            self.ModelParamLabel.setText("")
        else:
            model_pen = pyqtgraph.mkPen('c', width=4)
            self.ImageReader.attach_model(SELECTIONS.get(current_selection))
            self.ModelPlot.clear()
            self.ModelPlot = self.IntensityPlot.plot(*self.ImageReader.get_model_data(),
                                                     pen=model_pen)
            label_text = LABELS.get(current_selection)
            param_vals = self.ImageReader.get_model_params()[0]
            param_vals = [round(val, 4) for val in param_vals]
            self.ModelParamLabel.setText(label_text.format(*param_vals))


    def B_max_int_callback(self):
        """
        Make a maximum intensity projection.
        """
        ex = ChooseProjectionDialog(self.ImageReader.tdim, parent=self)
        if ex.exec_() == QDialog.Accepted:
            method, start_frame, stop_frame = ex.return_val

            # Perform the projection
            result = getattr(self.ImageReader, method)(start=int(start_frame),
                stop=int(stop_frame))

            # Make a standalone window showing the projection
            ex = SingleImageWindow(result, title=method, parent=self)
            ex.show()

class ChooseProjectionDialog(QDialog):
    def __init__(self, n_frames, parent=None):
        super(ChooseProjectionDialog, self).__init__(parent=parent)
        self.n_frames = n_frames
        self.initUI()

    def initUI(self):
        layout = QGridLayout(self)
        self.setWindowTitle("Select projection")

        # Menu to select type of projection
        proj_types = ['max_int_proj', 'sum_proj']
        self.M_proj = LabeledQComboBox(proj_types, "Projection type",
            init_value="max_int_proj", parent=self)
        layout.addWidget(self.M_proj, 0, 0, 1, 2, alignment=Qt.AlignRight)

        # Entry boxes to choose start and stop frames
        label_0 = QLabel(self)
        label_1 = QLabel(self)
        label_0.setText("Start frame")
        label_1.setText("Stop frame")
        layout.addWidget(label_0, 1, 0, alignment=Qt.AlignRight)
        layout.addWidget(label_1, 2, 0, alignment=Qt.AlignRight)

        self.EB_0 = QLineEdit(self)
        self.EB_1 = QLineEdit(self)
        self.EB_0.setText(str(0))
        self.EB_1.setText(str(self.n_frames))
        layout.addWidget(self.EB_0, 1, 1, alignment=Qt.AlignLeft)
        layout.addWidget(self.EB_1, 2, 1, alignment=Qt.AlignLeft)

        # Accept button
        self.B_accept = QPushButton("Accept", parent=self)
        self.B_accept.clicked.connect(self.B_accept_callback)
        layout.addWidget(self.B_accept, 3, 0, alignment=Qt.AlignRight)

    def B_accept_callback(self):
        """
        Accept the current projection settings and return to the
        client widget.
        """
        try:
            self.return_val = [
                self.M_proj.currentText(),
                coerce_type(self.EB_0.text(), int),
                coerce_type(self.EB_1.text(), int),
            ]
            self.accept()
        except ValueError:
            print("Frame values must be integers")


def launch(path):
    pyqtgraph.Qt.QT_LIB = "PySide2"
    app = QApplication([])
    set_dark_app(app)
    instance = ImageViewer(path)
    sys.exit(app.exec_())
