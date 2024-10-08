import re
import os
import sys
from pathlib import Path
import time
from math import floor, ceil

import tifffile
import numpy as np
import datetime
from scipy.ndimage import median_filter
from scipy.ndimage import shift
from scipy.ndimage import affine_transform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from silx.gui import qt
from silx.gui.plot.items import roi as roi_items
from silx.gui.utils.concurrent import submitToQtMainThread as _submit
import h5py

# from utils import fitPeak
from utils import magnification_corr_factors, find_nearest

from txm_pal_core import quadfit_mc, gaussianfit_mc, phase_cross_correlation_stack, renormalize_absorbance_stack
from roiTableWidget import RoiTableWidget
from widgets import MainToolBar

BASE_PATH = os.path.expanduser('~')

from PyQt5 import QtWidgets, QtCore, QtGui

if os.name == 'nt':
    # Enable highdpi scaling
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    # Use highdpi icons
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

if getattr(sys, 'frozen', False):
    # in exe file
    application_path = sys._MEIPASS
else:
    # started in script
    application_path = os.path.dirname(os.path.abspath(__file__))

ui_path = os.path.join(application_path, 'ui/main.ui')

class Main(qt.QMainWindow):
    """Main Window"""

    hided = qt.Signal(object)
    closed = qt.Signal(object)

    sigStop = qt.Signal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.basename = ""
        self.absorbanceImage = None
        self._preprocessImage = None
        self.backImage = None
        self.projImage = None
        self.middle_index = 0
        self.image_shifts = []
        self.thickness_image = None
        self.concentration_image = None
        self.selected_mask_path = ""
        self.selected_path = ""
        self.selected_path_h5 = ""
        self.event_queue = []
        self.processing = False
        self.crop_info = {}

        qt.loadUi(ui_path, self)

        self.setWindowTitle("TXM-Pal")
        self.widgetImageStack.getPlotWidget().setDataBackgroundColor([0,0,0])

        # Setup toolbar
        self.toolbar = MainToolBar(self)
        self.toolbar.sigSetPannel.connect(self.setPannel)
        self.addToolBar(qt.Qt.TopToolBarArea, self.toolbar)

        # Shift Plot
        # self.widgetPlotShift.setGraphTitle("Shift")
        self.widgetPlotShift.setGraphXLabel("Energy (eV)")
        self.widgetPlotShift.setGraphYLabel("Shift (pixel)")
        self.widgetPlotShift.setBackgroundColor('#FCF9F6')
        self.widgetPlotShift.setDataMargins(0.01, 0.01, 0.01, 0.01)
        self.widgetPlotShift.setAxesMargins(0.1, 0.05, 0.05, 0.15)

        # Spectrum Plot
        self.widgetPlotSpectrum.setGraphXLabel("Energy (eV)")
        self.widgetPlotSpectrum.setGraphYLabel("Intensity (a.u.)")
        self.widgetPlotSpectrum.getLegendsDockWidget().show()
        self.widgetPlotSpectrum.setBackgroundColor('#FCF9F6')
        self.widgetPlotSpectrum.setDataMargins(0.01, 0.01, 0.01, 0.01)
        self.widgetPlotSpectrum.setAxesMargins(0.15, 0.05, 0.05, 0.1)

        # Histogram Plot
        self.widgetPlotHistogram.setGraphXLabel("Energy (eV)")
        self.widgetPlotHistogram.setGraphYLabel("Ratio")
        self.widgetPlotHistogram.setBackgroundColor('#FCF9F6')
        self.widgetPlotHistogram.setDataMargins(0.01, 0.01, 0.01, 0.01)
        self.widgetPlotHistogram.setAxesMargins(0.12, 0.05, 0.05, 0.15)

        self.pushButtonHistogram.setCallable(self.plotHistogram)

        # Datafile selection
        self.pushButtonSelectPath.clicked.connect(self.select_load_path)
        self.pushButtonLoad.setCallable(self.load)
        self.pushButtonSelectSavePath.clicked.connect(self.select_save_path)

        # Load result file
        self.pushButtonSelectPathH5.clicked.connect(self.select_load_path_h5)
        self.pushButtonLoadH5.setCallable(self.load_h5)

        self.checkBoxAbsorbance.clicked.connect(self.reloadDisplay)
        self.checkBoxBackground.clicked.connect(self.reloadDisplay)
        self.checkBoxRawImage.clicked.connect(self.reloadDisplay)

        self.checkBoxROI.stateChanged.connect(self.toggleROI)

        self.widgetImageStack.sigEnergyKeV.connect(self.updateEnergy)

        self.pushButtonFiltering.setCallable(self.applyFiltering)
        self.pushButtonMagCorr.setCallable(self.magnificationCorrection)
        self.pushButtonAlign.setCallable(self.alignImages)
        self.pushButtonAuto.setCallable(self.doAuto)

        # ROI Table
        self.roiTableWidget = RoiTableWidget(plot=self.widgetImageStack.getPlotWidget())
        self.roiTableWidget.sigClearRoiCurves.connect(self.widgetPlotSpectrum.clearCurves)

        layout = self.widgetRoiTableHolder.parent().layout()
        layout.replaceWidget(self.widgetRoiTableHolder, self.roiTableWidget)
        self.roiManager = self.roiTableWidget.roiManager

        self.roiManager.sigRoiAboutToBeRemoved.connect(self.removeRoiSpectrum)

        # Initial creation of curve
        self.roiManager.sigInteractiveRoiFinalized.connect(self.updateRoiSpectrum)

        # Update ROI spectrum
        self.roiManager.sigUpdatedRoi.connect(self.updateRoiSpectrum)

        # ROI for cropping
        imgWidget = self.widgetImageStack.getPlotWidget()
        imgWidget.sigRoiUpdated.connect(self.updateRoi)

        # Use button for updating ROI
        self.pushButtonROIUpdate.clicked.connect(self.updateRoiFromValue)

        # Mask Tools
        self.maskToolsWidget = imgWidget.getMaskToolsDockWidget().widget()
        self.maskToolsWidget.otherToolGroup.hide()
        self.frameMask.layout().addWidget(self.maskToolsWidget)

        self.pushButtonCrop.setCallable(self.cropImage)

        self.pushButtonThickness.setCallable(self.calcThickness)
        # self.pushButtonSelectMask.clicked.connect(self.select_mask)
        self.pushButtonFitting.setCallable(self.calcPeakFitting)
        self.pushButtonConcentration.setCallable(self.calcConcentration)
        self.pushButtonConcentration.setCallable(self.calcConcentration)
        self.pushButtonConcentrationPlus.setCallable(self.calcConcentrationPlus)
        self.pushButtonConcentrationMinus.setCallable(self.calcConcentrationMinus)

        self.pushButtonSaving.clicked.connect(self.saveData)

        # Interpolation option
        self.comboBoxInterpolate.currentIndexChanged.connect(self.updateInterpolate)

        # Fitting algorithm selection
        self.comboBoxSmoothAlgorithm.currentTextChanged.connect(self.updateSmoothAlgorithm)

        # Adjust splitter ratio
        self.mainSplitter.setSizes([800, 800])

        # Disable zoom on wheel
        self.widgetPlotSpectrum.setInteractiveMode("zoom", zoomOnWheel=False)
        self.widgetPlotHistogram.setInteractiveMode("zoom", zoomOnWheel=False)
        self.widgetPlotShift.setInteractiveMode("zoom", zoomOnWheel=False)

        # Horizontal slider
        self.horizontalSliderAdj.setRange(0, 100)
        self.horizontalSliderAdj.valueChanged.connect(self.updateAdjLabel)

        # Set window icon
        icon_path = os.path.join(application_path, 'mainicon.ico')
        self.setWindowIcon(qt.QIcon(icon_path))

    def updateInterpolate(self, idx):
        qsettings = qt.QSettings('settings.ini', qt.QSettings.IniFormat)
        qsettings.setValue('interpolate', idx)
        qsettings.sync()

    def doAuto(self):
        filtering = self.checkBoxAutoFilter.isChecked()
        magCorr = self.checkBoxAutoMagCorr.isChecked()
        align = self.checkBoxAutoAlign.isChecked()
        numAlign = self.spinBoxAutoAlignNum.value()

        if filtering:
            self.applyFiltering()

        if magCorr:
            self.magnificationCorrection()

        if align:
            for idx in range(numAlign):
                self.toLog(f"{idx+1} / {numAlign} aligning...")
                if idx == numAlign-1:
                    notify = True
                else:
                    notify = False
                self.alignImages(notify=notify)

    def updateAdjLabel(self, value):
        lbl = f"{value}"
        _submit(self.labelThicknessAdj.setText, lbl)

    def setPannel(self, pannel):
        if pannel == "Preprocessing":
            _submit(self.controlStackedWidget.setCurrentIndex, 0)
        elif pannel == "Fitting":
            _submit(self.controlStackedWidget.setCurrentIndex, 1)
        elif pannel == "Spectrum Viewer":
            _submit(self.controlStackedWidget.setCurrentIndex, 2)

    def updateSmoothAlgorithm(self, algorithm):
        if algorithm == "3point":
            _submit(self.spinBoxWindowLength.setEnabled, True)
            _submit(self.spinBoxWindowLength.setValue, 3)
            _submit(self.spinBoxPolyorder.setEnabled, False)
            _submit(self.labelParam1.setText, "Iteration")
        elif algorithm == "boxcar":
            _submit(self.spinBoxWindowLength.setEnabled, True)
            _submit(self.spinBoxWindowLength.setValue, 3)
            _submit(self.spinBoxPolyorder.setEnabled, False)
            _submit(self.labelParam1.setText, "Window Length")
        else:
            _submit(self.labelParam1.setText, "Window Length")
            _submit(self.spinBoxWindowLength.setValue, 7)
            _submit(self.spinBoxPolyorder.setValue, 3)
            _submit(self.spinBoxWindowLength.setEnabled, True)
            _submit(self.spinBoxPolyorder.setEnabled, True)

    def plotHistogram(self):
        refEnergy = self.doubleSpinBoxRefEnergy.value()
        energyRange = self.doubleSpinBoxEnergyRange.value()
        energyStep = self.doubleSpinBoxEnergyStep.value()
        energyStart = refEnergy - energyRange
        energyStop = refEnergy + energyRange
        numBins = int(energyRange*2/energyStep)

        data = self.peak_image.copy()
        hist = np.histogram(data,
                            bins=numBins,
                            range=(energyStart, energyStop),
                            density=True)
        self.histogram = hist
        _submit(self.widgetPlotHistogram.addHistogram, hist[0], hist[1])

    def getSpectrum(self, roi):
        """Get spectrum from ROI"""
        ref_size = self.absorbanceImage[0].shape
        if isinstance(roi, roi_items.PointROI):
            position = roi.getPosition()
            if position[0] < 0 or position[1] < 0 or position[0] > ref_size[0] -1 or position[1] > ref_size[1] - 1:
                self.toLog("out of range!", "red")
            spectrum = self.absorbanceImage[:, floor(position[1]), floor(position[0])]

        elif isinstance(roi, roi_items.CircleROI):
            center = roi.getCenter()
            radius = roi.getRadius()
            ref_image = self.absorbanceImage[0]

            mask = np.zeros_like(ref_image, dtype=bool)
            xStart = floor(center[0] - radius)
            xStop = ceil(center[0] + radius)
            yStart = floor(center[1] - radius)
            yStop = ceil(center[1] + radius)

            for yIdx in range(yStart, yStop):
                for xIdx in range(xStart, xStop):
                    if (xIdx - center[0])**2 + (yIdx - center[1])**2 < radius**2:
                        if yIdx < mask.shape[0] and xIdx < mask.shape[1] and yIdx > 0 and xIdx > 0:
                            mask[yIdx, xIdx] = True
            # counts for average
            counts = np.sum(mask)
            mask = np.invert(mask)
            maskArray = np.array([mask for _ in range(len(self.energy_list))])
            maskedData = np.ma.masked_array(self.absorbanceImage, mask=maskArray)
            spectrum = np.sum(maskedData, axis=(1, 2)) / counts

        elif isinstance(roi, roi_items.RectangleROI):
            origin = roi.getOrigin()
            size = roi.getSize()
            xStart = floor(origin[0])
            xStop = floor(origin[0] + size[0])
            yStart = floor(origin[1])
            yStop = floor(origin[1] + size[1])

            ref_image = self.absorbanceImage[0]
            mask = np.zeros_like(ref_image, dtype=bool)
            for yIdx in range(yStart, yStop):
                for xIdx in range(xStart, xStop):
                    if yIdx > 0 and yIdx < ref_size[0] and xIdx > 0 and xIdx < ref_size[1]:
                        mask[yIdx, xIdx] = True

            # counts for average
            counts = np.sum(mask)
            mask = np.invert(mask)
            maskArray = np.array([mask for _ in range(len(self.energy_list))])
            maskedData = np.ma.masked_array(self.absorbanceImage, mask=maskArray)
            spectrum = np.sum(maskedData, axis=(1, 2)) / counts

        else:
            return

        return spectrum

    def removeRoiSpectrum(self, roi):
        roi_name = roi.getName()
        _submit(self.widgetPlotSpectrum.removeCurve, roi_name)
        self.toLog(f"ROI removed : {roi_name}")

    def updateRoiSpectrum(self, roi_updated):
        self.toLog("Updating ROI spectrum...")
        rois = self.roiManager.getRois()

        for roi in rois:
            roi_name = roi.getName()
            if roi_updated.getName() == roi.getName():
                roi_name = roi.getName()

                try:
                    spectrum = self.getSpectrum(roi)
                except Exception as e:
                    self.toLog(f"Err : {e}", "red")
                    return

                curve = self.widgetPlotSpectrum.getCurve(roi_name)

                if curve:
                    oldSpectrum = curve.getYData()
                    if np.any(oldSpectrum != spectrum):
                        _submit(self.widgetPlotSpectrum.addCurve,
                                self.energy_list,
                                spectrum,
                                legend=roi_name)

                else:
                    rois = self.roiManager.getRois()
                    all_rois = [roi.getName() for roi in rois]

                    if roi_name in all_rois:
                        _submit(self.widgetPlotSpectrum.addCurve,
                                self.energy_list,
                                spectrum,
                                legend=roi_name)

            # cleanup curves
            curves = self.widgetPlotSpectrum.getAllCurves()
            rois = self.roiManager.getRois()
            all_rois = [roi.getName() for roi in rois]
            for curve in curves:
                if curve.getLegend() not in all_rois:
                    _submit(self.widgetPlotSpectrum.removeCurve, curve.getLegend())
            self.toLog("Updating ROI spectrum... done")

    def toggleROI(self, state):
        plot = self.widgetImageStack.getPlotWidget()
        if state == qt.Qt.Checked:
            xLimits = plot.getGraphXLimits()
            yLimits = plot.getGraphYLimits()

            xSize = (xLimits[1] - xLimits[0]) * 0.15
            ySize = (yLimits[1] - yLimits[0]) * 0.15
            roiSize = (xSize+ySize) // 2

            centerX = (xLimits[1] + xLimits[0]) // 2
            centerY = (yLimits[1] + yLimits[0]) // 2

            origin = (centerX-roiSize//2, centerY-roiSize//2)
            size = (roiSize, roiSize)

            # Update roi start, stop position
            self.updateRoi(origin, size)

            # Relocate and resize ROI
            roi = plot.getRoi()
            _submit(roi.setOrigin, origin)
            _submit(roi.setSize, size)
            _submit(plot.toggleROI, True)

        else:
            _submit(plot.toggleROI, False)

    def magnificationCorrection(self):
        if self.absorbanceImage is not None:
            self.toLog("Correcting magnification...")
            corr_factors = magnification_corr_factors(self.energy_list.copy())
            for idx, img in enumerate(self.absorbanceImage):
                cf = corr_factors[idx]
                scaleMat = np.array([[cf,0,0], [0, cf, 0], [0, 0, 1]])
                self.absorbanceImage[idx] = affine_transform(img, scaleMat)

            self._preprocessImage = self.absorbanceImage.copy()
            self.reloadDisplay()
            self.toLog("Correcting magnification... done")
        else:
            self.toLog("Please load images first", "red")

    def saveData(self):
        save_path = self.lineEditSavePath.text()

        if save_path == "":
            self.toLog("Please select save path")
            return

        # Create directory if not exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        re_pat = re.compile(r"[0-9]+")
        last_num = -1

        with os.scandir(save_path) as it:
            for item in it:
                if item.is_file() and item.name.find(self.basename) >= -1:
                    idx = item.name[:-3].split('_result_')[-1]
                    if re_pat.match(idx):

                        try:
                            if int(idx) > last_num:
                                last_num = int(idx)
                        except:
                            ...
            num = last_num + 1

        save_file = os.path.join(save_path, f"{self.basename}_result_{num:d}.h5")

        try:
            with h5py.File(save_file, "w") as f:
                entry_1 = f.create_group("entry_1")
                entry_1.create_dataset("energies", data=self.energy_list)
                entry_1.create_dataset("absorbance", data=self._preprocessImage)
                entry_1.create_dataset("projection", data=self.projImage)
                entry_1.create_dataset("background", data=self.backImage)
                entry_1.create_dataset("crop_xstart", data=self.crop_info.get("xStart", 0))
                entry_1.create_dataset("crop_xstop", data=self.crop_info.get("xStop", 0))
                entry_1.create_dataset("crop_ystart", data=self.crop_info.get("yStart", 0))
                entry_1.create_dataset("crop_ystop", data=self.crop_info.get("yStop", 0))

                # entry_1.create_dataset("thickness", data=self.thickness_image)
                # entry_1.create_dataset("concentration", data=self.concentration_image)
                # entry_1.create_dataset("peak", data=self.peak_image)
                # entry_1.create_dataset("peak_energy_mean", data=self.peak_energy_mean)
                # entry_1.create_dataset("peak_energy_std", data=self.peak_energy_std)
                # entry_1.create_dataset("histogram_bins", data=self.histogram[1])
                # entry_1.create_dataset("histogram_count", data=self.histogram[0])

        except Exception as e:
            self.toLog(f"Error : {e}")
            return

        self.toLog(f"Result saved to {save_file}")

    def calculate_concentration(self, startE, stopE):
        self.toLog("Calculating concentration...")
        conc = (self.peak_image - startE) / (stopE - startE)
        thickness_img = self.thickness_image.copy()

        negative_mask = np.logical_and(conc<0 , np.logical_not(np.isnan(conc)))
        ceil_mask = np.logical_and(conc>1.0, np.logical_not(np.isnan(conc)))

        conc[negative_mask] = 0
        conc[ceil_mask] = 1

        shape = conc.shape

        slider_value = self.horizontalSliderAdj.value()
        if slider_value == 100:
            # use the maximum thickness as it is
            max_value = np.nanmax(thickness_img)
        else:
            # Calculate the new maximum value based on the slider's value
            max_value = np.nanmax(thickness_img) * (slider_value / 100.0)

        # Adjust the image's maximum value
        adjusted_img = np.minimum(thickness_img, max_value)

        # Normalize the image
        normalized_img = adjusted_img / max_value

        if not self.groupBoxColormap.isChecked():
            # Fill H, S, V channels
            hsv = np.zeros((shape[0], shape[1], 3))
            hsv[:, :, 0] = conc * (1/3)
            hsv[:, :, 1] = np.ones_like(conc)
            hsv[:, :, 2] = normalized_img

            # Convert HSV to RGB
            rgb = hsv_to_rgb(hsv)
        else:
            low = self.widgetColormap.minDoubleSpinBox.value()
            high = self.widgetColormap.maxDoubleSpinBox.value()
            cmap_text = self.widgetColormap.colorComboBox.currentText()
            cmap = matplotlib.colormaps[cmap_text]

            # Renormalize data from low to high
            conc = low + conc * (high - low)

            # Convert to RGBA
            rgb = cmap(conc)
            rgb[:,:,3] = normalized_img
            rgb[:,:,3][np.where(np.isnan(conc))] = 1

        # Remove Unmasked Pixels
        rgb[:,:,0][np.where(np.isnan(conc))] = 0
        rgb[:,:,1][np.where(np.isnan(conc))] = 0
        rgb[:,:,2][np.where(np.isnan(conc))] = 0

        # Clip RGB values
        rgb = np.clip(rgb, 0, 1)
        self.concentration_image = rgb

        plot = self.widgetImageStack.getPlotWidget()
        _submit(plot.addImage, self.concentration_image)
        title = f"{self.basename}, mean : {self.peak_energy_mean:.3f}, std : {self.peak_energy_std:.3f}\nstartE : {startE:.2f}, stopE : {stopE:.2f}, size : {self.concentration_image.shape[1]} x {self.concentration_image.shape[0]}"
        _submit(plot.setGraphTitle, title)
        self.toLog("Calculating concentration... done")

    def calcConcentration(self):
        startE = self.doubleSpinBoxStartE.value()
        stopE = self.doubleSpinBoxStopE.value()
        self.calculate_concentration(startE, stopE)

    def calcConcentrationPlus(self):
        centerE = self.doubleSpinBoxCenterE.value()
        stepE = self.doubleSpinBoxStepE.value()
        plusE = self.doubleSpinBoxPlusE.value()
        minusE = self.doubleSpinBoxMinusE.value()

        newMinusE = minusE - stepE
        newPlusE = plusE + stepE

        _submit(self.doubleSpinBoxPlusE.setValue, newPlusE)
        _submit(self.doubleSpinBoxMinusE.setValue, newMinusE)

        startE = centerE + newMinusE
        stopE = centerE + newPlusE

        self.calculate_concentration(startE, stopE)

    def calcConcentrationMinus(self):
        centerE = self.doubleSpinBoxCenterE.value()
        stepE = self.doubleSpinBoxStepE.value()
        plusE = self.doubleSpinBoxPlusE.value()
        minusE = self.doubleSpinBoxMinusE.value()

        newMinusE = minusE + stepE
        newPlusE = plusE - stepE

        _submit(self.doubleSpinBoxPlusE.setValue, newPlusE)
        _submit(self.doubleSpinBoxMinusE.setValue, newMinusE)

        startE = centerE + newMinusE
        stopE = centerE + newPlusE

        self.calculate_concentration(startE, stopE)

    def calcPeakFitting(self):
        if self.thickness_image is not None:
            self.toLog("Calculating peak fitting...")
            # self.peak_image = np.zeros_like(self.thickness_image)
            # self.peak_image.fill(np.nan)
            cutOff = self.spinBoxCutOff.value()
            mask = self.maskToolsWidget.getSelectionMask()
            if not np.any(mask):
                # fit all data if mask is empty
                mask = np.ones_like(self.thickness_image, dtype=np.uint8)

            fitModel = self.comboBoxFitModel.currentText()
            fitPoints = self.spinBoxFitPoints.value()

            nrj = np.array(self.energy_list, dtype=np.float64)
            stack = np.array(self._preprocessImage, dtype=np.float64)

            if self.groupBoxFitRange.isChecked():
                startE = self.doubleSpinBoxFitStartE.value()
                stopE = self.doubleSpinBoxFitStopE.value()
            else:
                startE = self.energy_list[0]
                stopE = self.energy_list[-1]

            smooth = self.groupBoxSmooth.isChecked()
            algorithm = self.comboBoxSmoothAlgorithm.currentText()
            window_length = self.spinBoxWindowLength.value()
            polyorder = self.spinBoxPolyorder.value()

            if fitModel == "Polynomial":
                self.peak_image = quadfit_mc(nrj,
                                          stack,
                                          fitPoints,
                                          mask,
                                          startE,
                                          stopE,
                                          smooth,
                                          algorithm,
                                          window_length,
                                          polyorder)
            elif fitModel == "Gaussian":
                self.peak_image = gaussianfit_mc(nrj,
                                              stack,
                                              fitPoints,
                                              mask,
                                              startE,
                                              stopE,
                                              smooth,
                                              algorithm,
                                              window_length,
                                              polyorder)

            # Calibrate Energy Shift
            if self.groupBoxLargeAreaCal.isChecked():
                shape = self.peak_image.shape
                energy_diff = self.doubleSpinBoxEnergyDifference.value()
                num_pixel = self.doubleSpinBoxNumPixel.value()
                slope = energy_diff / num_pixel
                index_array = np.arange(shape[0]) + self.crop_info.get("yStart", 0)
                shifted_peak_image = self.peak_image.copy() - slope * index_array[:, None]
                self.peak_image = shifted_peak_image
                nrj = np.array(self.energy_list, dtype=np.float64)

                # Update proj, back, and absorbance images
                self.toLog("Calibrating absorbance images...")

                self.absorbanceImage = renormalize_absorbance_stack(nrj,
                                                                    self.absorbanceImage.astype(np.float64),
                                                                    slope)
                self.toLog("Calibrating absorbance images... done")

            # Reject outliers
            # Average and standard deviation
            peak_image = self.peak_image.copy()
            peak_average = np.nanmean(self.peak_image)
            peak_image[np.abs(self.peak_image - peak_average) > cutOff ] = np.nan
            self.peak_energy_mean = np.nanmean(peak_image)
            self.peak_energy_std = np.nanstd(peak_image)

            self.widgetImageStack.setFitImage(self.peak_image)
            _submit(self.widgetImageStack.setStack, self.peak_image)

            plot = self.widgetImageStack.getPlotWidget()
            _submit(plot.setGraphTitle, f"Apex Energies, Mean : {self.peak_energy_mean:.2f} eV, Std : {self.peak_energy_std:.2f} eV")

            # Clear mask
            _submit(self.maskToolsWidget._handleClearMask)
            self.toLog("Calculating peak fitting... done")
            # _submit(qt.QMessageBox.information, plot, "Info", "Fitting finished")
        else:
            self.toLog("Please calculate thickness first", "red")

    def calcThickness(self):
        if self.absorbanceImage is not None:
            self.toLog("Calculating thickness...")
            num_pre_edge = self.spinBoxNumPreEdge.value()
            num_post_edge = self.spinBoxNumPostEdge.value()
            image_pre_edge = np.mean(self._preprocessImage[:num_pre_edge], axis=0)
            image_post_edge = np.mean(self._preprocessImage[-num_post_edge:], axis=0)
            thickness = image_post_edge - image_pre_edge
            self.thickness_image = thickness
            self.widgetImageStack.setStack(self.thickness_image)
            plot = self.widgetImageStack.getPlotWidget()
            _submit(plot.setGraphTitle, "Edge jump")
            self.toLog("Calculating thickness... done")
        else:
            self.toLog("Please load images first", "red")

    def cropImage(self):
        xStart = self.spinBoxXStart.value()
        xStop = self.spinBoxXStop.value()
        yStart = self.spinBoxYStart.value()
        yStop = self.spinBoxYStop.value()
        self.crop_info = {"xStart": xStart,
                          "xStop": xStop,
                          "yStart": yStart,
                          "yStop": yStop}

        self.absorbanceImage = self.absorbanceImage[:, yStart:yStop, xStart:xStop]
        self._preprocessImage = self.absorbanceImage.copy()
        self.backImage = self.backImage[:, yStart:yStop, xStart:xStop]
        self.projImage = self.projImage[:, yStart:yStop, xStart:xStop]
        self.toLog("Image cropped")

        # Hide, Reload, and Reset zoom
        _submit(self.checkBoxROI.setChecked, False)
        self.reloadDisplay()
        _submit(self.widgetImageStack.resetZoom)

    def updateRoi(self, origin, size):
        """Update ROI"""
        xStart = origin[0]
        xStop = origin[0] + size[0]
        yStart = origin[1]
        yStop = origin[1] + size[1]

        _submit(self.spinBoxXStart.setValue, xStart)
        _submit(self.spinBoxXStop.setValue, xStop)
        _submit(self.spinBoxYStart.setValue, yStart)
        _submit(self.spinBoxYStop.setValue, yStop)

    def updateRoiFromValue(self):
        """Update ROI from values"""
        xStart = self.spinBoxXStart.value()
        xStop = self.spinBoxXStop.value()
        yStart = self.spinBoxYStart.value()
        yStop = self.spinBoxYStop.value()

        origin = (xStart, yStart)
        size = (abs(xStop - xStart), abs(yStop - yStart))

        roi = self.widgetImageStack.getPlotWidget().getRoi()

        # Set origin, size
        _submit(roi.setOrigin, origin)
        _submit(roi.setSize, size)

    def alignImages(self, notify=True):
        if self.absorbanceImage is not None:
            refImageIdx = self.spinBoxRefNum.value()
            upsample_factor = self.spinBoxUpFactor.value()

            self.toLog("Aligning...")

            ##### Using rust
            shifts = phase_cross_correlation_stack(self.absorbanceImage.astype(np.float64),
                                            refImageIdx,
                                            upsample_factor)
            self.image_shifts = shifts
            self.image_shifts_abs = np.linalg.norm(shifts, axis=1)

            # Shift images
            for idx, image in enumerate(self.absorbanceImage):
                self.absorbanceImage[idx] = shift(image, shifts[idx], mode='constant', cval=-10)

            # To numpy array
            self.image_shifts = np.array(self.image_shifts)

            # self.image_shifts = np.array(self.image_shifts)
            shiftXMin = floor(np.min(self.image_shifts[:, 1]))
            shiftXMax = ceil(np.max(self.image_shifts[:, 1]))
            shiftYMin = floor(np.min(self.image_shifts[:, 0]))
            shiftYMax = ceil(np.max(self.image_shifts[:, 0]))

            # print(f"shiftXMin : {shiftXMin}, shiftXMax : {shiftXMax}, shiftYMin : {shiftYMin}, shiftYMax : {shiftYMax}")
            # Crop images
            if shiftYMin < 0 and shiftXMin < 0:
                self.absorbanceImage = self.absorbanceImage[:, shiftYMax:shiftYMin, shiftXMax:shiftXMin]
            elif shiftYMin < 0 and shiftXMin >= 0:
                self.absorbanceImage = self.absorbanceImage[:, shiftYMax:shiftYMin, shiftXMax:]
            elif shiftYMin >= 0 and shiftXMin < 0:
                self.absorbanceImage = self.absorbanceImage[:, shiftYMax:, shiftXMax:shiftXMin]
            else:
                self.absorbanceImage = self.absorbanceImage[:, shiftYMax:, shiftXMax:]

            self._preprocessImage = self.absorbanceImage.copy()
            self.reloadDisplay()
            self.toLog("Aligning... done")

            _submit(self.widgetPlotShift.addCurve, self.energy_list, self.image_shifts_abs, legend="shift_abs", color='blue')

            if notify:
                plot = self.widgetImageStack.getPlotWidget()
                _submit(qt.QMessageBox.information, plot, "Info", "Alignment finished")
        else:
            self.toLog("Please load images first", "red")

    def applyFiltering(self):
        filter_type = self.comboBoxFilterType.currentText()
        filter_size = self.comboBoxFilterSize.currentIndex() * 2 + 3
        self.toLog("Filtering...")
        if filter_type == 'Median':
            kernel_size = (1, filter_size, filter_size)
            self.absorbanceImage = median_filter(self.absorbanceImage, size=kernel_size)
            self._preprocessImage = self.absorbanceImage.copy()
            self.toLog("Filter finished")
            self.reloadDisplay()
        else:
            self.toLog("Filter not supported", "red")

    def updateEnergy(self, energy):
        _submit(self.widgetPlotShift.addXMarker,
                energy,
                legend="XMarker",
                color='red',
                text=f"{energy:.2f} eV")

    def reloadDisplay(self):
        # Get Selection and display
        if self.checkBoxAbsorbance.isChecked():
            imgStack = self.absorbanceImage
        elif self.checkBoxBackground.isChecked():
            imgStack = self.backImage
        else:
            imgStack = self.projImage

        if imgStack is not None:
            _submit(self.widgetImageStack.setStack, imgStack)

    def toLog(self, text, color='black'):
        """Append to log widget"""
        time_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"[ {time_string} ] {text}"
        _submit(self.logTextEdit.append, text)

    def load_h5(self):
        """Load energy, projection, background, and absorbance images from h5 file"""
        if os.path.exists(self.selected_path_h5):
            try:
                with h5py.File(self.selected_path_h5, "r") as f:
                    entry_1 = f["entry_1"]
                    self.energy_list = entry_1["energies"][()]
                    self.absorbanceImage = entry_1["absorbance"][()]
                    self._preprocessImage = self.absorbanceImage.copy()
                    self.projImage = entry_1["projection"][()]
                    self.backImage = entry_1["background"][()]
                    self.basename = os.path.basename(self.selected_path_h5)[:-3].split('_result')[0]
                    self.widgetPlotSpectrum.setFilename(self.basename)
                    self.widgetPlotSpectrum.setPath(str(Path(self.selected_path_h5).parent))
                    self.widgetPlotHistogram.setPath(str(Path(self.selected_path_h5).parent))

                    minEnergy = min(self.energy_list)
                    maxEnergy = max(self.energy_list)
                    self.middle_index = len(self.energy_list) // 2

                    # Set reference image
                    self.spinBoxRefNum.setMinimum(0)
                    self.spinBoxRefNum.setMaximum(len(self.energy_list) - 1)
                    _submit(self.spinBoxRefNum.setValue, self.middle_index)

                    _submit(self.widgetPlotShift.setGraphXLimits, minEnergy, maxEnergy)


                    # Get Selection and display
                    if self.checkBoxAbsorbance.isChecked():
                        imgStack = self.absorbanceImage
                    elif self.checkBoxBackground.isChecked():
                        imgStack = self.backImage
                    else:
                        imgStack = self.projImage

                    self.crop_info = {"xStart": entry_1["crop_xstart"][()],
                                      "xStop": entry_1["crop_xstop"][()],
                                      "yStart": entry_1["crop_ystart"][()],
                                      "yStop": entry_1["crop_ystop"][()]}

                    _submit(self.widgetImageStack.setStack, imgStack)
                    _submit(self.widgetImageStack.resetZoom)
                    _submit(self.widgetImageStack.setCurrentIndex, self.middle_index)
                    self.widgetImageStack._energy_list = self.energy_list

                    self.toLog(f"Loaded {self.selected_path_h5}")

            except Exception as e:
                self.toLog(f"Error : {e}")
                return
        else:
            self.toLog(f"Error : {self.selected_path_h5} not found")

    def load(self):
        self.energy_list = []
        self.image_stack = []

        self.crop_info = {}

        if os.path.exists(self.selected_path):
            if self.selected_path.endswith("_proj") or self.selected_path.endswith("_back"):
                base_path = self.selected_path[:-5]
                self.basename = os.path.basename(base_path)

                # Load projection images
                proj_path = base_path + "_proj"
                proj_files = sorted(os.listdir(proj_path))

                # Energy list
                all_energy = []
                image_dict = {}

                self.toLog(f"loading filenames...")
                _time = time.time()

                re_pat = re.compile(r"[0-9]+.[0-9]+_eV_proj")
                for f in proj_files:
                    res = re.findall(re_pat, f)
                    # print(f"res : {res}")
                    if res:
                        energy = float(res[0][:-8])
                        all_energy.append(energy)
                        if energy in image_dict.keys():
                            image_dict[energy].append(f)
                        else:
                            image_dict[energy] = [f]

                self.toLog(f"loading filenames... done ({time.time() - _time})")

                self.energy_list = sorted(image_dict.keys())
                minEnergy = min(self.energy_list)
                maxEnergy = max(self.energy_list)
                self.middle_index = len(self.energy_list) // 2

                # Set reference image
                self.spinBoxRefNum.setMinimum(0)
                self.spinBoxRefNum.setMaximum(len(self.energy_list) - 1)
                _submit(self.spinBoxRefNum.setValue, self.middle_index)

                _submit(self.widgetPlotShift.setGraphXLimits, minEnergy, maxEnergy)

                temp_image_path = os.path.join(proj_path, image_dict[self.energy_list[0]][0])
                image_shape = tifffile.imread(temp_image_path).shape
                imageStack = np.zeros((len(self.energy_list), image_shape[0], image_shape[1]), dtype=np.float32)

                self.toLog(f"loading images...")
                _time = time.time()
                for idx, energy in enumerate(self.energy_list):
                    images = []
                    for f in image_dict[energy]:
                        image_path = os.path.join(proj_path, f)
                        images.append(tifffile.imread(image_path))
                    imageStack[idx] = np.mean(images, axis=0)

                self.projImage = imageStack
                # print(f"projImage : {self.projImage}")

                self.toLog(f"loading images... done ({time.time() - _time})")

                # Load background images
                back_path = base_path + "_back"
                back_files = sorted(os.listdir(back_path))
                back_image_dict = {}
                re_pat = re.compile(r"[0-9]+.[0-9]+_eV_back")

                self.toLog(f"loading back filenames...")
                _time = time.time()

                for f in back_files:
                    res = re.findall(re_pat, f)
                    if res:
                        energy = float(res[0][:-8])
                        if energy in back_image_dict.keys():
                            back_image_dict[energy].append(f)
                        else:
                            back_image_dict[energy] = [f]
                self.toLog(f"loading back filenames... done ({time.time() - _time})")

                self.toLog(f"back_images ...")
                _time = time.time()
                backImageStack = np.zeros((len(self.energy_list), image_shape[0], image_shape[1]), dtype=np.float32)

                for idx, energy in enumerate(self.energy_list):
                    images = []
                    if energy not in back_image_dict.keys():
                        original_energy = energy
                        energy = find_nearest(list(back_image_dict.keys()), energy)
                        self.toLog(f"Error : {original_energy} not found in back images. Using nearest value {energy} instead.")
                    for f in back_image_dict[energy]:
                        image_path = os.path.join(back_path, f)
                        # print(f"image_path : {image_path}")
                        try:
                            images.append(tifffile.imread(image_path))
                        except:
                            self.toLog(f"Error loading {image_path}")
                    backImageStack[idx] = np.mean(images, axis=0)

                self.backImage = backImageStack
                # print(f"backImage : {self.backImage}")

                self.toLog(f"back_images ... done ({time.time() - _time})")

                # Calculate absorbance
                self.absorbanceImage = -np.log(self.projImage / self.backImage)
                self._preprocessImage = self.absorbanceImage.copy()

                # Get Selection and display
                if self.checkBoxAbsorbance.isChecked():
                    imgStack = self.absorbanceImage
                elif self.checkBoxBackground.isChecked():
                    imgStack = self.backImage
                else:
                    imgStack = self.projImage


                # print(f"absorbanceImage : {self.absorbanceImage}")

                _submit(self.widgetImageStack.setStack, imgStack)
                _submit(self.widgetImageStack.resetZoom)
                _submit(self.widgetImageStack.setCurrentIndex, self.middle_index)
                self.widgetImageStack._energy_list = self.energy_list

                # store filename and path
                base_path = str(Path(base_path).parent)
                self.widgetPlotSpectrum.setPath(base_path)
                self.widgetPlotHistogram.setPath(base_path)
                self.widgetPlotSpectrum.setFilename(self.basename)

                self.toLog(f"loading done")
            else:
                self.toLog("Error : Folder name should end with '_proj' or '_back'")
        else:
            self.toLog("Error : Path not found")

    def select_load_path(self):
        """ Select data save path """
        # Load previous path
        qsettings = qt.QSettings('settings.ini', qt.QSettings.IniFormat)
        previous_selection = qsettings.value('selected_path', BASE_PATH)

        # Get user input
        self.selected_path = qt.QFileDialog.getExistingDirectory(
                                        self,
                                        "Select a load path",
                                        previous_selection,
                                        qt.QFileDialog.ShowDirsOnly | qt.QFileDialog.DontUseNativeDialog)

        # Selection canceled
        if self.selected_path == "":
            return

        elif os.path.exists(self.selected_path):
            # Display current path
            _submit(self.lineEditFilePath.setText, str(self.selected_path))
            _submit(self.lineEditSavePath.setText, str(os.path.dirname(self.selected_path)))
            save_path = str(Path(self.selected_path).parent)
            qsettings.setValue('selected_path', save_path)

    def select_load_path_h5(self):
        """ Select result load path """
        # Load previous path
        qsettings = qt.QSettings('settings.ini', qt.QSettings.IniFormat)
        previous_selection = qsettings.value('selected_path_h5', BASE_PATH)

        # Get user input
        self.selected_path_h5 = qt.QFileDialog.getOpenFileName(
                                        self,
                                        "Select a h5 file",
                                        previous_selection,
                                        "H5 files (*.h5)")[0]

        # Selection canceled
        if self.selected_path_h5 == "":
            return

        elif os.path.exists(self.selected_path_h5):
            # Display current path
            _submit(self.lineEditFilePathH5.setText, str(self.selected_path_h5))
            save_path = str(Path(self.selected_path_h5).parent)
            qsettings.setValue('selected_path_h5', save_path)

    def select_save_path(self):
        """ Select data save path """
        # Load previous path
        qsettings = qt.QSettings('settings.ini', qt.QSettings.IniFormat)
        previous_selection = qsettings.value('selected_save_path', self.selected_path)

        # Get user input
        self.selected_save_path = qt.QFileDialog.getExistingDirectory(
                                        self,
                                        "Select a save path",
                                        previous_selection,
                                        qt.QFileDialog.ShowDirsOnly | qt.QFileDialog.DontUseNativeDialog)

        # Selection canceled
        if self.selected_save_path == "":
            return

        elif os.path.exists(self.selected_save_path):
            # Display current path
            _submit(self.lineEditSavePath.setText, str(self.selected_save_path))
            save_path = str(self.selected_save_path)
            self.widgetPlotSpectrum.setPath(save_path)
            qsettings.setValue('selected_save_path', save_path)

    def select_mask(self):
        """ Select mask path """
        # Load previous path
        qsettings = qt.QSettings('settings.ini', qt.QSettings.IniFormat)
        previous_selection = qsettings.value('selected_mask_path', BASE_PATH)

        # Get user input
        self.selected_mask_path = qt.QFileDialog.getOpenFileName(
                                        self,
                                        "Select a mask file",
                                        previous_selection,
                                        "Images (*.tif *.tiff *.edf)")[0]
        print(f"selected_mask_path : {self.selected_mask_path}")
        # Selection canceled
        if self.selected_mask_path == "":
            return

        elif os.path.exists(self.selected_mask_path):
            # Display current path
            _submit(self.lineEditMaskPath.setText, str(self.selected_mask_path))
            save_path = str(Path(self.selected_mask_path).parent)
            qsettings.setValue('selected_mask_path', save_path)

if __name__ == '__main__':
    app = qt.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())

