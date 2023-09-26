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
from skimage.registration import phase_cross_correlation
from matplotlib.colors import hsv_to_rgb
import dask
from dask import delayed

from silx.gui import qt
from silx.gui.utils.concurrent import submitToQtMainThread as _submit
import fabio

import h5py

from utils import fitPeak

BASE_PATH = os.path.expanduser('~')

class Main(qt.QMainWindow):
    """Main Window"""

    hided = qt.Signal(object)
    closed = qt.Signal(object)

    sigStop = qt.Signal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.basename = ""
        self.absorbanceImage = None
        self.backImage = None
        self.projImage = None
        self.middle_index = 0
        self.image_shifts = []
        self.thickness_image = None
        self.concentration_image = None

        qt.loadUi('ui/main.ui', self)

        # Adjust splitter ratio
        self.splitterMain.setSizes([100, 900])

        self.setWindowTitle("PAL XANES")
        self.widgetImageStack.getPlotWidget().setDataBackgroundColor([0,0,0])

        self.widgetPlotShift.setGraphTitle("Shift")
        self.widgetPlotShift.setGraphXLabel("Energy (eV)")
        self.widgetPlotShift.setGraphYLabel("Shift (pixel)")

        self.pushButtonSelectPath.clicked.connect(self.select_load_path)
        self.pushButtonSelectSavePath.clicked.connect(self.select_save_path)

        self.pushButtonLoad.setCallable(self.load)
        
        self.checkBoxAbsorbance.clicked.connect(self.reloadDisplay)
        self.checkBoxBackground.clicked.connect(self.reloadDisplay)
        self.checkBoxRawImage.clicked.connect(self.reloadDisplay)
        
        self.widgetImageStack.sigEnergyKeV.connect(self.updateEnergy)

        self.pushButtonFiltering.setCallable(self.applyFiltering)
        self.pushButtonAlign.setCallable(self.alignImages)

        # ROI for cropping
        imgWidget = self.widgetImageStack.getPlotWidget()
        imgWidget.sigRoiUpdated.connect(self.updateRoi)

        self.pushButtonCrop.setCallable(self.cropImage)

        self.pushButtonThickness.clicked.connect(self.calcThickness)
        self.pushButtonSelectMask.clicked.connect(self.select_mask)
        self.pushButtonFitting.clicked.connect(self.calcPeakFitting)
        self.pushButtonConcentration.clicked.connect(self.calcConcentration)
        
        self.pushButtonSaving.clicked.connect(self.saveData)

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

        with h5py.File(save_file, "w") as f:
            entry_1 = f.create_group("entry_1")
            entry_1.create_dataset("absorbance", data=self.absorbanceImage)
            entry_1.create_dataset("projection", data=self.projImage)
            entry_1.create_dataset("thickness", data=self.thickness_image)
            entry_1.create_dataset("concentration", data=self.concentration_image)
            entry_1.create_dataset("peak", data=self.peak_image)
            entry_1.create_dataset("peak_energy_mean", data=self.peak_energy_mean)
            entry_1.create_dataset("peak_energy_std", data=self.peak_energy_std)

        self.toLog(f"Result saved to {save_file}")

    def calcConcentration(self):
        self.toLog("Calculating concentration...")
        startE = self.doubleSpinBoxStartE.value()
        stopE = self.doubleSpinBoxStopE.value()
        conc = (self.peak_image - startE) / (stopE - startE)
        thickness_img = self.thickness_image.copy()
        thickness_img[conc < 0] = 0
        thickness_img[conc > 1.05] = 0
        negative_mask = np.logical_and(conc<0 , np.logical_not(np.isnan(conc)))
        ceil_mask = np.logical_and(conc>1.05, np.logical_not(np.isnan(conc)))

        conc[negative_mask] = 0
        conc[ceil_mask] = 1

        shape = conc.shape

        # Fill H, S, V channels
        hsv = np.zeros((shape[0], shape[1], 3))
        hsv[:, :, 0] = conc * (1/3)
        hsv[:, :, 1] = np.ones_like(conc)
        hsv[:, :, 2] = thickness_img / np.nanmax(thickness_img)

        # Replace Nan With 0
        # hsv = np.nan_to_num(hsv, nan=0.0)

        # Convert HSV to RGB
        rgb = hsv_to_rgb(hsv)

        # Remove Unmasked Pixels
        rgb[:,:,0][np.where(np.isnan(conc))] = 0
        rgb[:,:,1][np.where(np.isnan(conc))] = 0
        rgb[:,:,2][np.where(np.isnan(conc))] = 0
        
        # Clip RGB values
        rgb = np.clip(rgb, 0, 1)
        self.concentration_image = rgb

        plot = self.widgetImageStack.getPlotWidget()
        _submit(plot.addImage, self.concentration_image)
        _submit(plot.setGraphTitle, f"Concentraion")
        self.toLog("Calculating concentration... done")

    def calcPeakFitting(self):
        if self.thickness_image is not None:
            self.toLog("Calculating peak fitting...")
            self.peak_image = np.zeros_like(self.thickness_image)
            self.peak_iamge = self.peak_image.fill(np.nan)
            cutOff = self.spinBoxCutOff.value()
            fitModel = self.comboBoxFitModel.currentText()
            fitPoints = self.spinBoxFitPoints.value()
            algorithm = self.comboBoxFitModel.currentText()

            if os.path.exists(self.selected_mask_path):
                mask = fabio.open(self.selected_mask_path).data
            else:
                mask = np.ones_like(self.thickness_image)
            
            idx_arr = np.where(mask > 0)

            for idx, row in enumerate(idx_arr[0]):
                col = idx_arr[1][idx]
                spectrum = self.absorbanceImage[:, row, col]
                # print(f"row : {row}, col : {col}, spectrum : {spectrum}")
                maxIdx = np.argmax(spectrum)
                idx_start = maxIdx - fitPoints // 2
                idx_stop = maxIdx + fitPoints // 2

                # print(f"maxIdx : {maxIdx}, idx_start : {idx_start}, idx_stop : {idx_stop}")
                xdata = self.energy_list[idx_start:idx_stop]
                ydata = spectrum[idx_start:idx_stop]
                # print(f"xdata : {xdata}, ydata : {ydata}")
                if len(xdata) == 0 or len(ydata) == 0:
                    continue
                
                cen = fitPeak(xdata, ydata, algorithm=algorithm)
                # cen = delayed(fitPeak)(xdata, ydata)
                # print(f"cen : {cen}")

                self.peak_image[row, col] = cen

            # self.peak_image = dask.compute(self.peak_image)

            # Reject outliers
            peak_average = np.nanmean(self.peak_image)
            self.peak_image[np.abs(self.peak_image - peak_average) > cutOff ] = np.nan

            self.peak_energy_mean = np.nanmean(self.peak_image)
            self.peak_energy_std = np.nanstd(self.peak_image)

            _submit(self.widgetImageStack.setStack, self.peak_image)

            plot = self.widgetImageStack.getPlotWidget()
            _submit(plot.setGraphTitle, f"Apex Energies, Mean : {self.peak_energy_mean:.2f} eV, Std : {self.peak_energy_std:.2f} eV")
            self.toLog("Calculating peak fitting... done")

    def calcThickness(self):
        if self.absorbanceImage is not None:
            self.toLog("Calculating thickness...")
            num_pre_edge = self.spinBoxNumPreEdge.value()
            num_post_edge = self.spinBoxNumPostEdge.value()
            image_pre_edge = np.mean(self.absorbanceImage[:num_pre_edge], axis=0)
            image_post_edge = np.mean(self.absorbanceImage[-num_post_edge:], axis=0)
            thickness = image_post_edge - image_pre_edge
            self.thickness_image = thickness
            self.widgetImageStack.setStack(self.thickness_image)
            plot = self.widgetImageStack.getPlotWidget()
            _submit(plot.setGraphTitle, "Thickness")
            self.toLog("Calculating thickness... done")

    def cropImage(self):
        xStart = self.spinBoxXStart.value()
        xStop = self.spinBoxXStop.value()
        yStart = self.spinBoxYStart.value()
        yStop = self.spinBoxYStop.value()

        self.absorbanceImage = self.absorbanceImage[:, yStart:yStop, xStart:xStop]
        self.backImage = self.backImage[:, yStart:yStop, xStart:xStop]
        self.projImage = self.projImage[:, yStart:yStop, xStart:xStop]
        self.toLog("Image cropped")

        # Hide, Reload, and Reset zoom
        _submit(self.widgetImageStack.getPlotWidget().toggleROI, False)
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


    def alignImages(self):
        if self.absorbanceImage is not None:
            refImageIdx = self.spinBoxRefNum.value()
            referenceImage = self.absorbanceImage[refImageIdx]
            upsample_factor = self.spinBoxUpFactor.value()
            self.image_shifts = []
            self.image_shifts_abs = []
            self.toLog("Aligning...")
            for idx, image in enumerate(self.absorbanceImage):
                shift_values, error, diffphase = phase_cross_correlation(referenceImage,
                                            image,
                                            upsample_factor=upsample_factor)
                self.image_shifts.append(shift_values)
                self.absorbanceImage[idx] = shift(image, shift_values, mode='constant', cval=-10)
                self.image_shifts_abs.append(np.linalg.norm(shift_values))

            self.image_shifts = np.array(self.image_shifts)
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
                    
            self.reloadDisplay()
            self.toLog("Aligning... done")

            _submit(self.widgetPlotShift.addCurve, self.energy_list, self.image_shifts_abs, legend="shift_abs", color='blue')

    def applyFiltering(self):
        filter_type = self.comboBoxFilterType.currentText()
        filter_size = self.spinBoxFilterSize.value()
        self.toLog("Filtering...")
        if filter_type == 'Median':
            kernel_size = (1, filter_size, filter_size)
            self.absorbanceImage = median_filter(self.absorbanceImage, size=kernel_size)
            # self.backImage = median_filter(self.backImage, size=kernel_size)
            # self.projImage = median_filter(self.projImage, size=kernel_size)
            self.toLog("Filter finished")
            self.reloadDisplay()

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
        _submit(self.textEditLog.append, text)

    def load(self):
        self.energy_list = []
        self.image_stack = []

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
                    for f in back_image_dict[energy]:
                        image_path = os.path.join(back_path, f)
                        images.append(tifffile.imread(image_path))
                    backImageStack[idx] = np.mean(images, axis=0)

                self.backImage = backImageStack

                self.toLog(f"back_images ... done ({time.time() - _time})")

                # Calculate absorbance
                self.absorbanceImage = -np.log(self.projImage / self.backImage)
                
                # Get Selection and display
                if self.checkBoxAbsorbance.isChecked():
                    imgStack = self.absorbanceImage
                elif self.checkBoxBackground.isChecked():
                    imgStack = self.backImage
                else:
                    imgStack = self.projImage

                _submit(self.widgetImageStack.setStack, imgStack)
                _submit(self.widgetImageStack.resetZoom)
                _submit(self.widgetImageStack.setCurrentIndex, self.middle_index)
                self.widgetImageStack._energy_list = self.energy_list

                self.toLog(f"loading done")

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
            save_path = str(Path(self.selected_save_path).parent)
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

