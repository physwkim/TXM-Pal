import re
import os
import sys
from pathlib import Path
import time

import tifffile
import numpy as np
import datetime
from scipy.ndimage import median_filter
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation

from silx.gui import qt
from silx.gui.utils.concurrent import submitToQtMainThread as _submit

BASE_PATH = os.path.expanduser('~')

class Main(qt.QMainWindow):
    """Main Window"""

    hided = qt.Signal(object)
    closed = qt.Signal(object)

    sigStop = qt.Signal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.absorbanceImage = None
        self.backImage = None
        self.projImage = None
        self.middle_index = 0
        self.image_shifts = []

        qt.loadUi('ui/main.ui', self)

        self.widgetPlot1D.setGraphTitle("Shift")
        self.widgetPlot1D.setGraphXLabel("Energy (eV)")
        self.widgetPlot1D.setGraphYLabel("Shift (pixel)")

        self.pushButtonSelectPath.clicked.connect(self.select_path)
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
                self.image_shifts.append(shift)
                self.absorbanceImage[idx] = shift(image, shift_values, mode='reflect')
                self.image_shifts_abs.append(np.linalg.norm(shift_values))
                    
            self.reloadDisplay()
            self.toLog("Aligning... done")

            _submit(self.widgetPlot1D.addCurve, self.energy_list, self.image_shifts_abs, legend="shift_abs", color='blue')

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
        _submit(self.widgetPlot1D.addXMarker,
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
                self.spinBoxRefNum.setValue(self.middle_index)

                self.widgetPlot1D.setGraphXLimits(minEnergy, maxEnergy)

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

    def select_path(self):
        """ Select data save path """
        # Load previous path
        qsettings = qt.QSettings('settings.ini', qt.QSettings.IniFormat)
        previous_selection = qsettings.value('selected_path', BASE_PATH)

        # Get user input
        self.selected_path = qt.QFileDialog.getExistingDirectory(
                                        self,
                                        "Select a save path",
                                        previous_selection,
                                        qt.QFileDialog.ShowDirsOnly | qt.QFileDialog.DontUseNativeDialog)
        
        # Selection canceled
        if self.selected_path == "":
            return

        elif os.path.exists(self.selected_path):
            # Display current path
            _submit(self.lineEditFilePath.setText, str(self.selected_path))
            save_path = str(Path(self.selected_path).parent)
            qsettings.setValue('selected_path', save_path)

if __name__ == '__main__':
    app = qt.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())

