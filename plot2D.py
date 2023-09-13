import sys
import logging
from collections import OrderedDict

import numpy as np

from silx.gui import qt
from silx.gui import icons
from silx.gui.plot.PlotWindow import Plot2D as _Plot2D

from silx.gui.plot.items.roi import RectangleROI
from silx.gui.plot.tools.roi import RegionOfInterestManager

from saveAction import SaveAction

logger = logging.getLogger(__name__)

class Plot2D(_Plot2D):
    """Customized silx.gui.plot.PlotWindow.Plot2D window for roi selection

    :param parent: The parent of this widget
    :param backend: The backend to use for the plot (default: matplotlib).
                    See :class:`.PlotWidget` for the list of supported backend.
    """
    sigRoiUpdated = qt.Signal(object, object)

    def __init__(self, parent=None, backend='gl'):
        super().__init__(parent=parent, backend=backend)

        # ROI manager
        self._roiManager = RegionOfInterestManager(self)

        # Adjust margins
        self.setAxesMargins(0.05, 0.05, 0.05, 0.05)

        # Create default ROI
        self._roi = RectangleROI()
        self._roi.setGeometry(origin=(640, 640), size=(200, 200))
        self._roi.setName('Initial ROI')
        self._roi.setEditable(True)
        self._roi.setVisible(False)
        self._roi.sigEditingFinished.connect(self.updateRoiRegion)

        self._roiManager.addRoi(self._roi)

        self._roiAction = qt.QAction(icons.getQIcon('image-select-box'), 'Toggle ROI', self)
        self._roiAction.setCheckable(True)
        self._roiAction.toggled.connect(self.toggleROI)

        self._toolbar.addAction(self._roiAction)

        # Set tif as default selection on save action
        try:
            # remove oldSaveAction
            oldSaveAction = self._outputToolBar.getSaveAction()
            self._outputToolBar.removeAction(oldSaveAction)

            # create new saveAction
            saveAction = SaveAction(parent=self._outputToolBar, plot=self)
            self._outputToolBar._saveAction = saveAction
            self._outputToolBar.addAction(saveAction)

            imageFilters = self._outputToolBar.getSaveAction()._filters['image']
            imageFilters.move_to_end('Image data as EDF (*.edf)')

        except Exception as ex:
            print("Exception occured customizing SaveAction in Plot2D : {}".format(ex))

    def toggleROI(self, checked):
        """Show/Hide ROI"""

        # Automatically add a roi to RoiManager
        if checked and self._roi not in self._roiManager._rois:
            self._roiManager.addRoi(self._roi)

        self._roi.setVisible(checked)

    def getRoi(self):
        """Return RectangleROI"""
        return self._roi

    def setRoiEditable(self, value=True):
        """Change editing mode of roi"""
        editable = self.getRoi().isEditable()

        if editable != value:
            self.getRoi().setEditable(value)

    def updateRoiRegion(self):
        """Emit orgin and size signal when the ROI selection updated"""
        origin = np.array(self._roi.getOrigin()).astype(int)
        size = np.array(self._roi.getSize()).astype(int)
        # print("origin : {}, size : {}".format(origin, size))
        self.sigRoiUpdated.emit(origin, size)

if __name__ == '__main__':
    app = qt.QApplication([])
    plot = Plot2D()
    plot.show()
    sys.exit(app.exec_())
