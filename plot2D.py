import sys
import logging
from collections import OrderedDict

import numpy as np

from silx.gui import qt
from silx.gui import icons

import silx
from silx.gui.plot import PlotWindow
from silx.gui.plot.items.roi import RectangleROI
from silx.gui.plot import items
from silx.gui.plot.tools.roi import RegionOfInterestManager
from silx.gui.plot.Profile import ProfileToolBar
from silx.utils.weakref import WeakMethodProxy

from saveAction import SaveAction

logger = logging.getLogger(__name__)

class PlotWindowCustom(PlotWindow):
    """PlotWindow with a toolbar specific for images.

    This widgets provides the plot API of :~:`.PlotWidget`.

    :param parent: The parent of this widget
    :param backend: The backend to use for the plot (default: matplotlib).
                    See :class:`.PlotWidget` for the list of supported backend.
    :type backend: str or :class:`BackendBase.BackendBase`
    """

    def __init__(self, parent=None, backend=None):
        # List of information to display at the bottom of the plot
        posInfo = [
            ('X', lambda x, y: x),
            ('Y', lambda x, y: y),
            ('Data', WeakMethodProxy(self._getImageValue)),
            ('Dims', WeakMethodProxy(self._getImageDims)),
        ]

        super(PlotWindowCustom, self).__init__(parent=parent, backend=backend,
                                     resetzoom=True, autoScale=False,
                                     logScale=False, grid=False,
                                     curveStyle=False, colormap=True,
                                     aspectRatio=True, yInverted=True,
                                     copy=True, save=True, print_=True,
                                     control=False, position=posInfo,
                                     roi=False, mask=False)
        if parent is None:
            self.setWindowTitle('Plot2D')
        self.getXAxis().setLabel('Columns')
        self.getYAxis().setLabel('Rows')

        if silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION == 'downward':
            self.getYAxis().setInverted(True)

        self.profile = ProfileToolBar(plot=self)
        self.addToolBar(self.profile)

        self.colorbarAction.setVisible(True)
        self.getColorBarWidget().setVisible(True)

        # Put colorbar action after colormap action
        actions = self.toolBar().actions()
        for action in actions:
            if action is self.getColormapAction():
                break

        self.sigActiveImageChanged.connect(self.__activeImageChanged)

    def __activeImageChanged(self, previous, legend):
        """Handle change of active image

        :param Union[str,None] previous: Legend of previous active image
        :param Union[str,None] legend: Legend of current active image
        """
        if previous is not None:
            item = self.getImage(previous)
            if item is not None:
                item.sigItemChanged.disconnect(self.__imageChanged)

        if legend is not None:
            item = self.getImage(legend)
            item.sigItemChanged.connect(self.__imageChanged)

        positionInfo = self.getPositionInfoWidget()
        if positionInfo is not None:
            positionInfo.updateInfo()

    def __imageChanged(self, event):
        """Handle update of active image item

        :param event: Type of changed event
        """
        if event == items.ItemChangedType.DATA:
            positionInfo = self.getPositionInfoWidget()
            if positionInfo is not None:
                positionInfo.updateInfo()

    def _getImageValue(self, x, y):
        """Get status bar value of top most image at position (x, y)

        :param float x: X position in plot coordinates
        :param float y: Y position in plot coordinates
        :return: The value at that point or '-'
        """
        pickedMask = None
        for picked in self.pickItems(
                *self.dataToPixel(x, y, check=False),
                lambda item: isinstance(item, items.ImageBase)):
            if isinstance(picked.getItem(), items.MaskImageData):
                if pickedMask is None:  # Use top-most if many masks
                    pickedMask = picked
            else:
                image = picked.getItem()

                indices = picked.getIndices(copy=False)
                if indices is not None:
                    row, col = indices[0][0], indices[1][0]
                    value = image.getData(copy=False)[row, col]

                    if pickedMask is not None:  # Check if masked
                        maskItem = pickedMask.getItem()
                        indices = pickedMask.getIndices()
                        row, col = indices[0][0], indices[1][0]
                        if maskItem.getData(copy=False)[row, col] != 0:
                            return value, "Masked"
                    return value

        return '-'  # No image picked

    def _getImageDims(self, *args):
        activeImage = self.getActiveImage()
        if (activeImage is not None and
                    activeImage.getData(copy=False) is not None):
            dims = activeImage.getData(copy=False).shape[1::-1]
            return 'x'.join(str(dim) for dim in dims)
        else:
            return '-'

    def getProfileToolbar(self):
        """Profile tools attached to this plot

        See :class:`silx.gui.plot.Profile.ProfileToolBar`
        """
        return self.profile

    def getProfilePlot(self):
        """Return plot window used to display profile curve.

        :return: :class:`Plot1D`
        """
        return self.profile.getProfilePlot()

class Plot2D(PlotWindowCustom):
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

        # self._roiAction = qt.QAction(icons.getQIcon('image-select-box'), 'Toggle ROI', self)
        # self._roiAction.setCheckable(True)
        # self._roiAction.toggled.connect(self.toggleROI)
        # self._toolbar.addAction(self._roiAction)

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
