import sys

from silx.gui import qt
from silx.gui.plot.tools.roi import RegionOfInterestManager
from silx.gui.plot.tools.roi import RegionOfInterestTableWidget
from silx.gui.plot.items import SymbolMixIn
from silx.gui import icons
from silx.gui.utils.concurrent import submitToQtMainThread as _submit

roiClasses = ('PointROI', 'RectangleROI', 'CircleROI')

class RegionOfInterestManagerCustom(RegionOfInterestManager):
    sigUpdatedRoi = qt.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        plot = self.parent()
        plot.sigPlotSignal.connect(self.updateInteraction)

    def updateInteraction(self, event):
        """emit update roi signal when markerMoved event is triggered."""
        if event['event'] == 'markerMoved':
            currentRoi = self.getCurrentRoi()
            self.sigUpdatedRoi.emit(currentRoi)


def updateAddedRegionOfInterest(roi):
    rois = roi.parent().getRois()
    roiIndexes = []
    for roi in rois:
        name = roi.getName()
        if name.startswith('ROI'):
            roiIndexes.append(int(name[4:]))

    index = 1
    for i in range(1, len(rois)+1):
        if i not in roiIndexes:
            index = i
            break

    if roi.getName() == '':
        roi.setName(f'ROI {index}')
    if isinstance(roi, SymbolMixIn):
        roi.setSymbolSize(4)
    roi.setSelectable(True)
    roi.setEditable(True)

class RoiTableWidget(qt.QWidget):
    sigClearRoiCurves = qt.Signal()

    def __init__(self, parent=None, plot=None):
        super(RoiTableWidget, self).__init__(parent)
        self.roiManager = RegionOfInterestManagerCustom(parent=plot)
        self.roiManager.setColor('pink')
        self.roiManager.sigRoiAdded.connect(updateAddedRegionOfInterest)

        self.roiTable = RegionOfInterestTableWidget()
        self.roiTable.setRegionOfInterestManager(self.roiManager)

        self.roiToolbar = qt.QToolBar()
        self.roiToolbar.setIconSize(qt.QSize(16, 16))

        for roiClass in self.roiManager.getSupportedRoiClasses():
            if roiClass.__name__ in roiClasses:
                action = self.roiManager.getInteractionModeAction(roiClass)
                self.roiToolbar.addAction(action)

        # Need to delete all rois action
        icon = icons.getQIcon("remove")
        action = qt.QAction(icon, "Delete all ROIs", self)
        action.triggered.connect(self.roiManager.clear)
        action.triggered.connect(self.sigClearRoiCurves.emit)
        self.roiToolbar.addAction(action)

        layout = qt.QVBoxLayout()
        layout.addWidget(self.roiToolbar)
        layout.addWidget(self.roiTable)

        self.setLayout(layout)


if __name__ == '__main__':
    from silx.gui.plot import Plot2D

    app = qt.QApplication(sys.argv)
    plot = Plot2D()
    main = RoiTableWidget(plot=plot)
    main.show()
    sys.exit(app.exec_())
