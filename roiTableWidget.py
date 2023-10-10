import sys

from silx.gui import qt
from silx.gui.plot.tools.roi import RegionOfInterestManager
from silx.gui.plot.tools.roi import RegionOfInterestTableWidget
from silx.gui.plot.items import SymbolMixIn

roiClasses = ('PointROI', 'RectangleROI', 'CircleROI')

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

    print(f"roiIndexes: {roiIndexes}, index : {index}")

    if roi.getName() == '':
        roi.setName(f'ROI {index}')
    if isinstance(roi, SymbolMixIn):
        roi.setSymbolSize(4)
    roi.setSelectable(True)
    roi.setEditable(True)

class RoiTableWidget(qt.QWidget):
    def __init__(self, parent=None, plot=None):
        super(RoiTableWidget, self).__init__(parent)
        self.roiManager = RegionOfInterestManager(parent=plot)
        self.roiManager.setColor('pink')
        self.roiManager.sigRoiAdded.connect(updateAddedRegionOfInterest)
        # self.roiManager.sigRoiAdded.connect(self._roiAdded)
        # self.roiManager.sigRoiRemoved.connect(self._roiRemoved)
        # self.roiManager.sigRoiChanged.connect(self._roiChanged)

        self.roiTable = RegionOfInterestTableWidget()
        self.roiTable.setRegionOfInterestManager(self.roiManager)

        self.roiToolbar = qt.QToolBar()
        self.roiToolbar.setIconSize(qt.QSize(16, 16))

        for roiClass in self.roiManager.getSupportedRoiClasses():
            if roiClass.__name__ in roiClasses:
                action = self.roiManager.getInteractionModeAction(roiClass)
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
