import sys
import numpy as np
from silx.gui import qt
from silx.gui.colors import Colormap
from silx.gui.plot import PlotWidget
from silx.gui.utils.concurrent import submitToQtMainThread as submit

# Create a colormap
def createColormap(name):
    cm = Colormap(name=name, vmin=0, vmax=1)
    colors = cm.getNColors(nbColors=256)
    rgb = colors[:, :3]
    data = np.stack([rgb]*256)
    return data

class ColorBarWidget(qt.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        control_layout = qt.QHBoxLayout()
        self.colorComboBox = qt.QComboBox()
        self.colorComboBox.addItems(['hsv', 'turbo', 'jet', 'gist_rainbow_r', 'hsv_r'])
        self.colorComboBox.currentIndexChanged.connect(self.updateColor)
        control_layout.addWidget(self.colorComboBox)

        # min value
        self.minDoubleSpinBox = qt.QDoubleSpinBox()
        self.minDoubleSpinBox.setRange(0, 1)
        self.minDoubleSpinBox.setMinimumWidth(100)
        self.minDoubleSpinBox.valueChanged.connect(self.on_minDoubleSpinBox_changed)

        # max value
        self.maxDoubleSpinBox = qt.QDoubleSpinBox()
        self.maxDoubleSpinBox.setMinimumWidth(100)
        self.maxDoubleSpinBox.setRange(0, 1)
        self.maxDoubleSpinBox.setValue(0.33)
        self.maxDoubleSpinBox.valueChanged.connect(self.on_maxDoubleSpinBox_changed)


        control_layout.addStretch()
        control_layout.addWidget(qt.QLabel('Min: '))
        control_layout.addWidget(self.minDoubleSpinBox)
        control_layout.addWidget(qt.QLabel('Max: '))
        control_layout.addWidget(self.maxDoubleSpinBox)

        layout = qt.QVBoxLayout()
        # layout.setContentsMargins(0, 0, 0, 0)
        self.graph = PlotWidget(self, backend='matplotlib')
        self.graph.getXAxis().setLabel('')
        self.graph.getYAxis().setLabel('')
        self.graph.setAxesMargins(0.0, 0.0, 0.0, 0.0)
        self.graph.setMaximumHeight(30)
        self.updateColor()

        layout.addLayout(control_layout)
        layout.addWidget(self.graph)
        self.setLayout(layout)

        self.marker_min = self.graph.addXMarker(0,
                legend="XMinMarker",
                color='white',
                text=f"min",
                draggable=True)

        self.marker_min.sigDragFinished.connect(self.on_marker_min_dragged)

        self.marker_max = self.graph.addXMarker(255/3,
                legend="XMaxMarker",
                color='white',
                text=f"max",
                constraint=lambda x, y: (x if x > self.marker_min.getPosition()[0] \
                    else self.marker_min.getPosition()[0], y),
                draggable=True)
        self.marker_max.sigDragFinished.connect(self.on_marker_max_dragged)

        self.marker_min._setConstraint(lambda x, y: (x if x < self.marker_max.getPosition()[0] \
            else self.marker_max.getPosition()[0], y))

    def on_marker_min_dragged(self):
        self.minDoubleSpinBox.blockSignals(True)
        self.minDoubleSpinBox.setValue(self.marker_min.getPosition()[0]/255)
        self.minDoubleSpinBox.blockSignals(False)

    def on_marker_max_dragged(self):
        self.maxDoubleSpinBox.blockSignals(True)
        self.maxDoubleSpinBox.setValue(self.marker_max.getPosition()[0]/255)
        self.maxDoubleSpinBox.blockSignals(False)

    def on_minDoubleSpinBox_changed(self, value):
        submit(self.marker_min.setPosition, value*255.0, 0)

    def on_maxDoubleSpinBox_changed(self, value):
        submit(self.marker_max.setPosition, value*255.0, 0)

    def updateColor(self, index=None):
        data = createColormap(self.colorComboBox.currentText())
        self.graph.addImage(data)

    def getPosition(self):
        return self.marker_min.getPosition()[0], self.marker_max.getPosition()[0]

if __name__ == '__main__':
    app = qt.QApplication(sys.argv)
    main = ColorBarWidget()
    main.show()
    sys.exit(app.exec_())
