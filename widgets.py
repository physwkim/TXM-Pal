from silx.gui import qt

class OddSpinBox(qt.QSpinBox):
    """A spin box that only allows odd numbers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valueChanged.connect(self.ensure_odd)

    def ensure_odd(self, value):
        if value % 2 == 0:
            if value == self.maximum():
                self.setValue(value - 1)
            else:
                self.setValue(value + 1)

    def stepBy(self, steps):
        super().stepBy(steps*2)


class MainToolBar(qt.QToolBar):
    """
    Toolbar composed of labeld buttons
    """
    sigSetPannel = qt.Signal(object)

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # Setup font
        self.font = qt.QFont("Verdana", 12)

        # Mode action group
        self.actionGroup = qt.QActionGroup(self)

        # Align right
        self.setLayoutDirection(qt.Qt.LeftToRight)

        # Build actions
        self.buildActions()

    def buildActions(self):

        # Make pre-processing button
        action = qt.QAction("Preprocessing", self)
        action.setFont(self.font)
        action.setCheckable(True)
        action.setActionGroup(self.actionGroup)
        action.triggered.connect(lambda: self.sigSetPannel.emit("Preprocessing"))
        self.addAction(action)
        action.trigger()

        # Make spearator
        label = qt.QAction(" | ", self)
        label.setFont(self.font)
        label.setDisabled(True)
        self.addAction(label)

        # Make fitting button
        action = qt.QAction("Fitting", self)
        action.setFont(self.font)
        action.setCheckable(True)
        action.setActionGroup(self.actionGroup)
        action.triggered.connect(lambda: self.sigSetPannel.emit("Fitting"))
        self.addAction(action)

        # Make spearator
        label = qt.QAction(" | ", self)
        label.setFont(self.font)
        label.setDisabled(True)
        self.addAction(label)

        # Make Spectrum Viewer button
        action = qt.QAction("Spectrum Viewer", self)
        action.setFont(self.font)
        action.setCheckable(True)
        action.setActionGroup(self.actionGroup)
        action.triggered.connect(lambda: self.sigSetPannel.emit("Spectrum Viewer"))
        self.addAction(action)
