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
