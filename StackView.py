import numpy
from silx.gui import qt
from silx.gui.plot.StackView import StackView as _StackView
from silx.gui.plot.StackView import StackViewMainWindow as _StackViewMainWindow
from silx.gui.plot.tools import LimitsToolBar
from silx.gui.plot import actions
from silx.utils.array_like import DatasetView, ListOfImages
from silx.io.utils import is_dataset
from silx.utils.weakref import WeakMethodProxy
from silx.gui.widgets.FrameBrowser import HorizontalSliderWithBrowser
from silx.gui.utils.concurrent import submitToQtMainThread as _submit

# customized Plot2D
from plot2D import Plot2D

class StackView(qt.QWidget):
    """ Customized StackView from silx.gui.plot.StackView """

    # Signal emitted when the stack is changed.
    sigStackChanged = qt.Signal(int)

    # Signal emitter when the frame number has changed.
    sigFrameChanged = qt.Signal(int)

    sigEnergyKeV= qt.Signal(float)

    def __init__(self, parent=None, *args, **kwargs):
        super(StackView, self).__init__(parent, *args, **kwargs)

        # Datafile
        self._stack = None

        self._plot = Plot2D(self)
        self._plot.getXAxis().setLabel('')
        self._plot.getYAxis().setLabel('')
        self._plot.setKeepDataAspectRatio()
        self.setRoiEditable = self._plot.setRoiEditable

        self._auto_update = True
        self._reset = False

        self._energy_list = []

        # Stack browser
        self._browser_label = qt.QLabel("Stack:")
        self._browser = HorizontalSliderWithBrowser(self)
        self._browser.setRange(0, 0)
        self._browser.valueChanged[int].connect(self.__updateFrameNumber)
        self._browser.setEnabled(False)

        layout = qt.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot, 0, 0 , 1, 3)
        layout.addWidget(self._browser_label, 1, 0)
        layout.addWidget(self._browser, 1, 1)

        # central_widget.setLayout(layout)
        # self.setCentralWidget(central_widget)
        self.setLayout(layout)

    def setFitImage(self, fit_image):
        self._plot.fitImage = fit_image

    def setPath(self, path):
        self._plot.setPath(path)

    def setFilename(self, filename):
        self._plot.setFilename(filename)

    def resetZoom(self):
        """Reset the plot limits to the bounds of the data and redraw the plot.

        This method is a simple proxy to the legacy :class:`PlotWidget` method
        of the same name. Using the object oriented approach is now
        preferred::

            stackview.getPlot().resetZoom()
        """
        self._plot.resetZoom()

    def getPlotWidget(self):
        """Return the :class:`PlotWidget`.

        This gives access to advanced plot configuration options.
        Be warned that modifying the plot can cause issues, and some changes
        you make to the plot could be overwritten by the :class:`StackView`
        widget's internal methods and callbacks.

        :return: instance of :class:`PlotWidget` used in widget
        """
        return self._plot

    @property
    def autoUpdate(self):
        return self._auto_update

    @autoUpdate.setter
    def autoUpdate(self, value):
        self._auto_update = value

    def clear(self):
        """Clear the widget:

         - clear the plot
         - clear the loaded data volume
        """
        self._stack = None
        _submit(self._browser.setEnabled, False)
        # reset browser range
        _submit(self._browser.setRange, 0, 0)
        _submit(self._plot.clear)

    def setStack(self, stack, reset=False):
        """Set the 3D stack. """

        # Save current stack index
        self._browser_index = self._browser.value()

        if stack is None:
            _submit(self.clear)
            _submit(self.sigStackChanged.emit, 0)
            return

        # stack as list of 2D arrays: must be converted into an array_like
        if not isinstance(stack, numpy.ndarray):
            if not is_dataset(stack):
                try:
                    assert hasattr(stack, "__len__")
                    for img in stack:
                        assert hasattr(img, "shape")
                        assert len(img.shape) == 2
                except AssertionError:
                    raise ValueError(
                        "Stack must be a 3D array/dataset or a list of " +
                        "2D arrays.")
                stack = ListOfImages(stack)

        if len(stack.shape) == 2:
            stack = numpy.array([stack])

        assert len(stack.shape) == 3, "data must be 3D"

        self._stack = stack

        # init plot
        _submit(self._updateTitle)

        if stack is not None:
            size = len(self._stack)

            # enable and init browser
            _submit(self._browser.setEnabled, True)
            _submit(self._browser.setRange, 0, size-1)

            # display last image
            if self.autoUpdate:
                _submit(self._browser.setValue, size-1)
                _submit(self.setCurrentIndex, size-1)
            else:
                if self._browser_index > size-1:
                    _submit(self._browser.setValue, size-1)
                else:
                    _submit(self._browser.setValue, self._browser_index)

                    if self._browser.value() == self._browser_index:
                        _submit(self.setCurrentIndex, self._browser_index)

            if reset:
                _submit(self._plot.resetZoom)
        else:
            _submit(self._browser.setEnabled, False)
            _submit(self._browser.setRange, 0, 0)
            _submit(self._plot.clear)

    def setCurrentIndex(self, index, reset=False):
        """ Set current index of an image from the stack """
        size = len(self._stack)
        index = numpy.clip(index, 0, size)
        _submit(self._plot.addImage, self._stack[index], copy=False, resetzoom=reset)

    def __updateFrameNumber(self, index):
        """Update the current image.

        :param index: index of the frame to be displayed
        """
        self.setCurrentIndex(index, self._reset)
        self._updateTitle()
        self.sigFrameChanged.emit(index)

    def _updateTitle(self):
        frame_idx = self._browser.value()
        if len(self._energy_list) > frame_idx:
            energy = self._energy_list[frame_idx]
            self.sigEnergyKeV.emit(energy)
            _submit(self._plot.setGraphTitle, f"image {frame_idx:d} [{energy:.2f} eV]")
        else:
            _submit(self._plot.setGraphTitle, f"image {frame_idx:d}")


if __name__ == '__main__':
    app = qt.QApplication([])
    gui = StackView()
    gui.show()
    app.exec_()
