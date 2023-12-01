import os
import re
import numpy as np

from silx.gui import qt
from silx.gui import icons
from silx.gui.plot.PlotWindow import PlotWindow

import logging

_logger = logging.getLogger(__name__)

class Plot1D(PlotWindow):
    """PlotWindow with tools specific for curves.

    This widgets provides the plot API of :class:`.PlotWidget`.

    :param parent: The parent of this widget
    :param backend: The backend to use for the plot (default: matplotlib).
                    See :class:`.PlotWidget` for the list of supported backend.
    :type backend: str or :class:`BackendBase.BackendBase`
    """

    def __init__(self, parent=None, backend=None):
        super(Plot1D, self).__init__(parent=parent, backend=backend,
                                     resetzoom=True, autoScale=False,
                                     logScale=True, grid=True,
                                     curveStyle=True, colormap=False,
                                     aspectRatio=False, yInverted=False,
                                     copy=True, save=True, print_=True,
                                     control=True, position=True,
                                     roi=False, mask=False, fit=False)

        # Retrieve PlotWidget's plot area widget
        plotArea = self.getWidgetHandle()

        # Data margins
        self.setDataMargins(0.01, 0.01, 0.01, 0.01)

        self._path = ''
        self._filename = ''

        self.setDefaultPlotPoints(False)
        self.getGridAction().setChecked(False)
        self.setGraphGrid(False)

    @property
    def path(self):
        return self._path

    @property
    def filename(self):
        return self._filename

    def setPath(self, path):
        self._path = path

    def setFilename(self, filename):
        self._filename = filename

    def resetZoom(self, dataMargins=None):
        """Reset the plot limits to the bounds of the data and redraw the plot.

        It automatically scale limits of axes that are in autoscale mode
        (see :meth:`getXAxis`, :meth:`getYAxis` and :meth:`Axis.setAutoScale`).
        It keeps current limits on axes that are not in autoscale mode.

        Extra margins can be added around the data inside the plot area
        (see :meth:`setDataMargins`).
        Margins are given as one ratio of the data range per limit of the
        data (xMin, xMax, yMin and yMax limits).
        For log scale, extra margins are applied in log10 of the data.

        :param dataMargins: Ratios of margins to add around the data inside
                            the plot area for each side (default: no margins).
        :type dataMargins: A 4-tuple of float as (xMin, xMax, yMin, yMax).

        Changed zoom history to be deleted when resetZoom is called.
        """
        xLimits = self._xAxis.getLimits()
        yLimits = self._yAxis.getLimits()
        y2Limits = self._yRightAxis.getLimits()

        xAuto = self._xAxis.isAutoScale()
        yAuto = self._yAxis.isAutoScale()

        # With log axes, autoscale if limits are <= 0
        # This avoids issues with toggling log scale with matplotlib 2.1.0
        if self._xAxis.getScale() == self._xAxis.LOGARITHMIC and xLimits[0] <= 0:
            xAuto = True
        if self._yAxis.getScale() == self._yAxis.LOGARITHMIC and (yLimits[0] <= 0 or y2Limits[0] <= 0):
            yAuto = True

        if not xAuto and not yAuto:
            _logger.debug("Nothing to autoscale")
        else:  # Some axes to autoscale
            self._forceResetZoom(dataMargins=dataMargins)

            # Restore limits for axis not in autoscale
            if not xAuto and yAuto:
                self.setGraphXLimits(*xLimits)
            elif xAuto and not yAuto:
                if y2Limits is not None:
                    self.setGraphYLimits(
                        y2Limits[0], y2Limits[1], axis='right')
                if yLimits is not None:
                    self.setGraphYLimits(yLimits[0], yLimits[1], axis='left')

        if (xLimits != self._xAxis.getLimits() or
                yLimits != self._yAxis.getLimits() or
                y2Limits != self._yRightAxis.getLimits()):
            self._notifyLimitsChanged()

        # Changed Zoom history to be deleted when resetZoom is called.
        self._limitsHistory.clear()

class Plot1DHistogram(PlotWindow):
    def __init__(self, parent=None, backend=None):
        super(Plot1DHistogram, self).__init__(parent=parent, backend=backend,
                                           resetzoom=True, autoScale=False,
                                           logScale=True, grid=True,
                                           curveStyle=True, colormap=False,
                                           aspectRatio=False, yInverted=False,
                                           copy=True, save=False, print_=True,
                                           control=True, position=True,
                                           roi=False, mask=False, fit=False)
        # Retrieve PlotWidget's plot area widget
        plotArea = self.getWidgetHandle()

        # Data margins
        self.setDataMargins(0.01, 0.01, 0.01, 0.01)

        self._path = ''
        self._filename = ''

        self.setDefaultPlotPoints(False)
        self.getGridAction().setChecked(False)
        self.setGraphGrid(False)

        self._saveAction = qt.QAction(icons.getQIcon('document-save'), 'Save', self)
        self._saveAction.setCheckable(False)
        self._saveAction.triggered.connect(self.savePlot)
        self._outputToolBar.addAction(self._saveAction)

    @property
    def path(self):
        return self._path

    @property
    def filename(self):
        return self._filename

    def setPath(self, path):
        self._path = path

    def setFilename(self, filename):
        self._filename = filename

    def savePlot(self):
        """Save histogram to a file."""
        data = self.getHistogram().getData()
        stack = np.stack([data[1], np.append(data[0], 0)]).T

        selected_file = qt.QFileDialog.getSaveFileName(self,
                                                       'Save File',
                                                       self.path,
                                                       'dat (*.dat)')
        # save curves
        print(f"histogram save!!! : {stack}")
        np.savetxt(selected_file[0]+'.dat', stack, fmt='%.8e', delimiter='\t')

    def resetZoom(self, dataMargins=None):
        """Reset the plot limits to the bounds of the data and redraw the plot.

        It automatically scale limits of axes that are in autoscale mode
        (see :meth:`getXAxis`, :meth:`getYAxis` and :meth:`Axis.setAutoScale`).
        It keeps current limits on axes that are not in autoscale mode.

        Extra margins can be added around the data inside the plot area
        (see :meth:`setDataMargins`).
        Margins are given as one ratio of the data range per limit of the
        data (xMin, xMax, yMin and yMax limits).
        For log scale, extra margins are applied in log10 of the data.

        :param dataMargins: Ratios of margins to add around the data inside
                            the plot area for each side (default: no margins).
        :type dataMargins: A 4-tuple of float as (xMin, xMax, yMin, yMax).

        Changed zoom history to be deleted when resetZoom is called.
        """
        xLimits = self._xAxis.getLimits()
        yLimits = self._yAxis.getLimits()
        y2Limits = self._yRightAxis.getLimits()

        xAuto = self._xAxis.isAutoScale()
        yAuto = self._yAxis.isAutoScale()

        # With log axes, autoscale if limits are <= 0
        # This avoids issues with toggling log scale with matplotlib 2.1.0
        if self._xAxis.getScale() == self._xAxis.LOGARITHMIC and xLimits[0] <= 0:
            xAuto = True
        if self._yAxis.getScale() == self._yAxis.LOGARITHMIC and (yLimits[0] <= 0 or y2Limits[0] <= 0):
            yAuto = True

        if not xAuto and not yAuto:
            _logger.debug("Nothing to autoscale")
        else:  # Some axes to autoscale
            self._forceResetZoom(dataMargins=dataMargins)

            # Restore limits for axis not in autoscale
            if not xAuto and yAuto:
                self.setGraphXLimits(*xLimits)
            elif xAuto and not yAuto:
                if y2Limits is not None:
                    self.setGraphYLimits(
                        y2Limits[0], y2Limits[1], axis='right')
                if yLimits is not None:
                    self.setGraphYLimits(yLimits[0], yLimits[1], axis='left')

        if (xLimits != self._xAxis.getLimits() or
                yLimits != self._yAxis.getLimits() or
                y2Limits != self._yRightAxis.getLimits()):
            self._notifyLimitsChanged()

        # Changed Zoom history to be deleted when resetZoom is called.
        self._limitsHistory.clear()


class Plot1DCustom(PlotWindow):
    """PlotWindow with tools specific for curves.

    This widgets provides the plot API of :class:`.PlotWidget`.

    :param parent: The parent of this widget
    :param backend: The backend to use for the plot (default: matplotlib).
                    See :class:`.PlotWidget` for the list of supported backend.
    :type backend: str or :class:`BackendBase.BackendBase`
    """

    def __init__(self, parent=None, backend=None):
        super(Plot1DCustom, self).__init__(parent=parent, backend=backend,
                                           resetzoom=True, autoScale=False,
                                           logScale=True, grid=True,
                                           curveStyle=True, colormap=False,
                                           aspectRatio=False, yInverted=False,
                                           copy=True, save=False, print_=True,
                                           control=True, position=True,
                                           roi=False, mask=False, fit=False)

        # Retrieve PlotWidget's plot area widget
        plotArea = self.getWidgetHandle()

        # Data margins
        self.setDataMargins(0.01, 0.01, 0.01, 0.01)

        self._path = ''
        self._filename = ''

        self.setDefaultPlotPoints(False)
        self.getGridAction().setChecked(False)
        self.setGraphGrid(False)

        self._saveAction = qt.QAction(icons.getQIcon('document-save'), 'Save', self)
        self._saveAction.setCheckable(False)
        self._saveAction.triggered.connect(self.savePlots)
        self._outputToolBar.addAction(self._saveAction)

    @property
    def path(self):
        return self._path

    @property
    def filename(self):
        return self._filename

    def setPath(self, path):
        self._path = path

    def setFilename(self, filename):
        self._filename = filename

    def savePlots(self):
        """Save plots to file."""
        curve_names = []
        data = []
        header = ''

        curves = self.getAllCurves()
        if len(curves) > 0:
            for idx, curve in enumerate(curves):
                curve_names.append(curve.getName())

            curve_names = sorted(curve_names)
            header = 'Energy\t' + '\t'.join(curve_names)

            re_pat = re.compile(r"[0-9]+")

            print(f"path : {self.path}, filename : {self.filename}")
            last_num = -1
            with os.scandir(self.path) as it:
                for item in it:
                    if item.is_file() and item.name.find(self.filename) >= -1:
                        idx = item.name[:-4].split('_spectrum_')[-1]
                        if re_pat.match(idx):

                            try:
                                if int(idx) > last_num:
                                    last_num = int(idx)
                            except:
                                ...
                num = last_num + 1

            save_file = os.path.join(self.path, f"{self.filename}_spectrum_{num:d}.dat")

            for idx, curve_name in enumerate(curve_names):
                if idx == 0:
                    data.append(self.getCurve(curve_name).getXData())
                data.append(self.getCurve(curve_name).getYData())
            data = np.array(data).transpose()

            # save curves
            np.savetxt(save_file, data, header=header, fmt='%.4e', delimiter='\t')

    def resetZoom(self, dataMargins=None):
        """Reset the plot limits to the bounds of the data and redraw the plot.

        It automatically scale limits of axes that are in autoscale mode
        (see :meth:`getXAxis`, :meth:`getYAxis` and :meth:`Axis.setAutoScale`).
        It keeps current limits on axes that are not in autoscale mode.

        Extra margins can be added around the data inside the plot area
        (see :meth:`setDataMargins`).
        Margins are given as one ratio of the data range per limit of the
        data (xMin, xMax, yMin and yMax limits).
        For log scale, extra margins are applied in log10 of the data.

        :param dataMargins: Ratios of margins to add around the data inside
                            the plot area for each side (default: no margins).
        :type dataMargins: A 4-tuple of float as (xMin, xMax, yMin, yMax).

        Changed zoom history to be deleted when resetZoom is called.
        """
        xLimits = self._xAxis.getLimits()
        yLimits = self._yAxis.getLimits()
        y2Limits = self._yRightAxis.getLimits()

        xAuto = self._xAxis.isAutoScale()
        yAuto = self._yAxis.isAutoScale()

        # With log axes, autoscale if limits are <= 0
        # This avoids issues with toggling log scale with matplotlib 2.1.0
        if self._xAxis.getScale() == self._xAxis.LOGARITHMIC and xLimits[0] <= 0:
            xAuto = True
        if self._yAxis.getScale() == self._yAxis.LOGARITHMIC and (yLimits[0] <= 0 or y2Limits[0] <= 0):
            yAuto = True

        if not xAuto and not yAuto:
            _logger.debug("Nothing to autoscale")
        else:  # Some axes to autoscale
            self._forceResetZoom(dataMargins=dataMargins)

            # Restore limits for axis not in autoscale
            if not xAuto and yAuto:
                self.setGraphXLimits(*xLimits)
            elif xAuto and not yAuto:
                if y2Limits is not None:
                    self.setGraphYLimits(
                        y2Limits[0], y2Limits[1], axis='right')
                if yLimits is not None:
                    self.setGraphYLimits(yLimits[0], yLimits[1], axis='left')

        if (xLimits != self._xAxis.getLimits() or
                yLimits != self._yAxis.getLimits() or
                y2Limits != self._yRightAxis.getLimits()):
            self._notifyLimitsChanged()

        # Changed Zoom history to be deleted when resetZoom is called.
        self._limitsHistory.clear()
