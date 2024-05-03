
from io import BytesIO

import numpy
import tifffile

from silx.gui import qt
from silx.third_party.EdfFile import EdfFile
from silx.io.nxdata import save_NXdata
from silx.gui.plot.actions.io import SaveAction as _SaveAction
from silx.gui.plot.actions.PlotAction import PlotAction
from silx.gui.utils.image import convertArrayToQImage

import matplotlib.pyplot as plt

DPI=300
FACECOLOR = 'black'

class CopyAction(PlotAction):
    """QAction to copy :class: '.PlotWidget' content to clipboard.
    """

    def __init__(self, plot, parent=None):
        super(CopyAction, self).__init__(
            plot, icon='edit-copy', text='Copy plot',
            tooltip='Copy a snapshot of the plot into the clipboard',
            triggered=self.copyPlot,
            checkable=False, parent=parent)
        self.setShortcut(qt.QKeySequence.Copy)
        self.setShortcutContext(qt.Qt.WidgetShortcut)

    def copyPlot(self):
        """Copy a snapshot of the plot into the clipboard"""
        try:
            title = self.plot.getGraphTitle()
            data = self.plot.getImage().getData()

            # Get colormap
            colorbar = self.plot.getColorBarWidget()
            colormap = colorbar.getColormap()

            pngFile = BytesIO()
            fig, ax = plt.subplots(figsize=(10, 10), dpi=DPI)
            ax.set_facecolor(FACECOLOR)

            if colormap is not None:
                vmin = colormap.getVMin()
                vmax = colormap.getVMax()
                cm_name = colormap.getName()
                ax.imshow(data,
                          origin='lower',
                          cmap=cm_name,
                          vmin=vmin,
                          vmax=vmax,
                          interpolation='none',
                          interpolation_stage='rgba')
            else:
                ax.imshow(data,
                          origin='lower',
                          interpolation='none',
                          interpolation_stage='rgba')

            ax.set_title(title, fontdict={'fontsize': 30})
            plt.savefig(pngFile,
                        format='png',
                        bbox_inches='tight',
                        dpi=DPI)

            pngFile.flush()
            pngFile.seek(0)
            pngData = pngFile.read()

            image = qt.QImage.fromData(pngData, 'png')
            qt.QApplication.clipboard().setImage(image)
        except Exception as e:
            qt.QMessageBox.critical(
                self.plot, "Error",
                "Failed to copy plot to clipboard: %s" % e)


class SaveAction(_SaveAction):
    """QAction for saving Plot content.

    customized for tifffile saving

    It opens a Save as... dialog.
    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See : class:`QAction`.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove filters except png
        img_filters = self._filters['image']
        keys = list(img_filters.keys())

        for key in keys:
            if not 'png' in key:
                img_filters.pop(key)

        self._filters['image'] = img_filters

    def _saveImage(self, plot, filename, nameFilter):
        """Save an image from the plot.

        :param str filename: The name of the file to write
        :param str nameFilter: The selected name filter
        :return: False if format is not supported or save failed,
                 True otherwise.
        """
        if nameFilter not in self.DEFAULT_IMAGE_FILTERS:
            return False

        image = plot.getActiveImage()
        if image is None:
            qt.QMessageBox.warning(
                plot, "No Data", "No image to be saved")
            return False

        try:
            # Retrieve data
            data = image.getData(copy=False)

            # Get colormap
            colorbar = plot.getColorBarWidget()
            colormap = colorbar.getColormap()

            pngFile = BytesIO()
            fig, ax = plt.subplots(figsize=(10, 10), dpi=DPI)
            ax.set_facecolor(FACECOLOR)

            if colormap is not None:
                vmin = colormap.getVMin()
                vmax = colormap.getVMax()
                cm_name = colormap.getName()
                ax.imshow(data,
                          origin='lower',
                          cmap=cm_name,
                          vmin=vmin,
                          vmax=vmax,
                          interpolation='none',
                          interpolation_stage='rgba')
            else:
                ax.imshow(data,
                          origin='lower',
                          interpolation='none',
                          interpolation_stage='rgba')

            ax.axis('off')
            ax.set_position([0, 0, 1, 1])
            fig.set_facecolor(FACECOLOR)

            plt.savefig(filename,
                        format='png',
                        bbox_inches='tight',
                        pad_inches=0,
                        dpi=DPI)

        except Exception as e:
            qt.QMessageBox.critical(
                plot, "Error",
                "Failed to save image: %s" % e)
            return False

        return True



