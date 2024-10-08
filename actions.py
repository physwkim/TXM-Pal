
from collections import OrderedDict
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
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_facecolor(FACECOLOR)

            qsettings = qt.QSettings('settings.ini', qt.QSettings.IniFormat)
            interp = qsettings.value('interpolate', 0)
            print(f"Interpolation: {interp}, Bool : {interp == 1}")
            interp_algo = 'antialiased' if interp else 'none'
            interp_stage = 'data' if interp else 'rgba'


            if colormap is not None:
                vmin = colormap.getVMin()
                vmax = colormap.getVMax()
                cm_name = colormap.getName()


                ax.imshow(data,
                        origin='lower',
                        cmap=cm_name,
                        vmin=vmin,
                        vmax=vmax,
                        interpolation=interp_algo,
                        interpolation_stage=interp_stage)

            else:
                ax.imshow(data,
                        origin='lower',
                        interpolation=interp_algo,
                        interpolation_stage=interp_stage)



            ax.set_title(title, fontdict={'fontsize': 30})
            plt.savefig(pngFile,
                        format='png',
                        bbox_inches='tight')

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

    DEFAULT_ALL_FILTERS = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove filters except png, tiff
        img_filters = self._filters['image']
        keys = list(img_filters.keys())

        for key in keys:
            if ('png' not in key) and ('tif' not in key):
                img_filters.pop(key)

        # Set png as default
        img_filters = OrderedDict(img_filters)
        img_filters.move_to_end('Image data as TIFF (*.tif)')

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

        if nameFilter == self.IMAGE_FILTER_TIFF:
            # convert to uint16
            data = image.getData(copy=False)
            # data = numpy.array(data, dtype=numpy.uint16)
            data = numpy.array(data, dtype=numpy.float32)
            tifffile.imwrite(filename, data)
            return True

        elif nameFilter == self.IMAGE_FILTER_RGB_PNG:
            try:
                # Retrieve data
                data = image.getData(copy=False)

                # Get colormap
                colorbar = plot.getColorBarWidget()
                colormap = colorbar.getColormap()

                pngFile = BytesIO()
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.set_facecolor(FACECOLOR)

                qsettings = qt.QSettings('settings.ini', qt.QSettings.IniFormat)
                interp = qsettings.value('interpolate', 0)
                # print(f"Interpolation: {interp}, Bool : {interp == 1}")
                interp_algo = 'antialiased' if interp else 'none'
                interp_stage = 'data' if interp else 'rgba'

                if colormap is not None:
                    vmin = colormap.getVMin()
                    vmax = colormap.getVMax()
                    cm_name = colormap.getName()

                    ax.imshow(data,
                            origin='lower',
                            cmap=cm_name,
                            vmin=vmin,
                            vmax=vmax,
                            interpolation=interp_algo,
                            interpolation_stage=interp_stage)
                else:
                    ax.imshow(data,
                            origin='lower',
                            interpolation=interp_algo,
                            interpolation_stage=interp_stage)

                ax.axis('off')
                ax.set_position([0, 0, 1, 1])
                fig.set_facecolor(FACECOLOR)

                plt.savefig(filename,
                            format='png',
                            bbox_inches='tight',
                            pad_inches=0)

            except Exception as e:
                qt.QMessageBox.critical(
                    plot, "Error",
                    "Failed to save image: %s" % e)
                return False

            return True
