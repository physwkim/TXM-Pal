
import numpy
import tifffile
from silx.third_party.EdfFile import EdfFile
from silx.io.nxdata import save_NXdata
from silx.gui.plot.actions.io import SaveAction as _SaveAction
from silx.gui.utils.image import convertArrayToQImage


class SaveAction(_SaveAction):
    """QAction for saving Plot content.

    customized for tifffile saving

    It opens a Save as... dialog.
    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See : class:`QAction`.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        data = image.getData(copy=False)

        # TODO Use silx.io for writing files
        if nameFilter == self.IMAGE_FILTER_EDF:
            edfFile = EdfFile(filename, access="w+")
            edfFile.WriteImage({}, data, Append=0)
            return True

        elif nameFilter == self.IMAGE_FILTER_TIFF:
            # KSW customized
            # ORIG tiffFile = TiffIO(filename, mode='w')
            # ORIG tiffFile.writeImage(data, software='silx')

            # convert to uint16
            data = numpy.array(data, dtype=numpy.uint16)
            tifffile.imwrite(filename, data)
            return True

        elif nameFilter == self.IMAGE_FILTER_NUMPY:
            try:
                numpy.save(filename, data)
            except IOError:
                self._errorMessage('Save failed\n', parent=self.plot)
                return False
            return True

        elif nameFilter == self.IMAGE_FILTER_NXDATA:
            entryPath = self._selectWriteableOutputGroup(filename, parent=self.plot)
            if entryPath is None:
                return False
            xorigin, yorigin = image.getOrigin()
            xscale, yscale = image.getScale()
            xaxis = xorigin + xscale * numpy.arange(data.shape[1])
            yaxis = yorigin + yscale * numpy.arange(data.shape[0])
            xlabel, ylabel = self._getAxesLabels(image)
            interpretation = "image" if len(data.shape) == 2 else "rgba-image"

            return save_NXdata(filename,
                               nxentry_name=entryPath,
                               signal=data,
                               axes=[yaxis, xaxis],
                               signal_name="image",
                               axes_names=["y", "x"],
                               axes_long_names=[ylabel, xlabel],
                               title=plot.getGraphTitle(),
                               interpretation=interpretation)

        elif nameFilter in (self.IMAGE_FILTER_ASCII,
                            self.IMAGE_FILTER_CSV_COMMA,
                            self.IMAGE_FILTER_CSV_SEMICOLON,
                            self.IMAGE_FILTER_CSV_TAB):
            csvdelim, filetype = {
                self.IMAGE_FILTER_ASCII: (' ', 'txt'),
                self.IMAGE_FILTER_CSV_COMMA: (',', 'csv'),
                self.IMAGE_FILTER_CSV_SEMICOLON: (';', 'csv'),
                self.IMAGE_FILTER_CSV_TAB: ('\t', 'csv'),
                }[nameFilter]

            height, width = data.shape
            rows, cols = numpy.mgrid[0:height, 0:width]
            try:
                save1D(filename, rows.ravel(), (cols.ravel(), data.ravel()),
                       filetype=filetype,
                       xlabel='row',
                       ylabels=['column', 'value'],
                       csvdelim=csvdelim,
                       autoheader=True)

            except IOError:
                self._errorMessage('Save failed\n', parent=self.plot)
                return False
            return True

        elif nameFilter == self.IMAGE_FILTER_RGB_PNG:
            # Get displayed image
            rgbaImage = image.getRgbaImageData(copy=False)
            # Convert RGB QImage
            qimage = convertArrayToQImage(rgbaImage[:, :, :3])

            if qimage.save(filename, 'PNG'):
                return True
            else:
                _logger.error('Failed to save image as %s', filename)
                qt.QMessageBox.critical(
                    self.parent(),
                    'Save image as',
                    'Failed to save image')

        return False


