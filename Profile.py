import numpy
from silx.gui import qt
from silx.gui.plot.Profile import ProfileToolBar
from silx.gui.plot.tools.profile import rois
from silx.gui.plot.tools.profile import core
from silx.gui.plot import items
from silx.gui.plot.tools.profile.rois import _relabelAxes, _lineProfileTitle

class MyProfileImageHorizontalLineROI(rois.ProfileImageHorizontalLineROI):
    def __init__(self, parent=None):
        super(MyProfileImageHorizontalLineROI, self).__init__(parent)

    def computeProfile(self, item):
        if not isinstance(item, items.ImageBase):
            raise TypeError("Unexpected class %s" % type(item))

        origin = item.getOrigin()
        scale = item.getScale()
        method = self.getProfileMethod()
        lineWidth = self.getProfileLineWidth()
        roiInfo = self._getRoiInfo()

        def createProfile2(currentData):
            coords, profile, _area, profileName, xLabel = core.createProfile(
                roiInfo=roiInfo,
                currentData=currentData,
                origin=origin,
                scale=scale,
                lineWidth=lineWidth,
                method=method)
            return coords, profile, profileName, xLabel

        if isinstance(item, items.ImageRgba):
            currentData = item.getPlot().fitImage
        else:
            currentData = item.getData(copy=False)

        yLabel = "%s" % str(method).capitalize()
        coords, profile, title, xLabel = createProfile2(currentData)
        title = title + "; width = %d" % lineWidth

        # Use the axis names from the original plot
        profileManager = self.getProfileManager()
        plot = profileManager.getPlotWidget()
        title = _relabelAxes(plot, title)
        xLabel = _relabelAxes(plot, xLabel)

        data = core.CurveProfileData(
            coords=coords,
            profile=profile[0],
            title=title,
            xLabel=xLabel,
            yLabel=yLabel,
        )

        return data

class MyProfileImageVerticalLineROI(rois.ProfileImageVerticalLineROI):
    def __init__(self, parent=None):
        super(MyProfileImageVerticalLineROI, self).__init__(parent)

    def computeProfile(self, item):
        if not isinstance(item, items.ImageBase):
            raise TypeError("Unexpected class %s" % type(item))

        origin = item.getOrigin()
        scale = item.getScale()
        method = self.getProfileMethod()
        lineWidth = self.getProfileLineWidth()
        roiInfo = self._getRoiInfo()

        def createProfile2(currentData):
            coords, profile, _area, profileName, xLabel = core.createProfile(
                roiInfo=roiInfo,
                currentData=currentData,
                origin=origin,
                scale=scale,
                lineWidth=lineWidth,
                method=method)
            return coords, profile, profileName, xLabel

        if isinstance(item, items.ImageRgba):
            currentData = item.getPlot().fitImage
        else:
            currentData = item.getData(copy=False)

        yLabel = "%s" % str(method).capitalize()
        coords, profile, title, xLabel = createProfile2(currentData)
        title = title + "; width = %d" % lineWidth

        # Use the axis names from the original plot
        profileManager = self.getProfileManager()
        plot = profileManager.getPlotWidget()
        title = _relabelAxes(plot, title)
        xLabel = _relabelAxes(plot, xLabel)

        data = core.CurveProfileData(
            coords=coords,
            profile=profile[0],
            title=title,
            xLabel=xLabel,
            yLabel=yLabel,
        )

        return data

class MyProfileImageLineROI(rois.ProfileImageLineROI):
    def __init__(self, parent=None):
        super(MyProfileImageLineROI, self).__init__(parent)

    def computeProfile(self, item):
        if not isinstance(item, items.ImageBase):
            raise TypeError("Unexpected class %s" % type(item))

        origin = item.getOrigin()
        scale = item.getScale()
        method = self.getProfileMethod()
        lineWidth = self.getProfileLineWidth()
        roiInfo = self._getRoiInfo()

        def createProfile2(currentData):
            coords, profile, _area, profileName, xLabel = core.createProfile(
                roiInfo=roiInfo,
                currentData=currentData,
                origin=origin,
                scale=scale,
                lineWidth=lineWidth,
                method=method)
            return coords, profile, profileName, xLabel

        if isinstance(item, items.ImageRgba):
            currentData = item.getPlot().fitImage
        else:
            currentData = item.getData(copy=False)

        yLabel = "%s" % str(method).capitalize()
        coords, profile, title, xLabel = createProfile2(currentData)
        title = title + "; width = %d" % lineWidth

        # Use the axis names from the original plot
        profileManager = self.getProfileManager()
        plot = profileManager.getPlotWidget()
        title = _relabelAxes(plot, title)
        xLabel = _relabelAxes(plot, xLabel)

        data = core.CurveProfileData(
            coords=coords,
            profile=profile[0],
            title=title,
            xLabel=xLabel,
            yLabel=yLabel,
        )

        return data

class MyProfileImageDirectedLineROI(rois.ProfileImageDirectedLineROI):
    def __init__(self, parent=None):
        super(MyProfileImageDirectedLineROI, self).__init__(parent)

    def computeProfile(self, item):
        if not isinstance(item, items.ImageBase):
            raise TypeError("Unexpected class %s" % type(item))

        from silx.image.bilinear import BilinearImage

        origin = item.getOrigin()
        scale = item.getScale()
        method = self.getProfileMethod()
        lineWidth = self.getProfileLineWidth()
        if isinstance(item, items.ImageRgba):
            currentData = item.getPlot().fitImage
        else:
            currentData = item.getData(copy=False)

        roiInfo = self._getRoiInfo()
        roiStart, roiEnd, _lineProjectionMode = roiInfo

        startPt = ((roiStart[1] - origin[1]) / scale[1],
                   (roiStart[0] - origin[0]) / scale[0])
        endPt = ((roiEnd[1] - origin[1]) / scale[1],
                 (roiEnd[0] - origin[0]) / scale[0])

        if numpy.array_equal(startPt, endPt):
            return None

        bilinear = BilinearImage(currentData)
        profile = bilinear.profile_line(
            (startPt[0] - 0.5, startPt[1] - 0.5),
            (endPt[0] - 0.5, endPt[1] - 0.5),
            lineWidth,
            method=method)

        # Compute the line size
        lineSize = numpy.sqrt((roiEnd[1] - roiStart[1]) ** 2 +
                              (roiEnd[0] - roiStart[0]) ** 2)
        coords = numpy.linspace(0, lineSize, len(profile),
                                endpoint=True,
                                dtype=numpy.float32)

        title = _lineProfileTitle(*roiStart, *roiEnd)
        title = title + "; width = %d" % lineWidth
        xLabel = "√({xlabel}²+{ylabel}²)"
        yLabel = str(method).capitalize()

        # Use the axis names from the original plot
        profileManager = self.getProfileManager()
        plot = profileManager.getPlotWidget()
        xLabel = _relabelAxes(plot, xLabel)
        title = _relabelAxes(plot, title)

        data = core.CurveProfileData(
            coords=coords,
            profile=profile,
            title=title,
            xLabel=xLabel,
            yLabel=yLabel,
        )
        return data

class MyProfileImageCrossROI(rois.ProfileImageCrossROI):
    def __init__(self, parent=None):
        super(MyProfileImageCrossROI, self).__init__(parent)

    def _createLines(self, parent):
        vline = MyProfileImageVerticalLineROI(parent=parent)
        hline = MyProfileImageHorizontalLineROI(parent=parent)
        return hline, vline


class MyProfileToolBar(ProfileToolBar):
    """Profile toolbar with a custom action"""

    def __init__(self, plot=None):
        super(MyProfileToolBar, self).__init__(plot=plot)

    def _createProfileActions(self):
        self.hLineAction = self._manager.createProfileAction(MyProfileImageHorizontalLineROI, self)
        self.vLineAction = self._manager.createProfileAction(MyProfileImageVerticalLineROI, self)
        self.lineAction = self._manager.createProfileAction(MyProfileImageLineROI, self)
        self.freeLineAction = self._manager.createProfileAction(MyProfileImageDirectedLineROI, self)
        self.crossAction = self._manager.createProfileAction(MyProfileImageCrossROI, self)
        self.clearAction = self._manager.createClearAction(self)
