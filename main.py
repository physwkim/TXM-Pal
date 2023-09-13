import sys

from silx.gui import qt
from silx.gui.utils.concurrent import submitToQtMainThread as _submit

class Main(qt.QMainWindow):
    """Main Window"""

    hided = qt.Signal(object)
    closed = qt.Signal(object)

    sigStop = qt.Signal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        qt.loadUi('ui/main.ui', self)


if __name__ == '__main__':
    app = qt.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())

