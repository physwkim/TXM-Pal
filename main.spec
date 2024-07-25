# -*- mode: python -*-
import os.path
from PyInstaller.utils.hooks import collect_data_files

datas = []

PROJECT_PATH = os.path.abspath(os.path.join(SPECPATH))
datas.append((os.path.join(PROJECT_PATH, "ui/main.ui"), "ui"))
datas.append((os.path.join(PROJECT_PATH, "mainicon.ico"), "."))
datas += collect_data_files("silx.resources")

icon_path = os.path.join(PROJECT_PATH, "mainicon.ico")

block_cipher = None

a = Analysis(['main.py'],
             pathex=['C:\\codes\\TXM-Pal'],
             binaries=[],
             datas=datas,
			 hiddenimports=['silx', 'plot1D', 'plot2D', 'Profile', 'actions', 'colorbar', 'StackView', 'widgets','silx.gui.widgets.ThreadPoolPushButton', 'fabio.fabioimage', 'fabio.utils', 'fabio.file_series', 'fabio.openimage', 'fabio.adscimage', 'fabio.binaryimage', 'fabio.bruker100image', 'fabio.brukerimage', 'fabio.cbfimage', 'fabio.dm3image', 'fabio.edfimage', 'fabio.eigerimage', 'fabio.fit2dimage', 'fabio.fit2dmaskimage', 'fabio.fit2dspreadsheetimage', 'fabio.GEimage', 'fabio.hdf5image', 'fabio.HiPiCimage', 'fabio.kcdimage', 'fabio.mar345image','fabio.mrcimage','fabio.marccdimage','fabio.numpyimage','fabio.OXDimage', 'fabio.pilatusimage','fabio.pixiimage', 'fabio.pnmimage', 'fabio.raxisimage', 'fabio.tifimage','fabio.xsdimage', 'fabio.compression','fabio.converters', 'fabio.datIO', 'fabio.TiffIO', 'fabio.readbytestream', 'fabio.mpaimage'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='main',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True,
		  icon=icon_path)
