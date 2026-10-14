"""Extended native desktop with catalog, culling, scopes and original spectral engine."""
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage,QPixmap
from PySide6.QtWidgets import (QApplication,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QFileDialog,QComboBox,QListWidgetItem,QTabWidget,QMessageBox)
from .app import Window as FilmWindow
from .catalog import Catalog
from . import scopes

class Studio(FilmWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Spectral Studio | Library · Cull · Film · Scopes')
        self.catalog=Catalog(Path.home()/'.spectral_studio_catalog.sqlite')
        self.current=None
        self.filter=QComboBox();self.filter.addItems(['All','Picks','Rejected','Unflagged','Rated'])
        self.filter.currentTextChanged.connect(self.refresh_library)
        library=self.files.parentWidget()
        if isinstance(library,QTabWidget):
            controls=QWidget();v=QVBoxLayout(controls)
            folder=QPushButton('Import Folder');folder.clicked.connect(self.import_folder);v.addWidget(folder)
            v.addWidget(QLabel('Filter'));v.addWidget(self.filter)
            v.addWidget(QLabel('Select an image to rate or flag'))
            for text,value in [('Pick',1),('Reject',-1),('Clear Flag',0)]:
                b=QPushButton(text);b.clicked.connect(lambda _,val=value:self.set_pick(val));v.addWidget(b)
            row=QHBoxLayout();v.addLayout(row)
            for rating in range(6):
                b=QPushButton(str(rating));b.clicked.connect(lambda _,r=rating:self.set_rating(r));row.addWidget(b)
            v.addStretch();library.addTab(controls,'Cull')
            self.scopes_label=QLabel('Render a film image to measure output.');self.scopes_label.setWordWrap(True)
            self.scope_image=QLabel();self.scope_image.setMinimumSize(200,200)
            scope_page=QWidget();scope_layout=QVBoxLayout(scope_page);scope_layout.addWidget(self.scopes_label);scope_layout.addWidget(self.scope_image);library.addTab(scope_page,'Scopes')
        self.files.currentItemChanged.connect(self.select_item)
        self.refresh_library()
    def import_folder(self):
        folder=QFileDialog.getExistingDirectory(self,'Import photo folder')
        if folder:
            count=self.catalog.import_folder(folder)
            self.refresh_library();self.statusBar().showMessage(f'Cataloged {count} supported files (RAW decoding available; film color transform pending)')
    def refresh_library(self,*_):
        self.files.clear()
        for path,rating,pick in self.catalog.entries(self.filter.currentText()):
            icon='✓' if pick==1 else '×' if pick==-1 else ' '
            item=QListWidgetItem(f'{icon} {"★"*rating} {Path(path).name}')
            item.setData(Qt.ItemDataRole.UserRole,path)
            self.files.addItem(item)
    def select_item(self,item,*_):
        if item:self.current=item.data(Qt.ItemDataRole.UserRole)
    def set_pick(self,value):
        if self.current:self.catalog.flag(self.current,value);self.refresh_library()
    def set_rating(self,value):
        if self.current:self.catalog.rate(self.current,value);self.refresh_library()
    def load_path(self,path):
        super().load_path(path)
        if self.source is not None:
            self.current=path
    def render_done(self,result):
        super().render_done(result)
        if self.result is None:return
        h=scopes.histogram(result)
        white,black=scopes.clipping(result)
        # Scope arrays are computed from the same buffer used by the film preview.
        wave=scopes.waveform(result)
        parade=scopes.parade(result)
        vector=scopes.vectorscope(result)
        self.scopes_label.setText(f'Output-linear RGB (not display-encoded)\nWhite clipped: {white:.3f}%\nBlack clipped: {black:.3f}%\nHistogram channel samples: {h.sum(axis=1).tolist()}\nWaveform max density: {wave.max()}\nRGB parade max density: {parade.max()}\nVectorscope max density: {int(vector.max())}\nWaveform shown below')
        strength=np.log1p(wave.astype(np.float32))
        strength/=max(float(strength.max()),1.0)
        pixels=np.ascontiguousarray((strength*255).astype(np.uint8))
        img=QImage(pixels.data,pixels.shape[1],pixels.shape[0],pixels.shape[1],QImage.Format.Format_Grayscale8).copy()
        self.scope_image.setPixmap(QPixmap.fromImage(img).scaled(256,256,Qt.AspectRatioMode.KeepAspectRatio))

def main():
    import sys
    app=QApplication(sys.argv)
    window=Studio();window.show()
    return app.exec()
