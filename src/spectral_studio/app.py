"""Qt frontend for upstream SpektraFilm; upstream owns all spectral physics."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QLabel, QPushButton, QFileDialog, QComboBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QMessageBox, QScrollArea, QTabWidget, QListWidget)

class Worker(QObject):
    success = Signal(object)
    error = Signal(str)
    def __init__(self, image, film, paper, ev):
        super().__init__()
        self.image, self.film, self.paper, self.ev = image, film, paper, ev
    @Slot()
    def run(self):
        try:
            from spektrafilm import create_params, simulate
            params = create_params(film_profile=self.film, print_profile=self.paper)
            # Scene-linear exposure adjustment; spectral simulation remains upstream.
            output = simulate(self.image * np.float32(2.0 ** self.ev), params)
            self.success.emit(np.asarray(output))
        except Exception as exc:
            self.error.emit(f'{type(exc).__name__}: {exc}')

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Spectral Studio | Original SpektraFilm Engine')
        self.resize(1440, 900)
        self.source = self.result = self.thread = self.worker = None
        self.last_path = None
        self.setStyleSheet('''QMainWindow,QWidget{background:#191c20;color:#e9ebed} QGroupBox{border:1px solid #353b43;border-radius:7px;margin-top:12px;padding:12px} QGroupBox::title{subcontrol-origin:margin;left:12px} QPushButton,QComboBox,QDoubleSpinBox{background:#2b3037;border:1px solid #454d57;border-radius:5px;padding:7px} QPushButton:hover{background:#39434d} QTabWidget::pane{border:0} QTabBar::tab{padding:10px;background:#242930} QTabBar::tab:selected{background:#39434d}''')
        center = QWidget(); self.setCentralWidget(center)
        main = QVBoxLayout(center)
        top = QHBoxLayout(); main.addLayout(top)
        for label, callback in [('Open RAW / TIFF',self.open_image),('Render Film',self.render),('Export Float TIFF',self.export)]:
            button=QPushButton(label); button.clicked.connect(callback); top.addWidget(button)
            if label=='Render Film': self.render_button=button
        top.addStretch()
        split=QSplitter(Qt.Orientation.Horizontal); main.addWidget(split,1)
        left=QTabWidget(); split.addWidget(left)
        self.files=QListWidget(); self.files.itemDoubleClicked.connect(self.load_selected)
        left.addTab(self.files,'Library')
        self.viewer=QLabel('Open a scene-linear, floating-point RGB TIFF to begin')
        self.viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewer.setMinimumSize(350,350)
        split.addWidget(self.viewer)
        right=QScrollArea(); right.setWidgetResizable(True); split.addWidget(right)
        panel=QWidget(); right.setWidget(panel); controls=QVBoxLayout(panel)
        film_box=QGroupBox('Film simulation'); film_form=QFormLayout(film_box)
        self.film=QComboBox(); self.film.addItem('Kodak Portra 400','kodak_portra_400')
        self.paper=QComboBox(); self.paper.addItem('Kodak Portra Endura','kodak_portra_endura')
        film_form.addRow('Negative stock',self.film); film_form.addRow('Print paper',self.paper)
        controls.addWidget(film_box)
        exposure_box=QGroupBox('Scene-linear input'); form=QFormLayout(exposure_box)
        self.ev=QDoubleSpinBox(); self.ev.setRange(-8,8); self.ev.setSingleStep(0.1); self.ev.setSuffix(' EV')
        form.addRow('Exposure compensation',self.ev)
        auto=QPushButton('Suggest exposure (generic luminance percentile)')
        auto.clicked.connect(self.auto_exposure)
        form.addRow(auto)
        controls.addWidget(exposure_box)
        notice=QLabel('Only verified, connected controls are exposed.\nAdvanced film parameters, RAW development, masks,\nscopes and color-managed display require upstream\nsource-level integration and runtime validation.')
        notice.setWordWrap(True); controls.addWidget(notice); controls.addStretch()
        split.setSizes([240,920,280]); self.statusBar().showMessage('Ready — original spectral engine only, no LUT')
    def open_image(self):
        path,_=QFileDialog.getOpenFileName(self,'Open RAW or linear TIFF','','Images (*.dng *.cr2 *.cr3 *.nef *.nrw *.arw *.raf *.orf *.rw2 *.pef *.tif *.tiff)')
        if path: self.load_path(path)
    def load_selected(self,item): self.load_path(item.data(Qt.ItemDataRole.UserRole))
    def load_path(self,path):
        try:
            from .raw_loader import is_raw, decode_raw
            if is_raw(path):
                image=decode_raw(path)
                self.raw_camera_space=True
            else:
                import tifffile
                image=tifffile.imread(path)
                self.raw_camera_space=False
            if image.ndim!=3 or image.shape[-1]!=3 or not np.issubdtype(image.dtype,np.floating):
                raise ValueError('Expected H×W×3 floating-point scene-linear RGB TIFF')
            self.source=np.asarray(image,dtype=np.float32); self.result=None; self.last_path=path
            if not any(self.files.item(i).data(Qt.ItemDataRole.UserRole)==path for i in range(self.files.count())):
                from PySide6.QtWidgets import QListWidgetItem
                item=QListWidgetItem(Path(path).name); item.setData(Qt.ItemDataRole.UserRole,path); self.files.addItem(item)
            self.statusBar().showMessage('Loaded '+path+' — RAW is camera RGB; film input transform NOT validated' if self.raw_camera_space else 'Loaded '+path+' — confirm RGB primaries match upstream defaults')
        except Exception as exc: QMessageBox.critical(self,'Input failed',str(exc))
    def auto_exposure(self):
        if self.source is None: return
        from .scene_tools import suggest_exposure
        try:
            self.ev.setValue(suggest_exposure(self.source))
            self.statusBar().showMessage('Exposure suggested from linear RGB luminance; not darktable RAW deflicker')
        except Exception as exc:
            QMessageBox.warning(self,'Auto exposure unavailable',str(exc))
    def render(self):
        if self.source is None or self.thread is not None: return
        if getattr(self, "raw_camera_space", False):
            QMessageBox.warning(self, "RAW color contract", "RAW decoded successfully, but camera RGB has not been transformed into SpektraFilm’s required input color space. Film rendering is blocked to prevent incorrect colors.")
            return
        self.thread=QThread(self); self.worker=Worker(self.source.copy(),self.film.currentData(),self.paper.currentData(),self.ev.value())
        self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run)
        self.worker.success.connect(self.render_done); self.worker.error.connect(self.render_error)
        self.worker.success.connect(self.thread.quit); self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(self.cleanup); self.render_button.setEnabled(False); self.thread.start()
        self.statusBar().showMessage('Simulating with upstream SpektraFilm…')
    def render_done(self,result):
        if result.ndim!=3 or result.shape[-1]!=3:
            self.render_error('Unexpected engine output shape'); return
        self.result=result
        # Approximate sRGB preview only: requires validated output color-space contract.
        rgb=np.clip(result,0,1)
        encoded=np.where(rgb<=0.0031308,12.92*rgb,1.055*np.power(rgb,1/2.4)-0.055)
        pixels=np.ascontiguousarray((encoded*255+0.5).astype('uint8'))
        h,w,_=pixels.shape
        frame=QImage(pixels.data,w,h,3*w,QImage.Format.Format_RGB888).copy()
        self.viewer.setPixmap(QPixmap.fromImage(frame).scaled(self.viewer.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
        self.statusBar().showMessage('Rendered — display approximation, not ICC-managed')
    def render_error(self,message): QMessageBox.critical(self,'Render failed',message)
    def cleanup(self):
        self.worker.deleteLater(); self.thread.deleteLater(); self.worker=self.thread=None
        self.render_button.setEnabled(True)
    def export(self):
        if self.result is None: return
        path,_=QFileDialog.getSaveFileName(self,'Export scene-linear TIFF','film-output.tiff','TIFF (*.tiff)')
        if path:
            try:
                import tifffile
                tifffile.imwrite(path,np.asarray(self.result,dtype=np.float32),photometric='rgb')
                self.statusBar().showMessage('Exported '+path)
            except Exception as exc: QMessageBox.critical(self,'Export failed',str(exc))

def main():
    app=QApplication(sys.argv); window=Window(); window.show(); return app.exec()
if __name__=='__main__': raise SystemExit(main())
