# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Tuple, Callable

import numpy as np

from PyQt5 import QtWidgets
import Frontend.MainForm as MainForm
import Frontend.ExceptionPopup as ExceptionPopup
import Frontend.Matplot as Matplot


class MainWindow(QtWidgets.QMainWindow, MainForm.Ui_MainForm):
    def __init__(self) -> None:
        super().__init__()
        self.setupUi()

        self._fields()
        self._events()

    def setupUi(self) -> None:
        super().setupUi(self)
        self.tabWidget.setCurrentIndex(0)

        self.plotResult = Matplot.PlotCanvas(self.widgetResults)
        self.plotData = Matplot.PlotCanvas(
            self.widgetDataset, width=4.2, height=4)
        self.plotLoss = Matplot.PlotCanvas(
            self.widgetLoss, width=2.6, height=2.2)

    def plot_2d_result(self,
                       W: List[float],
                       X: np.ndarray,
                       Y: np.ndarray,
                       msg: str = '') -> None:
        self.plotResult.scatter2d(X, Y, msg=msg)
        self.plotResult.plot2d(W, np.min(X), int(np.max(X)))

    def plot_3d_result(self,
                       W: List[float],
                       X: np.ndarray,
                       Y: np.ndarray,
                       msg: str = '') -> None:
        self.plotResult.scatter3d(X, Y, msg=msg)
        minX1, maxX1 = np.min(X[:, 0]), np.max(X[:, 0])
        minX2, maxX2 = np.min(X[:, 1]), np.max(X[:, 1])
        self.plotResult.plot3d(W, minX1, maxX1, minX2, maxX2)

    def plot_2d_data(self, X: np.ndarray, Y: np.ndarray, msg: str = '') -> None:
        self.plotData.scatter2d(X, Y, msg=msg)

    def plot_3d_data(self, X: np.ndarray, Y: np.ndarray, msg: str = '') -> None:
        self.plotData.scatter3d(X, Y, msg=msg)

    def plot_loss(self, iters: List[int], losses: List[float]) -> None:
        self.plotLoss.plot2d_dots(iters, losses)

    def _events(self) -> None:
        # Data
        self.comboDataset.currentIndexChanged.connect(self._DsetDataset)
        # Model
        self.comboModel.currentIndexChanged.connect(self._MsetModel)
        self.comboOptimizer.currentIndexChanged.connect(self._MsetOptimizer)
        self.comboLoss.currentIndexChanged.connect(self._MsetLoss)
        # Training
        self.spinEpochLimit.valueChanged.connect(self._TsetEpochLimit)
        self.spinLearningRate.valueChanged.connect(self._TsetLearningRate)
        self.spinSatisLoss.valueChanged.connect(self._TsetSatisLoss)
        self.spinMinDLoss.valueChanged.connect(self._TsetMinDLoss)
        # Loss
        self.spinLossHistory.valueChanged.connect(self._LsetLossHistory)

    def _fields(self) -> None:
        self.pcf = 1000

        self.TepochLimit = int(self.spinEpochLimit.value())
        self.TlearningRate = float(self.spinLearningRate.value())
        self.TsatisLoss = float(self.spinSatisLoss.value())
        self.TminDLoss = float(self.spinMinDLoss.value())

        self.checkHotStart.setCheckable(False)

        self.LlossHistory = 0

        self.Ddataset = 0

        self.Mmodel = 0
        self.Moptimizer = 0
        self.Mloss = 0

        self.TepochCurrent = 0
        self.TlossCurrent = 0
        self.TdlossCurrent = 0

    def _DsetDataset(self) -> None:
        self.Ddataset = self.comboDataset.currentIndex()

    def _MsetModel(self) -> None:
        self.Mmodel = self.comboModel.currentIndex()

    def _MsetOptimizer(self) -> None:
        self.Moptimizer = self.comboOptimizer.currentIndex()

    def _MsetLoss(self) -> None:
        self.Mloss = self.comboLoss.currentIndex()

    def _TsetEpochLimit(self) -> None:
        self.TepochLimit = int(self.spinEpochLimit.value())

    def _TsetLearningRate(self) -> None:
        self.TlearningRate = float(self.spinLearningRate.value())

    def _TsetSatisLoss(self) -> None:
        self.TsatisLoss = float(self.spinSatisLoss.value())

    def _TsetMinDLoss(self) -> None:
        self.TminDLoss = float(self.spinMinDLoss.value())

    def _LsetLossHistory(self) -> None:
        self.LlossHistory = int(self.spinLossHistory.value())


class ExceptionWindow(QtWidgets.QMainWindow, ExceptionPopup.Ui_ExceptionPopup):
    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)

        self._events()

    def popup(self, msg: str) -> None:
        self.textException.setPlainText(msg)
        self.show()

    def _events(self) -> None:
        self.buttonOk.clicked.connect(self.close)
