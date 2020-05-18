#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Tuple, Callable, Union
import sys
import traceback

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from Backend import Loss
from Backend import Optimization
from Backend import Regression
from Frontend.Frontend import MainWindow
from Frontend.Frontend import ExceptionWindow


def Linear2DWithNoise(arange: int = 50) -> Tuple[np.matrix, np.matrix]:
    X = np.arange(-arange, arange, dtype=float)
    X = X.reshape(arange * 2, 1)

    Y = np.copy(X)
    Y += np.random.uniform(-arange / 10, arange / 10, (arange * 2, 1))

    Y = np.matrix(Y)
    X = np.matrix(X)

    return X, Y


def Fish2D() -> Tuple[np.matrix, np.matrix]:
    data = pd.read_csv("Demos/Fish.csv")
    X = np.array(data[["Length1"]])
    Y = np.array(data["Weight"]).reshape(159, 1)

    return X, Y


def Linear3DWithNoise(arange: int = 50) -> Tuple[np.matrix, np.matrix]:
    X0 = np.arange(-arange, arange, dtype=float)
    X0 = X0.reshape(arange * 2, 1)

    X1 = np.copy(X0)
    Y = np.copy(X0)

    X0 += np.random.uniform(-arange, arange, (int(arange) * 2, 1))
    X1 += np.random.randn(int(arange) * 2, 1)

    X = np.hstack((X0, X1))

    return (X, Y)


def Fish3D() -> Tuple[np.matrix, np.matrix]:
    data = pd.read_csv("Demos/Fish.csv")
    X = np.array(data[["Length1", "Height"]])
    Y = np.array(data["Weight"]).reshape(159, 1)

    return X, Y


class Main:
    _window: MainWindow
    _exceptionpopup: ExceptionWindow
    _modelDict: Dict[int, Regression.Regression]
    _optimizerDict: Dict[int, Optimization.Optimization]
    _lossDict: Dict[int, Loss.Loss]
    _datasetDict: Dict[int, Callable[..., Tuple[np.matrix, np.matrix]]]
    _last_history: Dict[str, List[Union[int, float]]]

    def __init__(self, argv: List[Any]) -> None:
        self._app = QtWidgets.QApplication(argv)
        self._window = MainWindow()
        self._exceptionpopup = ExceptionWindow()

        self._fields()
        self._events()

        self._create_dataset()
        self._create_loss()
        self._create_optim()
        self._create_model()

    def start(self) -> None:
        self._window.show()
        self._app.exec_()

    def _fields(self) -> None:
        self._modelDict = {
            0: Regression.LinearRegression,
            1: Regression.LinearRandom
        }

        self._optimizerDict = {
            0: Optimization.Gradient,
            1: Optimization.SGD,
            2: Optimization.MomentGrad,
            3: Optimization.Gradient,  # NesterovGrad
            4: Optimization.Gradient,  # MyGrad
        }

        self._lossDict = {
            0: Loss.LinearMSE
        }

        self._datasetDict = {
            0: Linear2DWithNoise,
            1: Fish2D,
            2: Linear3DWithNoise,
            3: Fish3D
        }

    def _events(self) -> None:
        self._window.comboDataset.currentIndexChanged.connect(
            self._create_dataset)
        self._window.comboModel.currentIndexChanged.connect(self._create_model)
        self._window.comboOptimizer.currentIndexChanged.connect(
            self._create_optim)
        self._window.comboLoss.currentIndexChanged.connect(self._create_loss)
        self._window.buttonFit.clicked.connect(self._fit)
        self._window.buttonLossHist.clicked.connect(self._draw_loss_history)

    def _plaintokw(self, text: str) -> Dict[str, str]:
        args = ' '.join(text.split('\n'))
        kwargs = dict(arg.split("=") for arg in args.split())
        return kwargs

    def _create_model(self) -> None:
        try:
            kwargs = self._window.plainModel.toPlainText()
            kwargs = self._plaintokw(kwargs)

            model = self._modelDict[self._window.Mmodel]
            self._model = model(**kwargs)

            self._window.checkHotStart.setCheckable(False)
        except Exception:
            msg = f"{traceback.format_exc()}"
            self._exceptionpopup.textException.setPlainText(msg)
            self._exceptionpopup.show()
        self._window.checkHotStart.setCheckable(False)
        
    def _create_optim(self) -> None:
        kwargs = self._window.plainOptimizer.toPlainText()
        kwargs = self._plaintokw(kwargs)

        optimizer = self._optimizerDict[self._window.Moptimizer]
        try:
            self._optimizer = optimizer(self._loss, **kwargs)
        except TypeError:
            self._create_loss()
            self._optimizer = optimizer(self._loss, **kwargs)

    def _create_loss(self) -> None:
        kwargs = self._window.plainLoss.toPlainText()
        kwargs = self._plaintokw(kwargs)
        loss = self._lossDict[self._window.Mloss]
        self._loss = loss(**kwargs)

    def _create_dataset(self) -> None:
        dataset = self._datasetDict[self._window.Ddataset]
        self.X, self.Y = dataset()

        if self.X.shape[1] == 1:
            X = np.array(self.X.reshape(self.X.shape[0]))
            Y = np.array(self.Y.reshape(self.Y.shape[0]))
            self._window.plot_2d_data(X, Y, msg='2D dataset')
        elif self.X.shape[1] == 2:
            Y = np.array(self.Y.reshape(self.Y.shape[0]))
            X = self.X
            self._window.plot_3d_data(X, Y, msg='3D dataset')
        self._window.checkHotStart.setCheckable(False)

    def _draw_loss_history(self) -> None:
        h = self._last_history
        losshist = self._window.LlossHistory
        if not losshist:
            losshist = len(h['i'])
        self._window.plot_loss(h['i'][-losshist:], h['loss'][-losshist:])

    def _fit(self) -> None:
        try:
            self._app.setOverrideCursor(Qt.WaitCursor)
            fit_kwargs = self._plaintokw(self._window.plainFit.toPlainText())
            fit_kwargs['lr'] = self._window.TlearningRate
            fit_kwargs['max_iter'] = self._window.TepochLimit
            fit_kwargs['loss_stop'] = self._window.TsatisLoss
            fit_kwargs['delta_loss_stop'] = self._window.TminDLoss
            fit_kwargs['hot_start'] = self._window.checkHotStart.isChecked()

            self._last_history = self._model.fit(
                self.X, self.Y, self._optimizer, **fit_kwargs)
            h = self._last_history
            if h['i']:
                self._window.checkHotStart.setCheckable(True)

                msg = f"Epoch: {h['i'][-1]+1},"
                msg += f" Loss: {h['loss'][-1]:.7},"
                msg += f" DLoss: {h['delta_loss'][-1]:.6},"
                msg += f" Min LR: {min(h['lr']):.5}"

                if self.X.shape[1] == 1:
                    X = np.array(self.X.reshape(self.X.shape[0]))
                    Y = np.array(self.Y.reshape(self.Y.shape[0]))
                    self._window.plot_2d_result(
                        h['weights'][-1], X, Y, msg=msg)
                elif self.X.shape[1] == 2:
                    Y = np.array(self.Y.reshape(self.Y.shape[0]))
                    X = self.X
                    self._window.plot_3d_result(
                        h['weights'][-1], X, Y, msg=msg)
                self._draw_loss_history()
        except Exception:
            msg = f"{traceback.format_exc()}"
            self._exceptionpopup.popup(msg)
        finally:
            self._app.restoreOverrideCursor()


if __name__ == "__main__":
    main = Main(sys.argv)
    main.start()
