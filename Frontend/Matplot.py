# -*- coding: utf-8 -*-
from typing import Any, List

from PyQt5.QtWidgets import QSizePolicy
import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


class PlotCanvas(FigureCanvas):
    def __init__(self,
                 parent: Any = None,
                 width: float = 6.5,
                 height: float = 6.5,
                 dpi: int = 100) -> None:
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        self.ax = self.figure.add_subplot()

        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def scatter2d(self, X: np.ndarray, Y: np.ndarray, msg: str = '') -> None:
        self.figure.clf()
        self.ax = self.figure.add_subplot()
        self.ax.scatter(X, Y)
        self.ax.set_title(msg)
        self.draw()

    def plot2d(self, W: np.matrix, minX: float, maxX: float) -> None:
        xs = np.arange(minX, maxX)
        ys = [np.sum(W[0]) * x + np.sum(W[1]) for x in xs]
        self.ax.plot(xs, ys, 'r')
        self.draw()

    def plot2d_dots(self, x: List[float], y: List[float]) -> None:
        self.figure.clf()
        self.ax = self.figure.add_subplot()
        self.ax.plot(x, y, 'g')
        self.draw()

    def scatter3d(self, X: np.ndarray, Y: np.ndarray, msg: str = '') -> None:
        self.figure.clf()
        self.ax = self.figure.add_subplot(projection='3d')
        X1 = X[:, 0]
        X2 = X[:, 1]
        self.ax.scatter(X1, X2, Y)
        self.ax.set_title(msg)
        self.draw()

    def plot3d(self,
               W: np.matrix,
               minX1: float,
               maxX1: float,
               minX2: float,
               maxX2: float) -> None:
        A, B, C = W
        A, B, C = np.sum(A), np.sum(B), np.sum(C)
        x, y = np.meshgrid(np.arange(minX1, maxX1), np.arange(minX2, maxX2))
        z = A * x + B * y + C

        self.ax.plot_surface(x, y, z, color='r')
        self.draw()
