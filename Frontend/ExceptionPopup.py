# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './ExceptionPopup.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtWidgets


class Ui_ExceptionPopup(object):
    def setupUi(self, ExceptionPopup):
        ExceptionPopup.setObjectName("ExceptionPopup")
        ExceptionPopup.resize(818, 300)
        self.buttonOk = QtWidgets.QPushButton(ExceptionPopup)
        self.buttonOk.setGeometry(QtCore.QRect(330, 270, 105, 30))
        self.buttonOk.setObjectName("buttonOk")
        self.textException = QtWidgets.QTextEdit(ExceptionPopup)
        self.textException.setGeometry(QtCore.QRect(0, 10, 821, 251))
        self.textException.setReadOnly(True)
        self.textException.setObjectName("textException")

        self.retranslateUi(ExceptionPopup)
        QtCore.QMetaObject.connectSlotsByName(ExceptionPopup)

    def retranslateUi(self, ExceptionPopup):
        _translate = QtCore.QCoreApplication.translate
        ExceptionPopup.setWindowTitle(_translate(
            "ExceptionPopup", "Unhandled Exception"))
        self.buttonOk.setText(_translate("ExceptionPopup", "OK"))
