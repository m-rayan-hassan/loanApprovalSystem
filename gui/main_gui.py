# gui/main_gui.py
import sys
import os
import csv
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QLineEdit, QComboBox,
                             QPushButton, QLabel, QGroupBox, QMessageBox, QSpinBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon

# Matplotlib for PyQt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Import Predictor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import LoanPredictor


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Loan Approval Decision Support System")
        self.setGeometry(100, 100, 1100, 700)

        # Load Styles
        with open(os.path.join(os.path.dirname(__file__), 'styles.qss'), 'r') as f:
            self.setStyleSheet(f.read())

        try:
            self.predictor = LoanPredictor()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Model not found! Run src/train_model.py first.\n\nError: {e}")
            sys.exit()

        self.initUI()

    def initUI(self):
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- LEFT PANEL: INPUT FORM ---
        left_panel = QGroupBox("Applicant Details")
        left_layout = QFormLayout()
        left_layout.setVerticalSpacing(15)

        # Inputs
        self.inputs = {}

        # Categorical Fields
        self.inputs['Gender'] = self.create_combo(['Male', 'Female'])
        self.inputs['Married'] = self.create_combo(['Yes', 'No'])
        self.inputs['Dependents'] = self.create_combo(['0', '1', '2', '3+'])
        self.inputs['Education'] = self.create_combo(['Graduate', 'Not Graduate'])
        self.inputs['Self_Employed'] = self.create_combo(['No', 'Yes'])
        self.inputs['Property_Area'] = self.create_combo(['Urban', 'Semiurban', 'Rural'])

        # Numeric Fields
        self.inputs['ApplicantIncome'] = QSpinBox()
        self.inputs['ApplicantIncome'].setRange(0, 1000000)
        self.inputs['ApplicantIncome'].setValue(5000)
        self.inputs['CoapplicantIncome'] = QSpinBox()
        self.inputs['CoapplicantIncome'].setRange(0, 1000000)
        self.inputs['CoapplicantIncome'].setValue(0)
        self.inputs['LoanAmount'] = QSpinBox()
        self.inputs['LoanAmount'].setRange(0, 1000000)
        self.inputs['LoanAmount'].setValue(120)
        self.inputs['Loan_Amount_Term'] = QComboBox()
        self.inputs['Loan_Amount_Term'].addItems(['360', '180', '480', '120', '240'])
        self.inputs['Credit_History'] = QComboBox()
        self.inputs['Credit_History'].addItems(['1.0', '0.0'])  # 1.0=Good, 0.0=Bad

        # Add to Layout
        left_layout.addRow("Gender:", self.inputs['Gender'])
        left_layout.addRow("Marital Status:", self.inputs['Married'])
        left_layout.addRow("Dependents:", self.inputs['Dependents'])
        left_layout.addRow("Education:", self.inputs['Education'])
        left_layout.addRow("Self Employed:", self.inputs['Self_Employed'])
        left_layout.addRow("Applicant Income ($):", self.inputs['ApplicantIncome'])
        left_layout.addRow("Co-Applicant Income ($):", self.inputs['CoapplicantIncome'])
        left_layout.addRow("Loan Amount (k$):", self.inputs['LoanAmount'])
        left_layout.addRow("Loan Term (Days):", self.inputs['Loan_Amount_Term'])
        left_layout.addRow("Credit History:", self.inputs['Credit_History'])
        left_layout.addRow("Property Area:", self.inputs['Property_Area'])

        # Predict Button
        self.btn_predict = QPushButton("ANALYZE LOAN RISK")
        self.btn_predict.setObjectName("PredictBtn")
        self.btn_predict.setCursor(Qt.PointingHandCursor)
        self.btn_predict.clicked.connect(self.run_prediction)

        left_layout.addRow(self.btn_predict)
        left_panel.setLayout(left_layout)

        # --- RIGHT PANEL: OUTPUT & VISUALS ---
        right_panel = QVBoxLayout()

        # 1. Result Box
        self.result_group = QGroupBox("Prediction Results")
        result_layout = QVBoxLayout()

        self.lbl_status = QLabel("Ready to Predict")
        self.lbl_status.setObjectName("ResultLabel")
        self.lbl_status.setStyleSheet("color: #7f8c8d;")

        self.lbl_risk = QLabel("")
        self.lbl_risk.setObjectName("RiskLabel")

        self.lbl_details = QLabel("")
        self.lbl_details.setAlignment(Qt.AlignCenter)

        result_layout.addWidget(self.lbl_status)
        result_layout.addWidget(self.lbl_risk)
        result_layout.addWidget(self.lbl_details)
        self.result_group.setLayout(result_layout)

        # 2. Graph Box
        self.graph_group = QGroupBox("Feature Importance")
        graph_layout = QVBoxLayout()
        self.figure = plt.figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        graph_layout.addWidget(self.canvas)
        self.graph_group.setLayout(graph_layout)

        right_panel.addWidget(self.result_group, 1)
        right_panel.addWidget(self.graph_group, 2)

        # Final Assembly
        main_layout.addWidget(left_panel, 1)
        main_layout.addLayout(right_panel, 2)

    def create_combo(self, items):
        cb = QComboBox()
        cb.addItems(items)
        return cb

    def run_prediction(self):
        # 1. Gather Data
        data = {
            'Gender': self.inputs['Gender'].currentText(),
            'Married': self.inputs['Married'].currentText(),
            'Dependents': self.inputs['Dependents'].currentText(),
            'Education': self.inputs['Education'].currentText(),
            'Self_Employed': self.inputs['Self_Employed'].currentText(),
            'ApplicantIncome': self.inputs['ApplicantIncome'].value(),
            'CoapplicantIncome': self.inputs['CoapplicantIncome'].value(),
            'LoanAmount': self.inputs['LoanAmount'].value(),
            'Loan_Amount_Term': float(self.inputs['Loan_Amount_Term'].currentText()),
            'Credit_History': float(self.inputs['Credit_History'].currentText()),
            'Property_Area': self.inputs['Property_Area'].currentText()
        }

        # 2. Get Prediction
        try:
            res = self.predictor.predict(data)

            # 3. Update UI
            # Status
            self.lbl_status.setText(res['status'])
            if res['status'] == "Approved":
                self.lbl_status.setStyleSheet("color: #27ae60; font-size: 26px; font-weight: bold;")
            else:
                self.lbl_status.setStyleSheet("color: #c0392b; font-size: 26px; font-weight: bold;")

            # Risk
            color_map = {"Low": "#27ae60", "Medium": "#f39c12", "High": "#c0392b"}
            self.lbl_risk.setText(f"Risk Level: {res['risk']} ({res['risk_score']})")
            self.lbl_risk.setStyleSheet(
                f"color: {color_map.get(res['risk'], 'black')}; font-size: 16px; font-weight: bold;")

            self.lbl_details.setText(f"Confidence: {res['confidence']}")

            # 4. Update Graph
            self.update_graph(res['top_features'])

            # 5. Log to CSV
            self.log_prediction(data, res)

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", str(e))

    def update_graph(self, features):
        self.figure.clear()
        if not features:
            return

        ax = self.figure.add_subplot(111)
        names = list(features.keys())
        values = list(features.values())

        # Modern Bar Chart
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        bars = ax.barh(names, values, color=colors[:len(names)])
        ax.set_title("Key Factors Influencing Decision")
        ax.set_xlabel("Importance")
        ax.invert_yaxis()  # Top factor on top

        self.figure.tight_layout()
        self.canvas.draw()

    def log_prediction(self, inputs, results):
        file_exists = os.path.isfile('history.csv')
        with open('history.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                headers = list(inputs.keys()) + ['Status', 'Risk', 'Date']
                writer.writerow(headers)

            row = list(inputs.values()) + [results['status'], results['risk'],
                                           datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            writer.writerow(row)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Global Font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())