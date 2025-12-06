import sys
import os
import csv
from datetime import datetime

# PyQt5 - The tools for the window
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QComboBox, QPushButton,
                             QLabel, QGroupBox, QMessageBox, QSpinBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# Matplotlib - The tool for the graph
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Import our Logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import LoanPredictor


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Basic Window Setup
        self.setWindowTitle("Loan Approval Decision Support System")
        self.setGeometry(100, 100, 1100, 700)

        # Load Styling (CSS)
        with open(os.path.join(os.path.dirname(__file__), 'styles.qss'), 'r') as f:
            self.setStyleSheet(f.read())

        # Load the Brain
        try:
            self.predictor = LoanPredictor()
        except:
            QMessageBox.critical(self, "Error", "Model not found! Run src/train_model.py first.")
            sys.exit()

        self.setup_ui()

    def setup_ui(self):
        # The Main Container
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)  # Horizontal Layout (Left | Right)

        # --- LEFT PANEL (Inputs) ---
        left_box = QGroupBox("Applicant Details")
        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(15)  # Add space between rows

        self.inputs = {}

        # Dropdowns (Combo Boxes)
        self.inputs['Gender'] = self.make_combo(['Male', 'Female'])
        self.inputs['Married'] = self.make_combo(['Yes', 'No'])
        self.inputs['Dependents'] = self.make_combo(['0', '1', '2', '3+'])
        self.inputs['Education'] = self.make_combo(['Graduate', 'Not Graduate'])
        self.inputs['Self_Employed'] = self.make_combo(['No', 'Yes'])
        self.inputs['Property_Area'] = self.make_combo(['Urban', 'Semiurban', 'Rural'])

        # Number Inputs (Spin Boxes)
        self.inputs['ApplicantIncome'] = self.make_spin(5000)
        self.inputs['CoapplicantIncome'] = self.make_spin(0)
        self.inputs['LoanAmount'] = self.make_spin(120)

        # Loan Term and Credit History
        self.inputs['Loan_Amount_Term'] = self.make_combo(['360', '180', '480', '120'])
        self.inputs['Credit_History'] = self.make_combo(['1.0', '0.0'])

        # Add everything to the form
        form_layout.addRow("Gender:", self.inputs['Gender'])
        form_layout.addRow("Married:", self.inputs['Married'])
        form_layout.addRow("Dependents:", self.inputs['Dependents'])
        form_layout.addRow("Education:", self.inputs['Education'])
        form_layout.addRow("Self Employed:", self.inputs['Self_Employed'])
        form_layout.addRow("Applicant Income:", self.inputs['ApplicantIncome'])
        form_layout.addRow("Co-Applicant Income:", self.inputs['CoapplicantIncome'])
        form_layout.addRow("Loan Amount (k):", self.inputs['LoanAmount'])
        form_layout.addRow("Loan Term (Days):", self.inputs['Loan_Amount_Term'])
        form_layout.addRow("Credit History:", self.inputs['Credit_History'])
        form_layout.addRow("Property Area:", self.inputs['Property_Area'])

        # The Big Blue Button
        self.btn_predict = QPushButton("ANALYZE RISK")
        self.btn_predict.setObjectName("PredictBtn")
        self.btn_predict.clicked.connect(self.run_analysis)
        form_layout.addRow(self.btn_predict)

        left_box.setLayout(form_layout)

        # --- RIGHT PANEL (Results) ---
        right_layout = QVBoxLayout()

        # Result Text
        self.result_box = QGroupBox("Prediction")
        res_layout = QVBoxLayout()
        self.lbl_status = QLabel("Enter details to predict")
        self.lbl_status.setObjectName("ResultLabel")
        self.lbl_risk = QLabel("")
        self.lbl_risk.setObjectName("RiskLabel")
        res_layout.addWidget(self.lbl_status)
        res_layout.addWidget(self.lbl_risk)
        self.result_box.setLayout(res_layout)

        # Graph
        self.graph_box = QGroupBox("Key Factors")
        graph_layout = QVBoxLayout()
        self.figure = plt.figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        graph_layout.addWidget(self.canvas)
        self.graph_box.setLayout(graph_layout)

        right_layout.addWidget(self.result_box)
        right_layout.addWidget(self.graph_box)

        # Add panels to main layout
        layout.addWidget(left_box, 1)  # Left takes 1 part width
        layout.addLayout(right_layout, 2)  # Right takes 2 parts width

    # Helper to create Dropdowns
    def make_combo(self, items):
        cb = QComboBox()
        cb.addItems(items)
        return cb

    # Helper to create Number inputs
    def make_spin(self, val):
        sb = QSpinBox()
        sb.setRange(0, 999999)
        sb.setValue(val)
        return sb

    # --- THE MAIN LOGIC ---
    def run_analysis(self):
        # 1. Collect Data from inputs
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

        # 2. Get Prediction from src/predict.py
        try:
            res = self.predictor.predict(data)

            # 3. Update UI
            self.lbl_status.setText(res['status'])

            # Change Colors
            if res['status'] == "Approved":
                self.lbl_status.setStyleSheet("color: #27ae60;")  # Green
            else:
                self.lbl_status.setStyleSheet("color: #c0392b;")  # Red

            self.lbl_risk.setText(f"Risk: {res['risk']} | Confidence: {res['confidence']}")

            # 4. Update Graph
            self.update_graph(res['top_features'])

            # 5. Save Log
            self.save_log(data, res)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def update_graph(self, features):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        names = list(features.keys())
        values = list(features.values())

        # Draw Horizontal Bar Chart
        ax.barh(names, values, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax.set_title("Top Decision Factors")
        self.canvas.draw()

    def save_log(self, data, res):
        file_exists = os.path.isfile('history.csv')
        with open('history.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(list(data.keys()) + ['Status', 'Date'])
            writer.writerow(list(data.values()) + [res['status'], datetime.now()])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))  # Set a nice font
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())