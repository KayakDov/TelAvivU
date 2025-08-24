import sys
import subprocess
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit

class Launcher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LaplaceFiniteDifference GUI")
        self.resize(400, 200)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Input file
        self.input_label = QLabel("Input file:")
        self.input_line = QLineEdit()
        self.input_button = QPushButton("Browse")
        self.input_button.clicked.connect(self.browse_input)

        self.layout.addWidget(self.input_label)
        self.layout.addWidget(self.input_line)
        self.layout.addWidget(self.input_button)

        # Output file
        self.output_label = QLabel("Output file:")
        self.output_line = QLineEdit()
        self.output_button = QPushButton("Browse")
        self.output_button.clicked.connect(self.browse_output)

        self.layout.addWidget(self.output_label)
        self.layout.addWidget(self.output_line)
        self.layout.addWidget(self.output_button)

        # Run button
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_executable)
        self.layout.addWidget(self.run_button)

        # Status
        self.status_label = QLabel("")
        self.layout.addWidget(self.status_label)

    def browse_input(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Input File")
        if file_name:
            self.input_line.setText(file_name)

    def browse_output(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Select Output File")
        if file_name:
            self.output_line.setText(file_name)

    def run_executable(self):
        exe_path = "./testRun"  # Or full path to your compiled executable
        input_file = self.input_line.text()
        output_file = self.output_line.text()
        
        if not input_file or not output_file:
            self.status_label.setText("Please select both input and output files")
            return

        try:
            subprocess.run([exe_path, input_file, output_file], check=True)
            self.status_label.setText("Execution finished successfully!")
        except subprocess.CalledProcessError as e:
            self.status_label.setText(f"Execution failed: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Launcher()
    window.show()
    sys.exit(app.exec())

