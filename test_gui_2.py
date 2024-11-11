from baldr import _baldr as ba
from baldr import sardine as sa
import sys
import json
import numpy as np
import subprocess
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg


class AOControlApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Main layout as a 2x2 grid
        main_layout = QtWidgets.QGridLayout(self)

        # Camera frame placeholder (G[0,0])
        self.camera_view = pg.ImageView()
        main_layout.addWidget(self.camera_view, 0, 0)  # Top-left corner, large

        # DM images in a vertical column (G[0,1])
        dm_layout = QtWidgets.QVBoxLayout()
        self.dm_images = []
        for i in range(4):
            dm_label = QtWidgets.QLabel(f"DM {i + 1}")
            dm_image = pg.ImageView()
            dm_image.setImage(np.random.rand(12, 12) * 255)  # Placeholder 12x12 image for DMs
            dm_layout.addWidget(dm_label)
            dm_layout.addWidget(dm_image)
            self.dm_images.append(dm_image)

        main_layout.addLayout(dm_layout, 0, 1)  # Top-right corner, tall

        # Command line prompt area (G[1,0])
        command_layout = QtWidgets.QVBoxLayout()
        self.prompt_history = QtWidgets.QTextEdit()
        self.prompt_history.setReadOnly(True)
        self.prompt_input = QtWidgets.QLineEdit()
        self.prompt_input.returnPressed.connect(self.handle_command)
        command_layout.addWidget(self.prompt_history)
        command_layout.addWidget(self.prompt_input)

        main_layout.addLayout(command_layout, 1, 0)  # Bottom-left, wide

        # 4x2 button grid (G[1,1])
        button_layout = QtWidgets.QGridLayout()
        self.buttons = []
        for i in range(8):
            button = QtWidgets.QPushButton(f"Button {i + 1}")
            self.buttons.append(button)
            button_layout.addWidget(button, i // 2, i % 2)  # Arrange in 4 rows, 2 columns

        main_layout.addLayout(button_layout, 1, 1)  # Bottom-right, compact

        # Adjust row and column stretch to set proportions
        main_layout.setRowStretch(0, 3)
        main_layout.setRowStretch(1, 1)
        main_layout.setColumnStretch(0, 3)
        main_layout.setColumnStretch(1, 1)

        # Start the external process
        self.start_external_process()

        # Set up shared memory frame
        self.setup_shared_memory()

        # Timer to update the frame at a given frame rate (e.g., 30 FPS)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_camera_frame)
        self.timer.start(1000 // 30)  # 30 FPS

    def start_external_process(self):
        """Start the baldr_main process using subprocess."""
        command = ["build/Release/baldr_main", "--config", "baldr_config.json"]
        self.process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Process started with PID:", self.process.pid)

    def setup_shared_memory(self):
        """Load configuration file and set up shared memory frame."""
        with open("baldr_config.json", "r") as f:
            config = json.load(f)
        frame_url = config[0]["io"]["frame"]

        # Access shared memory frame using sardine
        self.frame = sa.from_url(np.ndarray, frame_url)
        print(f"Shared memory frame accessed at: {frame_url}")

    def update_camera_frame(self):
        """Update the main camera image from shared memory."""
        try:
            img_data = np.array(self.frame, copy=False)  # Access shared memory directly
            self.camera_view.setImage(img_data, autoLevels=True)  # Update camera view
        except Exception as e:
            print(f"Failed to update camera frame: {e}")

    def handle_command(self):
        """Process commands from the command line input."""
        command = self.prompt_input.text()
        self.command_history.append(command)
        self.history_index = len(self.command_history)
        self.prompt_history.append(f"> {command}")

        try:
            exec(command, globals())
            self.prompt_history.append("Command executed.")
        except Exception as e:
            self.prompt_history.append(f"Error: {str(e)}")
        
        self.prompt_input.clear()

    def closeEvent(self, event):
        """Ensure the subprocess is terminated when the GUI is closed."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("External process terminated.")
        event.accept()

    def keyPressEvent(self, event):
        """Handle up/down arrows for command history."""
        if event.key() == QtCore.Qt.Key_Up:
            if self.history_index > 0:
                self.history_index -= 1
                self.prompt_input.setText(self.command_history[self.history_index])
        elif event.key() == QtCore.Qt.Key_Down:
            if self.history_index < len(self.command_history) - 1:
                self.history_index += 1
                self.prompt_input.setText(self.command_history[self.history_index])
            else:
                self.history_index = len(self.command_history)
                self.prompt_input.clear()

    def __del__(self):
        print("killing all")
        cam_command.exit()
        frame_lock.unlock()
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = AOControlApp()
    window.setWindowTitle("AO Control GUI")
    window.resize(1000, 700)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
