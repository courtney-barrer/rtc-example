import os
import subprocess

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap  # Add these imports
# run script test to generate the config files (baldr_config.json")
from script_test import *


## THIS IS SET UP WITH FAKE CONFIG - NEED TO CHANGE TO REAL CONFIG
# WHERE WE HAVE commands_dict[beam] for the DM commands. Currently 
# copying the same commands to each beam.

def get_DM_command_in_2D(cmd, Nx_act=12):
    # function so we can easily plot the DM shape (since DM grid is not perfectly square raw cmds can not be plotted in 2D immediately )
    #puts nan values in cmd positions that don't correspond to actuator on a square grid until cmd length is square number (12x12 for BMC multi-2.5 DM) so can be reshaped to 2D array to see what the command looks like on the DM.
    corner_indices = [0, Nx_act-1, Nx_act * (Nx_act-1), Nx_act*Nx_act]
    cmd_in_2D = list(cmd.copy())
    for i in corner_indices:
        cmd_in_2D.insert(i,np.nan)
    return( np.array(cmd_in_2D).reshape(Nx_act, Nx_act) )


# config everything 
command = ["build/Release/baldr_main", "--config", "baldr_config.json"]
process = subprocess.Popen(command,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Process started with PID:", process.pid)

# start the camera writing to SHM 
cam_command.run()  # 
dm_command.run()  # commands_lock.unlock() # unlock the commands_lock so dm recieves rtc output for one iteration 
rtc_command.run() #

# requirements: 

# A gui showing four images aligned in columns across the screen 
# with some user input to change modes/settings

# the global settings should be above the images (spanning all the columns)
# and contain the following : 
    
#     double det_dit = 0.0016; // detector integration time (s)
#     double det_fps = 600.0; // frames per second (Hz)
#     std::string det_gain ="medium"; // "low" or "medium" or "high"
#     int frame_rate = 30; // frame rate (Hz)
# local settings should be below the images in each column and contain the following :
#     zoom = 1.0; // zoom factor
#     int row1 = 0; // top row of the subregion
#     int row2 = 128; // bottom row of the subregion
#     int col1 = 0; // left column of the subregion
#     int col2 = 128; // right column of the subregion

# a new frame can be called using the following command :
#     frame
# the four images correspond to subregions of the frame that we want to display as images
# organised in columns across the screen. These are defined by local variables row1, row2, col1, col2
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Display and Settings")
        self.setGeometry(100, 100, 1200, 800)  # Set window size
        
        # Initialize lists to store image labels and settings
        self.image_labels = []
        self.dm_labels = []
        self.local_settings = []

        # Main layout
        main_layout = QVBoxLayout()
        
        # Global settings
        global_settings = self.create_global_settings()
        main_layout.addWidget(global_settings)

        # Start/Stop button for frame capturing
        self.start_button = QPushButton("Start Frame Capture")
        self.start_button.setCheckable(True)
        self.start_button.clicked.connect(self.toggle_frame_capture)
        main_layout.addWidget(self.start_button)
        
        # Timer for continuous frame capture
        self.timer = QTimer()
        self.timer.timeout.connect(self.get_and_display_frame)

        # Image display area with local settings for each image
        image_layout = QHBoxLayout()
        
        # Create columns for DM command images, frame images, and local settings
        for i in range(4):
            column_layout, dm_label, image_label, settings_dict = self.create_image_column(i + 1)
            self.dm_labels.append(dm_label)
            self.image_labels.append(image_label)
            self.local_settings.append(settings_dict)
            image_layout.addLayout(column_layout)
        
        main_layout.addLayout(image_layout)
        
        # Set main layout to the window
        self.setLayout(main_layout)
    
    def create_global_settings(self):
        """Create the global settings panel."""
        group_box = QGroupBox("Global Settings")
        layout = QFormLayout()

        # Detector integration time
        self.det_dit = QDoubleSpinBox()
        self.det_dit.setValue(0.0016)
        self.det_dit.setSingleStep(0.0001)
        layout.addRow("Detector Integration Time (s):", self.det_dit)
        
        # Frames per second
        self.det_fps = QDoubleSpinBox()
        self.det_fps.setValue(600.0)
        self.det_fps.setSingleStep(10)
        layout.addRow("Frames per Second (Hz):", self.det_fps)

        # Detector gain
        self.det_gain = QComboBox()
        self.det_gain.addItems(["low", "medium", "high"])
        self.det_gain.setCurrentText("medium")
        layout.addRow("Detector Gain:", self.det_gain)

        # Frame rate
        self.frame_rate = QSpinBox()
        self.frame_rate.setValue(30)
        layout.addRow("Frame Rate (Hz):", self.frame_rate)
        
        group_box.setLayout(layout)
        return group_box

    def create_image_column(self, index):
        """Create a column with a DM command image, frame display, and local settings."""
        column_layout = QVBoxLayout()
        
        # DM command display placeholder
        dm_label = QLabel(f"DM Command {index}")
        dm_label.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        dm_label.setFixedSize(200, 200)  # Placeholder size
        column_layout.addWidget(dm_label)

        # Image display placeholder for frames
        image_label = QLabel(f"Image {index}")
        image_label.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        image_label.setFixedSize(200, 200)  # Placeholder size
        column_layout.addWidget(image_label)

        # Local settings for the image
        local_settings_groupbox, settings_dict = self.create_local_settings()
        column_layout.addWidget(local_settings_groupbox)
        
        return column_layout, dm_label, image_label, settings_dict  # Return layout, DM label, image label, and settings dict
    
    def create_local_settings(self):
        """Create local settings panel for each image."""
        settings = {}
        group_box = QGroupBox("Local Settings")
        layout = QFormLayout()
        
        # Zoom factor
        zoom = QDoubleSpinBox()
        zoom.setValue(1.0)
        zoom.setSingleStep(0.1)
        settings["zoom"] = zoom
        layout.addRow("Zoom:", zoom)
        
        # Row and column boundaries
        row1 = QSpinBox()
        row1.setValue(0)
        settings["row1"] = row1
        layout.addRow("Top Row (row1):", row1)
        
        row2 = QSpinBox()
        row2.setValue(128)
        settings["row2"] = row2
        layout.addRow("Bottom Row (row2):", row2)
        
        col1 = QSpinBox()
        col1.setValue(0)
        settings["col1"] = col1
        layout.addRow("Left Column (col1):", col1)
        
        col2 = QSpinBox()
        col2.setValue(128)
        settings["col2"] = col2
        layout.addRow("Right Column (col2):", col2)
        
        group_box.setLayout(layout)
        return group_box, settings

    def toggle_frame_capture(self):
        """Start or stop the continuous frame capture."""
        if self.start_button.isChecked():
            self.start_button.setText("Stop Frame Capture")
            interval = int(1000 / self.frame_rate.value())  # Set interval based on frame rate
            self.timer.start(interval)
        else:
            self.start_button.setText("Start Frame Capture")
            self.timer.stop()

    def get_and_display_frame(self):
        """Retrieve and display the frame in specified subregions and update DM command images."""
        # Simulate fetching a new frame (replace with actual frame fetching)
        frame = np.random.randint(0, 256, (256, 256), dtype=np.uint8)  # Example: random image

        # Retrieve DM commands and update each DM command image
        dm_commands = [np.random.rand(140) for _ in range(4)]  # Simulated DM commands
        for i, command in enumerate(dm_commands):
            command_2D = get_DM_command_in_2D(command)  # Convert command to 2D
            
            # Convert command_2D to QImage for display
            height, width = command_2D.shape
            image = np.stack((command_2D * 255,) * 3, axis=-1).astype(np.uint8)  # Scale and convert to RGB
            qimage = QImage(image.data, width, height, 3 * width, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage).scaled(200, 200)  # Resize for display
            self.dm_labels[i].setPixmap(pixmap)

        # Update each image label based on subregion settings
        for i, (label, settings) in enumerate(zip(self.image_labels, self.local_settings)):
            row1 = settings["row1"].value()
            row2 = settings["row2"].value()
            col1 = settings["col1"].value()
            col2 = settings["col2"].value()

            # Extract subregion from frame
            subregion = frame[row1:row2, col1:col2]
            
            # Convert to a QImage and display (requires conversion from NumPy array)
            height, width = subregion.shape
            image = np.stack((subregion,) * 3, axis=-1)  # Convert grayscale to RGB
            qimage = QImage(image.data, width, height, 3 * width, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage).scaled(200, 200)  # Resize for display
            label.setPixmap(pixmap)

# Run the application
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())

