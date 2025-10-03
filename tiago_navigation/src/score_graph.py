#!/usr/bin/env python
import rospy
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import Float32MultiArray
import signal
import sys
import threading

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow
import pyqtgraph

class ScoreSubscriber:
    def __init__(self, data):
        self.data = data
        self.sub = rospy.Subscriber(
            '/spatial_attention_scores',
            Float32MultiArray,
            self.data_callback
        )

    def data_callback(self, msg):
        self.data.receive_data(msg)

class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle('Attention Scores')
        self.setGeometry(50, 50, 600, 350)

        self.data = np.zeros((1, 10))  # initial dummy data (1 row for heatmap, 10 scores)

        self.plot()

        # ROS 1 initialization
        rospy.init_node('graph_gui', anonymous=True)
        self.ros_subscriber = ScoreSubscriber(self)
        self.ros_thread = threading.Thread(target=self.ros_spin, daemon=True)
        self.ros_thread.start()

    def ros_spin(self):
        rospy.spin()

    def receive_data(self, msg):
        self.data = np.array(msg.data).reshape(1, -1)  # store as 1xN for heatmap

    def plot(self):
        # --- Bar Graph Plot ---
        self.barPlot = pyqtgraph.PlotWidget(self, title="Attention Scores - Bar Graph")
        self.barPlot.setGeometry(0, 0, 600, 300)
        self.barPlot.setYRange(0, 1)  # adjust range to your expected score range
        self.barItem = None

        # --- Timer for Updates ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(200)  # update every 200ms

        self.show()

    def update(self):
        # --- Update Bar Graph ---
        if isinstance(self.data, np.ndarray):
            data_1d = self.data.flatten()
            x = np.arange(len(data_1d))
            self.barPlot.clear()
            self.barItem = pyqtgraph.BarGraphItem(x=x, height=data_1d, width=0.8, brush='b')
            self.barPlot.addItem(self.barItem)

    def closeEvent(self, event):
        rospy.signal_shutdown("GUI closed")
        event.accept()

def main():
    app = QApplication(sys.argv)
    win = Window()

    def shutdown_handler(sig, frame):
        print('shutdown')
        rospy.signal_shutdown("SIGINT received")
        app.quit()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    sys.exit(app.exec())

if __name__ == '__main__':
    main()