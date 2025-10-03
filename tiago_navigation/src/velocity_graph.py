#!/usr/bin/env python

import signal
import sys
import threading

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow
import pyqtgraph
import rospy
from std_msgs.msg import Float32MultiArray

class GraphSubscriber:
    def __init__(self, window):
        self.window = window
        self.sub = rospy.Subscriber(  # ROS 1 style subscriber
            '/mean_velocity', 
            Float32MultiArray, 
            self.data_callback
        )

    def data_callback(self, msg):
        self.window.receive_data(msg)


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle('Network Error')
        self.setGeometry(50, 50, 600, 650)

        self.ep = []
        self.critic_loss = []
        self.actor_loss = []
        self.count = 1

        self.plot()

        # ROS 1 initialization
        rospy.init_node('graph_gui_node', anonymous=True)
        self.ros_subscriber = GraphSubscriber(self)
        self.ros_thread = threading.Thread(
            target=self.ros_spin, daemon=True  # Custom spin for ROS 1
        )
        self.ros_thread.start()

    def ros_spin(self):
        rospy.spin()  # ROS 1 spin

    def receive_data(self, msg):
        self.critic_loss.append(msg.data[0])
        self.ep.append(self.count)
        self.count += 1
        self.actor_loss.append(msg.data[1])

    def plot(self):
        self.criticPlt = pyqtgraph.PlotWidget(self, title='Linear Velocity')
        self.criticPlt.setGeometry(0, 320, 600, 300)

        self.actorPlt = pyqtgraph.PlotWidget(self, title='Angular Velocity')
        self.actorPlt.setGeometry(0, 10, 600, 300)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(200)

        self.show()

    def update(self):
        self.actorPlt.showGrid(x=True, y=True)
        self.criticPlt.showGrid(x=True, y=True)

        self.actorPlt.plot(self.ep, self.actor_loss, pen=(255, 0, 0), clear=True)
        self.criticPlt.plot(self.ep, self.critic_loss, pen=(0, 255, 0), clear=True)

    def closeEvent(self, event):
        rospy.signal_shutdown("GUI closed")  # ROS 1 shutdown
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = Window()

    def shutdown_handler(sig, frame):
        print('shutdown')
        rospy.signal_shutdown("SIGINT received")  # ROS 1 shutdown
        app.quit()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    sys.exit(app.exec())

if __name__ == '__main__':
    main()