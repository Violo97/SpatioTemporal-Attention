#!/usr/bin/env python


import signal
import sys
import threading
import time  # <-- New import

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow
import pyqtgraph
import rospy
from std_msgs.msg import Float32MultiArray

class GraphSubscriber:
    def __init__(self, window):
        self.window = window
        self.sub = rospy.Subscriber(
            '/tiago_navigation/reward', 
            Float32MultiArray, 
            self.data_callback
        )

    def data_callback(self, msg):
        self.window.receive_data(msg)

class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle('Reward')
        self.setGeometry(50, 50, 1250, 1000)

        # Data storage with thread protection
        self.data_lock = threading.Lock()
        self.ep = []
        self.collision_reward = [] 
        self.proximity_reward = [] 
        self.guide_reward = [] 
        self.angular_reward = []
        self.reward = []
        self.count = 1
        self.last_received_time = time.time()  # <-- Track last received time

        self.plot()

        # ROS initialization
        rospy.init_node('reward_gui_node', anonymous=True)
        self.ros_subscriber = GraphSubscriber(self)
        self.ros_thread = threading.Thread(
            target=self.ros_spin, daemon=True
        )
        self.ros_thread.start()

    def ros_spin(self):
        rospy.spin()

    def receive_data(self, msg):
        with self.data_lock:
            self.collision_reward.append(msg.data[0])
            self.proximity_reward.append(msg.data[1])
            self.guide_reward.append(msg.data[2])
            self.angular_reward.append(msg.data[3])
            self.ep.append(self.count)
            self.count += 1
            self.reward.append(msg.data[4])
        self.last_received_time = time.time()  # <-- Update timestamp on new data

    def plot(self):
        self.CollisionPlt = pyqtgraph.PlotWidget(self, title='Collision reward')
        self.CollisionPlt.setGeometry(0, 10, 600, 300)
        self.ProximityPlt = pyqtgraph.PlotWidget(self, title='Proximity reward')
        self.ProximityPlt.setGeometry(610, 10, 600, 300)
        self.GuidePlt = pyqtgraph.PlotWidget(self, title='Guide reward')
        self.GuidePlt.setGeometry(0, 320, 600, 300)
        self.AngularPlt = pyqtgraph.PlotWidget(self, title='Angular reward')
        self.AngularPlt.setGeometry(610, 320, 600, 300)
        self.rewardsPlt = pyqtgraph.PlotWidget(self, title='Total reward')
        self.rewardsPlt.setGeometry(0, 640, 600, 300)
        

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(200)  # Check every 200ms
        self.show()

    def update(self):
        current_time = time.time()
        
        # Reset plot if no data for 1 second
        if current_time - self.last_received_time > 1.0:
            with self.data_lock:
                self.ep = []
                self.collision_reward = [] 
                self.proximity_reward = [] 
                self.guide_reward = [] 
                self.angular_reward = []
                self.reward = []
                self.count = 1
            self.CollisionPlt.clear()
            self.ProximityPlt.clear()
            self.GuidePlt.clear()
            self.AngularPlt.clear()
            self.rewardsPlt.clear()
            
        else:
            with self.data_lock:
                ep_copy = self.ep.copy()
                collision_reward_copy = self.collision_reward.copy()
                proximity_reward_copy = self.proximity_reward.copy()
                guide_reward_copy = self.guide_reward.copy()
                angular_reward_copy = self.angular_reward.copy()
                reward_copy = self.reward.copy()
            self.CollisionPlt.showGrid(x=True, y=True)
            self.ProximityPlt.showGrid(x=True, y=True)
            self.GuidePlt.showGrid(x=True, y=True)
            self.AngularPlt.showGrid(x=True, y=True)
            self.rewardsPlt.showGrid(x=True, y=True)
            self.CollisionPlt.plot(ep_copy, collision_reward_copy, pen=(255, 0, 0), clear=True)
            self.ProximityPlt.plot(ep_copy, proximity_reward_copy, pen=(0, 255, 0), clear=True)
            self.GuidePlt.plot(ep_copy, guide_reward_copy, pen=(0, 0, 255), clear=True)
            self.AngularPlt.plot(ep_copy, angular_reward_copy, pen=(255, 255, 0), clear=True)
            self.rewardsPlt.plot(ep_copy, reward_copy, pen=(255, 0, 0), clear=True)
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