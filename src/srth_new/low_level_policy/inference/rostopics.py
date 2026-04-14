#!/usr/bin/env python
# for ros stuff
import rospy
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, JointState

class ros_topics:

  def __init__(self):
    self.bridge = CvBridge()
    self._topics_with_messages = set()
    self._expected_topics = {
      "/jhu_daVinci/left/image_raw/compressed",
      "/jhu_daVinci/right/image_raw/compressed",
      "/PSM2/endoscope_img/compressed",
      "/PSM1/endoscope_img/compressed",
      "/PSM1/setpoint_cp",
      "PSM1/jaw/measured_js",
      "SUJ/PSM1/measured_cp",
      "/PSM2/setpoint_cp",
      "PSM2/jaw/measured_js",
      "SUJ/PSM2/measured_cp",
      "/ECM/measured_cp",
      "/SUJ/ECM/measured_cp",
    }
    self._all_topics_received_logged = False

    self.usb_image_left = None
    self.usb_image_right = None
    self.endo_cam_psm1 = None
    self.endo_cam_psm2 = None
    self.psm1_pose = None
    self.psm1_jaw = None
    self.psm1_rcm_pose = None

    self.psm2_pose = None
    self.psm2_jaw = None
    self.psm2_rcm_pose = None

    self.ecm_pose = None
    self.ecm_rcm_pose = None

    # subscribers
    self.usb_camera_sub_left = rospy.Subscriber("/jhu_daVinci/left/image_raw/compressed", 
                                            CompressedImage, self.get_camera_image_left)
    self.usb_camera_sub_right = rospy.Subscriber("/jhu_daVinci/right/image_raw/compressed", 
                                            CompressedImage, self.get_camera_image_right)
    
    # endoscope imgs
    self.endo_cam_psm1_sub = rospy.Subscriber("/PSM2/endoscope_img/compressed", 
                                            CompressedImage, self.get_endo_cam_psm1)
    self.endo_cam_psm2_sub = rospy.Subscriber("/PSM1/endoscope_img/compressed", 
                                            CompressedImage, self.get_endo_cam_psm2)

    #psm1
    #self.psm1_sub = rospy.Subscriber("/PSM1/measured_cp", 
    self.psm1_sub = rospy.Subscriber("/PSM1/setpoint_cp", 
                                            PoseStamped, self.get_psm1_pose)

    self.psm1_jaw_sub = rospy.Subscriber("PSM1/jaw/measured_js",
                                      JointState, self.get_psm1_jaw)

    self.psm1_rcm_sub = rospy.Subscriber("SUJ/PSM1/measured_cp", 
                                            PoseStamped, self.get_psm1_rcm_pose)

    #psm2
    #self.psm2_sub = rospy.Subscriber("/PSM2/measured_cp", 
    self.psm2_sub = rospy.Subscriber("/PSM2/setpoint_cp", 
                                            PoseStamped, self.get_psm2_pose)
    
    self.psm2_jaw_sub = rospy.Subscriber("PSM2/jaw/measured_js",
                                         JointState, self.get_psm2_jaw)
    
    self.psm2_rcm_sub = rospy.Subscriber("SUJ/PSM2/measured_cp", 
                                            PoseStamped, self.get_psm2_rcm_pose)

    # ecm
    self.ecm_sub = rospy.Subscriber("/ECM/measured_cp",
                                      PoseStamped, self.get_ecm_pose)
    self.ecm_rcm_sub = rospy.Subscriber("/SUJ/ECM/measured_cp",
                                          PoseStamped, self.get_ecm_rcm_pose)

  def _store_topic_value(self, topic_name, attr_name, value):
    if topic_name not in self._topics_with_messages:
      rospy.loginfo("Received first message on topic %s", topic_name)
      self._topics_with_messages.add(topic_name)
      if (
        not self._all_topics_received_logged
        and self._topics_with_messages == self._expected_topics
      ):
        rospy.loginfo(
          "\033[92mAll subscribed ROS topics have received at least one message.\033[0m"
        )
        self._all_topics_received_logged = True
    setattr(self, attr_name, value)

  def get_missing_topics(self):
    return sorted(self._expected_topics - self._topics_with_messages)

  def has_received_all_topics(self):
    return len(self.get_missing_topics()) == 0

  def get_camera_image_left(self,data):
    self._store_topic_value("/jhu_daVinci/left/image_raw/compressed", "usb_image_left", data)
  
  def get_camera_image_right(self,data):
    self._store_topic_value("/jhu_daVinci/right/image_raw/compressed", "usb_image_right", data)

  def get_endo_cam_psm1(self, data):
    #self.endo_cam_psm1 = self.bridge.imgmsg_to_cv2(data, desired_encoding = 'passthrough')
    self._store_topic_value("/PSM2/endoscope_img/compressed", "endo_cam_psm1", data)

  def get_endo_cam_psm2(self, data):
    #self.endo_cam_psm2 = self.bridge.imgmsg_to_cv2(data, desired_encoding = 'passthrough')
    self._store_topic_value("/PSM1/endoscope_img/compressed", "endo_cam_psm2", data)

  def get_ecm_rcm_pose(self, data):
    self._store_topic_value("/SUJ/ECM/measured_cp", "ecm_rcm_pose", data.pose)

  def get_ecm_pose(self, data):
    self._store_topic_value("/ECM/measured_cp", "ecm_pose", data.pose)

  def get_psm1_pose(self, data):
    self._store_topic_value("/PSM1/setpoint_cp", "psm1_pose", data.pose)

  def get_psm1_jaw(self, data):
    self._store_topic_value("PSM1/jaw/measured_js", "psm1_jaw", data.position[0])

  def get_psm1_rcm_pose(self, data):
    self._store_topic_value("SUJ/PSM1/measured_cp", "psm1_rcm_pose", data.pose)

  def get_psm2_pose(self, data):
    self._store_topic_value("/PSM2/setpoint_cp", "psm2_pose", data.pose)
  
  def get_psm2_jaw(self, data):
    self._store_topic_value("PSM2/jaw/measured_js", "psm2_jaw", data.position[0])

  def get_psm2_rcm_pose(self, data):
    self._store_topic_value("SUJ/PSM2/measured_cp", "psm2_rcm_pose", data.pose)
