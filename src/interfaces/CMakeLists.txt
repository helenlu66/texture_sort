cmake_minimum_required(VERSION 3.8)
project(interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObjectRecord.msg"
  "msg/ObjectPose.msg"
  "msg/ObjectGrounding.msg"
  "msg/ObjectGroundingArray.msg"
  "msg/TextureClassification.msg"
  "msg/DeliveryState.msg"
  "msg/TaskState.msg"
  "srv/InitializeSceneObjects.srv"
  "srv/DetectObjectPose.srv"
  "srv/ClassifyTexture.srv"
  "srv/CheckBaseReady.srv"
  "srv/StartSortAndDelivery.srv"
  "srv/GetNextCategory.srv"
  "srv/GetObjectsInCategory.srv"
  "srv/MarkCategoryDelivered.srv"
  "action/ExecuteGrasp.action"
  "action/ExecutePlace.action"
  "action/LoadObjectIntoBox.action"
  DEPENDENCIES action_msgs builtin_interfaces geometry_msgs sensor_msgs std_msgs std_srvs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
