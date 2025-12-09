#include <array>
#include <memory>
#include <string>

#include <geometry_msgs/msg/pose.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

#include <Eigen/Geometry>
#include <fmt/format.h>

/**
 * @brief Node that subscribes to mocap data and publishes to ROS2 Odometry
 */
class OdomNode : public rclcpp::Node {
  public:
    OdomNode()
        : Node("odom_node") {
        // Initialize ROS2 publisher for odometry
        odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/go2_odom", 10);

        // Initialize odometry subscription
        odometry_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odometry", 10,
            std::bind(&OdomNode::odometry_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Odom Node has been started");
    }

  private:
    void odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        // Extract position
        float pos_x = msg->pose.pose.position.x;
        float pos_y = msg->pose.pose.position.y;
        float pos_z = msg->pose.pose.position.z;

        // Extract orientation (quaternion)
        float quat_w = msg->pose.pose.orientation.w;
        float quat_x = msg->pose.pose.orientation.x;
        float quat_y = msg->pose.pose.orientation.y;
        float quat_z = msg->pose.pose.orientation.z;

        // Create Eigen objects for vector math
        Eigen::Quaternionf quat(quat_w, quat_x, quat_y, quat_z);
        Eigen::Vector3f body_forward(1.0f, 0.0f, 0.0f);
        Eigen::Vector3f world_forward = quat * body_forward;
        Eigen::Vector3f position(pos_x, pos_y, pos_z);

        // Log position and forward vector
        RCLCPP_INFO(
            this->get_logger(),
            "world pos: [%.3f, %.3f, %.3f] | world forward: [%.3f, %.3f, %.3f] | world "
            "quat: [%.3f, %.3f, %.3f, %.3f]",
            position.x(), position.y(), position.z(), world_forward.x(), world_forward.y(),
            world_forward.z(), quat_w, quat_x, quat_y, quat_z);

        // Simply forward the odometry message
        odom_publisher_->publish(*msg);
    }

    // ROS subscriber
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_subscription_;

    // ROS publisher for odometry messages
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<OdomNode>();

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
