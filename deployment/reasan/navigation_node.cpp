#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fmt/format.h>
#include <memory>
#include <string>
#include <vector>

// ROS2 headers
#include "geometry_msgs/msg/point.hpp"
#include "livox_ros_driver2/msg/custom_msg.hpp"
#include "livox_ros_driver2/msg/custom_point.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

// Eigen headers
#include <Eigen/Core>
#include <Eigen/Geometry>

// argument parser
#include "cxxopts.hpp"
#include "yaml-cpp/yaml.h"

// ONNX Runtime headers
#include "onnxruntime_cxx_api.h"

// Unitree SDK headers
#include "unitree/common/time/time_tool.hpp"
#include "unitree/idl/go2/LowState_.hpp"
#include "unitree/robot/channel/channel_factory.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"

using namespace std::chrono_literals;
using namespace Eigen;
using namespace unitree::robot;

struct NavigationConfig {
    int control_freq = 50;
    std::string lowstate_topic = "rt/lowstate";
    std::string navigation_policy_path;
    std::string navigation_policy_name = "navigation_policy";
    int num_estimated_rays = 180;
    int num_actions = 3;
    int num_proprio_obs = 11; // 3 gyro + 3 gravity + 2 goal + 3 high_actions
    int lstm_num_layers = 1;
    int lstm_hidden_size = 256;
    bool enable_visualization = false;
    bool manual_rotation = false;

    // Frequency monitoring
    float frequency_ema_alpha = 0.1f;
    int frequency_publish_interval = 50;

    void init() {
        char result[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
        std::string exe_path = std::string(result, (count > 0) ? count : 0);
        std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();

        navigation_policy_path =
            (exe_dir / "./model" / fmt::format("{}.onnx", navigation_policy_name)).string();

        // Load frequency monitoring config
        YAML::Node yaml_conf = YAML::LoadFile("config.yaml");
        if (yaml_conf["frequency_monitoring"]) {
            frequency_ema_alpha = yaml_conf["frequency_monitoring"]["ema_alpha"].as<float>(0.1f);
            frequency_publish_interval =
                yaml_conf["frequency_monitoring"]["publish_interval"].as<int>(50);
        }
    }
};

class NavigationNode : public rclcpp::Node {
  public:
    NavigationNode(NavigationConfig config)
        : Node("navigation_node")
        , config_(config)
        , initialized_(false)
        , is_rotating_to_goal_(false) {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "NavigationNode");

        estimated_rays_subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/estimated_rays", 10,
            std::bind(&NavigationNode::estimated_rays_callback, this, std::placeholders::_1));

        navigation_goal_subscription_ = this->create_subscription<geometry_msgs::msg::Point>(
            "/navigation_goal", 10,
            std::bind(&NavigationNode::navigation_goal_callback, this, std::placeholders::_1));

        odom_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/go2_odom", 10,
            std::bind(&NavigationNode::odom_callback, this, std::placeholders::_1));

        command_publisher_ =
            this->create_publisher<geometry_msgs::msg::Point>("/navigation_vel_cmd", 10);

        frequency_publisher_ = this->create_publisher<std_msgs::msg::Float32>(
            "/navigation_node/control_frequency", 10);
        last_loop_time_ = std::chrono::steady_clock::now();

        if (config_.enable_visualization) {
            viz_marker_publisher_ =
                this->create_publisher<visualization_msgs::msg::MarkerArray>("/navigation_viz", 10);
            pointcloud_viz_publisher_ =
                this->create_publisher<sensor_msgs::msg::PointCloud2>("/navigation_pointcloud", 10);
            // Subscribe to Livox CustomMsg format
            pointcloud_subscription_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
                "/livox/lidar", 10,
                std::bind(&NavigationNode::pointcloud_callback, this, std::placeholders::_1));
            RCLCPP_INFO(this->get_logger(), "Visualization enabled");
        }

        load_config();
        initialize_controller();

        RCLCPP_INFO(this->get_logger(), "NavigationNode initialized");
    }

    void run_control_loop() {
        if (!initialized_) {
            RCLCPP_ERROR(this->get_logger(), "NavigationNode not initialized. Exiting.");
            return;
        }

        rclcpp::Rate rate(config_.control_freq);
        int i = 0;
        while (rclcpp::ok()) {
            auto loop_start = std::chrono::steady_clock::now();

            if (i++ % 50 == 0) {
                RCLCPP_INFO(
                    this->get_logger(),
                    "Current goal: x: %.2f, y: %.2f | Robot pos: x: %.2f, y: %.2f",
                    navigation_goal_[0], navigation_goal_[1], robot_pos_[0], robot_pos_[1]);
            }
            run_controller();
            if (config_.enable_visualization) {
                publish_visualization();
            }

            // Update and publish frequency
            auto loop_end = std::chrono::steady_clock::now();
            float loop_dt = std::chrono::duration<float>(loop_end - last_loop_time_).count();
            last_loop_time_ = loop_end;
            if (loop_dt > 0.0f) {
                float instantaneous_freq = 1.0f / loop_dt;
                estimated_frequency_ =
                    (estimated_frequency_ == 0.0f)
                        ? instantaneous_freq
                        : config_.frequency_ema_alpha * instantaneous_freq +
                              (1.0f - config_.frequency_ema_alpha) * estimated_frequency_;
            }
            if (++frequency_publish_counter_ >= config_.frequency_publish_interval) {
                auto freq_msg = std_msgs::msg::Float32();
                freq_msg.data = estimated_frequency_;
                frequency_publisher_->publish(freq_msg);
                frequency_publish_counter_ = 0;
            }

            rate.sleep();
        }
    }

  private:
    void load_config() {
        estimated_rays_.resize(config_.num_estimated_rays, 10.0f);
        navigation_goal_ = {0.0f, 0.0f};
        high_actions_ = {0.0f, 0.0f, 0.0f};
        robot_pos_ = {0.0f, 0.0f, 0.0f};
        robot_quat_ = Quaternionf(1.0f, 0.0f, 0.0f, 0.0f);

        // Initialize LSTM states
        navigation_policy_h_.resize(config_.lstm_num_layers * config_.lstm_hidden_size, 0.0f);
        navigation_policy_c_.resize(config_.lstm_num_layers * config_.lstm_hidden_size, 0.0f);

        RCLCPP_INFO(this->get_logger(), "Navigation configuration loaded");
    }

    void initialize_controller() {
        try {
            // Initialize Unitree SDK subscriber for low state
            lowstate_subscriber_.reset(
                new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(config_.lowstate_topic));
            lowstate_subscriber_->InitChannel(
                std::bind(&NavigationNode::low_state_callback, this, std::placeholders::_1), 10);

            initialize_onnx_runtime();
            initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "Navigation controller initialized");
        } catch (const std::exception &e) {
            RCLCPP_ERROR(
                this->get_logger(), "Failed to initialize navigation controller: %s", e.what());
        }
    }

    void initialize_onnx_runtime() {
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        OrtTensorRTProviderOptionsV2 *tensorrt_options = nullptr;
        Ort::GetApi().CreateTensorRTProviderOptions(&tensorrt_options);
        std::vector<const char *> option_keys = {
            "trt_fp16_enable", "trt_engine_cache_enable", "trt_engine_cache_path"};
        std::vector<const char *> option_values = {"1", "1", "./trt_engine_cache"};
        Ort::GetApi().UpdateTensorRTProviderOptions(
            tensorrt_options, option_keys.data(), option_values.data(), option_keys.size());
        session_options.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);
        Ort::GetApi().ReleaseTensorRTProviderOptions(tensorrt_options);

        try {
            RCLCPP_INFO(
                this->get_logger(), "Loading navigation policy from %s",
                config_.navigation_policy_path.c_str());
            navigation_session_ = std::make_unique<Ort::Session>(
                env_, config_.navigation_policy_path.c_str(), session_options);
            RCLCPP_INFO(this->get_logger(), "Navigation policy ONNX model loaded successfully");
        } catch (const Ort::Exception &e) {
            RCLCPP_ERROR(
                this->get_logger(), "Navigation policy ONNX initialization error: %s", e.what());
            throw;
        }
    }

    void estimated_rays_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
        if (msg->data.size() == config_.num_estimated_rays) {
            estimated_rays_ = msg->data;
        } else {
            RCLCPP_WARN(
                this->get_logger(),
                "Received estimated_rays with incorrect size: %zu, expected: %d", msg->data.size(),
                config_.num_estimated_rays);
        }
    }

    void navigation_goal_callback(const geometry_msgs::msg::Point::SharedPtr msg) {
        navigation_goal_[0] = msg->x;
        navigation_goal_[1] = msg->y;
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        robot_pos_[0] = msg->pose.pose.position.x;
        robot_pos_[1] = msg->pose.pose.position.y;
        robot_pos_[2] = msg->pose.pose.position.z;

        robot_quat_.w() = msg->pose.pose.orientation.w;
        robot_quat_.x() = msg->pose.pose.orientation.x;
        robot_quat_.y() = msg->pose.pose.orientation.y;
        robot_quat_.z() = msg->pose.pose.orientation.z;
    }

    void pointcloud_callback(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg) {
        // Store raw points for visualization
        latest_points_.clear();
        latest_points_.reserve(msg->point_num);

        for (int i = 0; i < msg->point_num; ++i) {
            auto &point = msg->points[i];
            if (std::abs(point.x) < 1e-3 && std::abs(point.y) < 1e-3 && std::abs(point.z) < 1e-3) {
                continue;
            }
            Vector3f p(point.x, point.y, point.z);
            latest_points_.push_back(p);
        }
    }

    void run_controller() {
        try {
            std::vector<float> modified_action = run_navigation_policy();

            // Apply manual rotation if enabled
            if (config_.manual_rotation) {
                // Calculate goal direction in world frame
                float goal_direction_world = std::atan2(
                    navigation_goal_[1] - robot_pos_[1], navigation_goal_[0] - robot_pos_[0]);

                // Calculate current yaw from quaternion
                float current_yaw = std::atan2(
                    2.0f * (robot_quat_.w() * robot_quat_.z() + robot_quat_.x() * robot_quat_.y()),
                    1.0f - 2.0f * (robot_quat_.y() * robot_quat_.y() +
                                   robot_quat_.z() * robot_quat_.z()));

                // Calculate yaw error
                float yaw_error = goal_direction_world - current_yaw;

                // Normalize angle to [-pi, pi]
                while (yaw_error > M_PI)
                    yaw_error -= 2.0f * M_PI;
                while (yaw_error < -M_PI)
                    yaw_error += 2.0f * M_PI;

                // Threshold for considering "facing the goal" (30 degrees = pi/6)
                const float rotation_threshold = M_PI / 6.0f;
                // Threshold for completing rotation (5.7 degrees = 0.1 rad)
                const float rotation_complete_threshold = 0.1f;

                if (is_rotating_to_goal_) {
                    // Currently rotating - check if rotation is complete
                    if (std::abs(yaw_error) < rotation_complete_threshold) {
                        // Rotation complete
                        is_rotating_to_goal_ = false;
                        RCLCPP_INFO(
                            this->get_logger(), "Rotation complete, resuming policy control");
                        // Use policy output
                    } else {
                        // Continue rotating, override policy output
                        modified_action[0] = 0.0f; // No forward velocity
                        modified_action[1] = 0.0f; // No lateral velocity
                        modified_action[2] = std::clamp(yaw_error, -0.5f, 0.5f); // Angular velocity
                    }
                } else {
                    // Not currently rotating - check if we need to start
                    if (std::abs(yaw_error) > rotation_threshold) {
                        // Not facing goal, enter rotation mode
                        is_rotating_to_goal_ = true;
                        RCLCPP_INFO(
                            this->get_logger(),
                            "Not facing goal (error: %.2f deg), entering rotation mode",
                            yaw_error * 180.0f / M_PI);
                        modified_action[0] = 0.0f;
                        modified_action[1] = 0.0f;
                        modified_action[2] = std::clamp(yaw_error, -0.5f, 0.5f);
                    }
                    // else: facing goal, use policy output as is
                }
            }

            auto command_msg = std::make_unique<geometry_msgs::msg::Point>();
            command_msg->x = modified_action[0];
            command_msg->y = modified_action[1];
            command_msg->z = modified_action[2];
            command_publisher_->publish(std::move(command_msg));
        } catch (const Ort::Exception &e) {
            RCLCPP_ERROR(
                this->get_logger(), "Navigation policy ONNX inference error: %s", e.what());
        }
    }

    std::vector<float> run_navigation_policy() {
        // Create proprio observation (11 dimensions)
        std::vector<float> proprio_obs(config_.num_proprio_obs);

        // Base angular velocity * 0.25 (3 values) - from IMU
        proprio_obs[0] = low_state_.imu_state().gyroscope()[0] * 0.25f;
        proprio_obs[1] = low_state_.imu_state().gyroscope()[1] * 0.25f;
        proprio_obs[2] = low_state_.imu_state().gyroscope()[2] * 0.25f;

        // Projected gravity in body frame (3 values)
        Vector3f gravity_orientation = get_gravity_orientation_eigen(robot_quat_);
        proprio_obs[3] = gravity_orientation[0];
        proprio_obs[4] = gravity_orientation[1];
        proprio_obs[5] = gravity_orientation[2];

        // Goal position in body frame (2 values)
        // Transform goal from world frame to body frame
        Vector3f goal_world(navigation_goal_[0], navigation_goal_[1], 0.0f);
        Vector3f robot_to_goal_world = goal_world - robot_pos_;
        Vector3f goal_body = robot_quat_.inverse() * robot_to_goal_world;
        proprio_obs[6] = goal_body[0];
        proprio_obs[7] = goal_body[1];

        // High actions (3 values: last raw output from navigation policy)
        proprio_obs[8] = high_actions_[0];
        proprio_obs[9] = high_actions_[1];
        proprio_obs[10] = high_actions_[2];

        // Prepare input dimensions
        const std::vector<int64_t> proprio_dims = {1, config_.num_proprio_obs};
        const std::vector<int64_t> ray_dims = {1, config_.num_estimated_rays};
        const std::vector<int64_t> lstm_dims = {
            config_.lstm_num_layers, 1, config_.lstm_hidden_size};

        // Define input/output names
        const std::array<const char *, 4> input_names = {"proprio_obs", "ray_obs", "h_in", "c_in"};
        const std::array<const char *, 3> output_names = {"actions", "h_out", "c_out"};

        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        std::vector<Ort::Value> input_tensors;

        // Create input tensors
        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, proprio_obs.data(), proprio_obs.size(), proprio_dims.data(),
                proprio_dims.size()));

        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float *>(estimated_rays_.data()), estimated_rays_.size(),
                ray_dims.data(), ray_dims.size()));

        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, navigation_policy_h_.data(), navigation_policy_h_.size(),
                lstm_dims.data(), lstm_dims.size()));

        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, navigation_policy_c_.data(), navigation_policy_c_.size(),
                lstm_dims.data(), lstm_dims.size()));

        auto output_tensors = navigation_session_->Run(
            Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
            input_tensors.size(), output_names.data(), output_names.size());

        // Extract actions
        float *actions_ptr = output_tensors[0].GetTensorMutableData<float>();
        std::vector<float> actions(actions_ptr, actions_ptr + config_.num_actions);

        // Update LSTM hidden and cell states
        float *h_out_ptr = output_tensors[1].GetTensorMutableData<float>();
        float *c_out_ptr = output_tensors[2].GetTensorMutableData<float>();

        std::copy_n(h_out_ptr, navigation_policy_h_.size(), navigation_policy_h_.begin());
        std::copy_n(c_out_ptr, navigation_policy_c_.size(), navigation_policy_c_.begin());

        // Update high_actions_ for next iteration
        high_actions_ = actions;

        return actions;
    }

    Vector3f get_gravity_orientation_eigen(const Quaternionf &quat) {
        Vector3f gravity_orientation;
        gravity_orientation.x() = 2.0f * (-quat.z() * quat.x() + quat.w() * quat.y());
        gravity_orientation.y() = -2.0f * (quat.z() * quat.y() + quat.w() * quat.x());
        gravity_orientation.z() = 1.0f - 2.0f * (quat.w() * quat.w() + quat.z() * quat.z());
        return gravity_orientation;
    }

    void publish_visualization() {
        visualization_msgs::msg::MarkerArray marker_array;
        auto now = this->get_clock()->now();

        // 1. Robot position marker (arrow showing orientation)
        visualization_msgs::msg::Marker robot_marker;
        robot_marker.header.frame_id = "odom";
        robot_marker.header.stamp = now;
        robot_marker.ns = "navigation";
        robot_marker.id = 0;
        robot_marker.type = visualization_msgs::msg::Marker::ARROW;
        robot_marker.action = visualization_msgs::msg::Marker::ADD;
        robot_marker.pose.position.x = robot_pos_[0];
        robot_marker.pose.position.y = robot_pos_[1];
        robot_marker.pose.position.z = robot_pos_[2];
        robot_marker.pose.orientation.w = robot_quat_.w();
        robot_marker.pose.orientation.x = robot_quat_.x();
        robot_marker.pose.orientation.y = robot_quat_.y();
        robot_marker.pose.orientation.z = robot_quat_.z();
        robot_marker.scale.x = 0.5;
        robot_marker.scale.y = 0.1;
        robot_marker.scale.z = 0.1;
        robot_marker.color.r = 0.0;
        robot_marker.color.g = 1.0;
        robot_marker.color.b = 0.0;
        robot_marker.color.a = 1.0;
        marker_array.markers.push_back(robot_marker);

        // 2. Goal position marker (red sphere)
        visualization_msgs::msg::Marker goal_marker;
        goal_marker.header.frame_id = "odom";
        goal_marker.header.stamp = now;
        goal_marker.ns = "navigation";
        goal_marker.id = 1;
        goal_marker.type = visualization_msgs::msg::Marker::SPHERE;
        goal_marker.action = visualization_msgs::msg::Marker::ADD;
        goal_marker.pose.position.x = navigation_goal_[0];
        goal_marker.pose.position.y = navigation_goal_[1];
        goal_marker.pose.position.z = 0.0;
        goal_marker.pose.orientation.w = 1.0;
        goal_marker.scale.x = 0.3;
        goal_marker.scale.y = 0.3;
        goal_marker.scale.z = 0.3;
        goal_marker.color.r = 1.0;
        goal_marker.color.g = 0.0;
        goal_marker.color.b = 0.0;
        goal_marker.color.a = 1.0;
        marker_array.markers.push_back(goal_marker);

        // 3. Visualize estimated rays in world frame
        for (size_t i = 0; i < estimated_rays_.size(); ++i) {
            float distance = estimated_rays_[i];
            if (distance >= 10.0f)
                continue; // Skip max range rays

            // Calculate ray direction in body frame
            float angle = -M_PI + (2.0f * M_PI * i / estimated_rays_.size());
            Vector3f ray_body(distance * std::cos(angle), distance * std::sin(angle), 0.0f);

            // Transform to world frame
            Vector3f ray_world = robot_quat_ * ray_body + robot_pos_;

            // Sphere marker at ray endpoint
            visualization_msgs::msg::Marker ray_sphere;
            ray_sphere.header.frame_id = "odom";
            ray_sphere.header.stamp = now;
            ray_sphere.ns = "navigation_rays";
            ray_sphere.id = 100 + i;
            ray_sphere.type = visualization_msgs::msg::Marker::SPHERE;
            ray_sphere.action = visualization_msgs::msg::Marker::ADD;
            ray_sphere.pose.position.x = ray_world[0];
            ray_sphere.pose.position.y = ray_world[1];
            ray_sphere.pose.position.z = ray_world[2];
            ray_sphere.pose.orientation.w = 1.0;
            ray_sphere.scale.x = 0.05;
            ray_sphere.scale.y = 0.05;
            ray_sphere.scale.z = 0.05;
            float normalized_distance = distance / 10.0f;
            ray_sphere.color.r = 1.0 - normalized_distance;
            ray_sphere.color.g = normalized_distance;
            ray_sphere.color.b = 0.2;
            ray_sphere.color.a = 0.8;
            ray_sphere.lifetime = rclcpp::Duration(0, 200000000);
            marker_array.markers.push_back(ray_sphere);

            // Line marker from robot to ray endpoint
            visualization_msgs::msg::Marker ray_line;
            ray_line.header.frame_id = "odom";
            ray_line.header.stamp = now;
            ray_line.ns = "navigation_ray_lines";
            ray_line.id = 100 + estimated_rays_.size() + i;
            ray_line.type = visualization_msgs::msg::Marker::LINE_LIST;
            ray_line.action = visualization_msgs::msg::Marker::ADD;
            geometry_msgs::msg::Point p1, p2;
            p1.x = robot_pos_[0];
            p1.y = robot_pos_[1];
            p1.z = robot_pos_[2];
            p2.x = ray_world[0];
            p2.y = ray_world[1];
            p2.z = ray_world[2];
            ray_line.points = {p1, p2};
            ray_line.scale.x = 0.01;
            ray_line.color.r = 1.0;
            ray_line.color.g = 0.2;
            ray_line.color.b = 0.2;
            ray_line.color.a = 0.6;
            ray_line.lifetime = rclcpp::Duration(0, 200000000);
            marker_array.markers.push_back(ray_line);
        }

        viz_marker_publisher_->publish(marker_array);

        // 4. Publish point cloud in world frame (PointCloud2 format)
        if (!latest_points_.empty() && pointcloud_viz_publisher_) {
            auto msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
            msg->header.stamp = now;
            msg->header.frame_id = "odom";

            // Transform points to world frame
            std::vector<Vector3f> world_points;
            world_points.reserve(latest_points_.size());
            for (const auto &point_body : latest_points_) {
                Vector3f point_world = robot_quat_ * point_body + robot_pos_;
                world_points.push_back(point_world);
            }

            msg->height = 1;
            msg->width = world_points.size();
            msg->is_dense = true;
            msg->is_bigendian = false;

            // Define PointCloud2 fields
            sensor_msgs::msg::PointField field_x, field_y, field_z;
            field_x.name = "x";
            field_x.offset = 0;
            field_x.datatype = sensor_msgs::msg::PointField::FLOAT32;
            field_x.count = 1;
            field_y.name = "y";
            field_y.offset = 4;
            field_y.datatype = sensor_msgs::msg::PointField::FLOAT32;
            field_y.count = 1;
            field_z.name = "z";
            field_z.offset = 8;
            field_z.datatype = sensor_msgs::msg::PointField::FLOAT32;
            field_z.count = 1;
            msg->fields = {field_x, field_y, field_z};

            msg->point_step = 12;
            msg->row_step = msg->point_step * world_points.size();
            msg->data.resize(msg->row_step);

            // Fill point cloud data
            for (size_t i = 0; i < world_points.size(); ++i) {
                memcpy(&msg->data[i * 12 + 0], &world_points[i].x(), sizeof(float));
                memcpy(&msg->data[i * 12 + 4], &world_points[i].y(), sizeof(float));
                memcpy(&msg->data[i * 12 + 8], &world_points[i].z(), sizeof(float));
            }

            pointcloud_viz_publisher_->publish(std::move(msg));
        }
    }

    void low_state_callback(const void *msg) {
        low_state_ = *(unitree_go::msg::dds_::LowState_ *)msg;
    }

    NavigationConfig config_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr estimated_rays_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr navigation_goal_subscription_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr pointcloud_subscription_;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr command_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viz_marker_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_viz_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr frequency_publisher_;

    // Add Unitree SDK subscriber
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber_;
    unitree_go::msg::dds_::LowState_ low_state_;

    Ort::Env env_;
    std::unique_ptr<Ort::Session> navigation_session_;

    std::vector<float> estimated_rays_;
    std::vector<float> navigation_goal_;
    std::vector<float> high_actions_;
    Vector3f robot_pos_;
    Quaternionf robot_quat_;

    // LSTM states
    std::vector<float> navigation_policy_h_;
    std::vector<float> navigation_policy_c_;

    // For visualization - store latest point cloud in body frame
    std::vector<Vector3f> latest_points_;

    bool initialized_;
    bool is_rotating_to_goal_;

    // Frequency monitoring
    std::chrono::steady_clock::time_point last_loop_time_;
    float estimated_frequency_ = 0.0f;
    int frequency_publish_counter_ = 0;
};

int main(int argc, char *argv[]) {
    cxxopts::Options options("NavigationNode", "Unitree Go2 Navigation Node");
    options.add_options()(
        "n,net", "network interface", cxxopts::value<std::string>()->default_value("lo"))(
        "p,policy", "navigation policy", cxxopts::value<std::string>()->default_value(""))(
        "v,viz", "enable visualization", cxxopts::value<bool>()->default_value("false"))(
        "r,rotation", "enable manual rotation",
        cxxopts::value<bool>()->default_value("false"))("h,help", "Print usage");
    auto args = options.parse(argc, argv);

    // Initialize Unitree SDK
    auto net_interface = args["net"].as<std::string>();
    int net_idx = net_interface == "lo" ? 1 : 0;
    ChannelFactory::Instance()->Init(net_idx, net_interface);

    rclcpp::init(argc, argv);

    NavigationConfig config;
    if (!args["policy"].as<std::string>().empty()) {
        config.navigation_policy_name = args["policy"].as<std::string>();
    }
    config.enable_visualization = args["viz"].as<bool>();
    config.manual_rotation = args["rotation"].as<bool>();

    config.init();
    auto node = std::make_shared<NavigationNode>(config);

    std::thread spin_thread([&]() {
        rclcpp::executors::MultiThreadedExecutor executor;
        executor.add_node(node);
        executor.spin();
        rclcpp::shutdown();
    });
    spin_thread.detach();

    node->run_control_loop();

    return 0;
}