#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
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

Vector3f get_gravity_orientation_eigen(const Quaternionf &quat) {
    Vector3f gravity_orientation;
    gravity_orientation.x() = 2.0f * (-quat.z() * quat.x() + quat.w() * quat.y());
    gravity_orientation.y() = -2.0f * (quat.z() * quat.y() + quat.w() * quat.x());
    gravity_orientation.z() = 1.0f - 2.0f * (quat.w() * quat.w() + quat.z() * quat.z());
    return gravity_orientation;
}

VectorXf project_points_to_spherical_grid(
    const std::vector<Vector3f> &points,
    float phi_range_min,
    float phi_range_max,
    float theta_range_min,
    float theta_range_max,
    int phi_res_deg,
    int theta_res_deg,
    float max_range) {

    float phi_res = phi_res_deg * M_PI / 180.0f;
    float theta_res = theta_res_deg * M_PI / 180.0f;

    int phi_bins = static_cast<int>((phi_range_max - phi_range_min) / phi_res_deg);
    int theta_bins = static_cast<int>((theta_range_max - theta_range_min) / theta_res_deg);

    float phi_min = phi_range_min * M_PI / 180.0f;
    float phi_max = phi_range_max * M_PI / 180.0f;
    float theta_min = theta_range_min * M_PI / 180.0f;
    float theta_max = theta_range_max * M_PI / 180.0f;

    VectorXf grid = VectorXf::Constant(theta_bins * phi_bins, max_range);

    for (const auto &point : points) {
        float x = point.x();
        float y = point.y();
        float z = point.z();

        float r = std::sqrt(x * x + y * y + z * z);
        if (r < 0.1f)
            continue;

        float theta = std::asin(z / r);
        float phi = std::atan2(y, x);

        if (phi < phi_min || phi > phi_max || theta < theta_min || theta > theta_max) {
            continue;
        }
        if (r >= max_range) {
            continue;
        }

        int phi_idx = static_cast<int>((phi - phi_min) / phi_res);
        int theta_idx = static_cast<int>((theta - theta_min) / theta_res);

        phi_idx = std::clamp(phi_idx, 0, phi_bins - 1);
        theta_idx = std::clamp(theta_idx, 0, theta_bins - 1);

        int grid_idx = theta_idx * phi_bins + phi_idx;
        grid[grid_idx] = std::min(grid[grid_idx], r);
    }

    return grid;
}

struct RayEstimationConfig {
    int control_freq = 50;
    std::string lowstate_topic = "rt/lowstate";
    std::string ray_estimation_policy_path;
    std::string ray_predictor_model = "ray_predictor";

    float phi_range_min = -180.0f;
    float phi_range_max = 180.0f;
    float theta_range_min = -5.0f;
    float theta_range_max = 55.0f;
    int phi_res_deg = 2;
    int theta_res_deg = 2;
    float max_range = 3.0f;

    int phi_bins = (phi_range_max - phi_range_min) / phi_res_deg;
    int theta_bins = (theta_range_max - theta_range_min) / theta_res_deg;

    int spherical_grid_history_len = 15;
    int imu_history_len = 15;
    int num_estimated_rays = 180;

    // Will be detected automatically from the model
    int imu_channels = 6;                    // Default to 6 (gravity + angular velocity)
    std::string imu_input_name = "imu_data"; // Will be detected from model

    // Lidar rotation angle (in degrees) - positive rotates downward
    float lidar_pitch_angle_deg = 10.0f;

    // Lidar position offset from robot center (in meters)
    float lidar_offset_x = 0.3f; // Forward offset
    float lidar_offset_y = 0.0f; // Lateral offset
    float lidar_offset_z = 0.0f; // Vertical offset

    // Visualization toggle
    bool enable_visualization = false;

    // Frequency monitoring
    float frequency_ema_alpha = 0.1f;
    int frequency_publish_interval = 50;

    // Ray filtering
    float ray_filter_threshold = 0.5f;
    float ray_filter_multiplier = 0.5f;
    float ray_filter_smoothness = 0.2f; // Transition width around threshold

    // Point cloud filtering
    float min_point_distance = 0.1f; // Minimum distance from origin
    float filter_cube_x_min = -0.3f; // Cube bounds for filtering points
    float filter_cube_x_max = 0.3f;
    float filter_cube_y_min = -0.2f;
    float filter_cube_y_max = 0.2f;
    float filter_cube_z_min = -0.2f;
    float filter_cube_z_max = 0.2f;

    void init() {
        char result[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
        std::string exe_path = std::string(result, (count > 0) ? count : 0);
        std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();

        ray_estimation_policy_path =
            (exe_dir / "./model" / fmt::format("{}.onnx", ray_predictor_model)).string();

        // Load frequency monitoring config
        YAML::Node yaml_conf = YAML::LoadFile("config.yaml");
        if (yaml_conf["frequency_monitoring"]) {
            frequency_ema_alpha = yaml_conf["frequency_monitoring"]["ema_alpha"].as<float>(0.1f);
            frequency_publish_interval =
                yaml_conf["frequency_monitoring"]["publish_interval"].as<int>(50);
        }

        // Load point cloud filtering config
        if (yaml_conf["pointcloud_filtering"]) {
            min_point_distance =
                yaml_conf["pointcloud_filtering"]["min_point_distance"].as<float>(0.1f);
            if (yaml_conf["pointcloud_filtering"]["filter_cube"]) {
                filter_cube_x_min =
                    yaml_conf["pointcloud_filtering"]["filter_cube"]["x_min"].as<float>(-0.3f);
                filter_cube_x_max =
                    yaml_conf["pointcloud_filtering"]["filter_cube"]["x_max"].as<float>(0.3f);
                filter_cube_y_min =
                    yaml_conf["pointcloud_filtering"]["filter_cube"]["y_min"].as<float>(-0.2f);
                filter_cube_y_max =
                    yaml_conf["pointcloud_filtering"]["filter_cube"]["y_max"].as<float>(0.2f);
                filter_cube_z_min =
                    yaml_conf["pointcloud_filtering"]["filter_cube"]["z_min"].as<float>(-0.2f);
                filter_cube_z_max =
                    yaml_conf["pointcloud_filtering"]["filter_cube"]["z_max"].as<float>(0.2f);
            }
        }

        // Load ray filtering config
        if (yaml_conf["ray_filtering"]) {
            ray_filter_threshold = yaml_conf["ray_filtering"]["threshold"].as<float>(0.5f);
            ray_filter_multiplier = yaml_conf["ray_filtering"]["multiplier"].as<float>(0.5f);
            ray_filter_smoothness = yaml_conf["ray_filtering"]["smoothness"].as<float>(0.2f);
        }
    }
};

class RayEstimationNode : public rclcpp::Node {
  public:
    RayEstimationNode(RayEstimationConfig config)
        : Node("ray_estimation_node")
        , config_(config)
        , initialized_(false) {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "RayEstimationNode");

        lidar_subscription_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
            "/livox/lidar", 10,
            std::bind(&RayEstimationNode::pointcloud_callback, this, std::placeholders::_1));

        estimated_rays_publisher_ =
            this->create_publisher<std_msgs::msg::Float32MultiArray>("/estimated_rays", 10);

        frequency_publisher_ = this->create_publisher<std_msgs::msg::Float32>(
            "/ray_estimation_node/control_frequency", 10);
        last_loop_time_ = std::chrono::steady_clock::now();

        // Create visualization publishers only if enabled
        if (config_.enable_visualization) {
            pointcloud_viz_publisher_ =
                this->create_publisher<sensor_msgs::msg::PointCloud2>("/visualized_pointcloud", 10);
            ray_markers_publisher_ =
                this->create_publisher<visualization_msgs::msg::MarkerArray>("/ray_markers", 10);
            grid_pointcloud_publisher_ =
                this->create_publisher<sensor_msgs::msg::PointCloud2>("/grid_pointcloud", 10);
            RCLCPP_INFO(this->get_logger(), "Visualization enabled");
        }

        // Create rotation matrix for lidar pitch angle
        float angle_rad = config_.lidar_pitch_angle_deg * M_PI / 180.0f;
        lidar_rotation_ = AngleAxisf(angle_rad, Vector3f::UnitY());

        // Create offset vector
        lidar_offset_ =
            Vector3f(config_.lidar_offset_x, config_.lidar_offset_y, config_.lidar_offset_z);

        RCLCPP_INFO(
            this->get_logger(),
            "Lidar correction - Rotation: %.1f deg around Y-axis, Offset: (%.3f, %.3f, %.3f) m",
            config_.lidar_pitch_angle_deg, lidar_offset_.x(), lidar_offset_.y(), lidar_offset_.z());

        load_config();
        initialize_controller();

        RCLCPP_INFO(this->get_logger(), "RayEstimationNode initialized");
    }

    void run_pred_loop() {
        if (!initialized_) {
            RCLCPP_ERROR(this->get_logger(), "RayEstimationNode not initialized. Exiting.");
            return;
        }

        rclcpp::Rate rate(config_.control_freq);
        while (rclcpp::ok()) {
            auto loop_start = std::chrono::steady_clock::now();

            run_prediction();

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
        spherical_grid_history_.resize(config_.spherical_grid_history_len);
        for (auto &grid : spherical_grid_history_) {
            grid.setConstant(config_.theta_bins * config_.phi_bins, config_.max_range);
        }

        gravity_history_.resize(config_.imu_history_len);
        for (auto &gravity : gravity_history_) {
            gravity = Vector3f(0.0f, 0.0f, -1.0f);
        }

        angular_velocity_history_.resize(config_.imu_history_len);
        for (auto &ang_vel : angular_velocity_history_) {
            ang_vel = Vector3f(0.0f, 0.0f, 0.0f);
        }
        RCLCPP_INFO(this->get_logger(), "Ray estimation configuration loaded");
    }

    void initialize_controller() {
        try {
            lowstate_subscriber_.reset(
                new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(config_.lowstate_topic));
            lowstate_subscriber_->InitChannel(
                std::bind(&RayEstimationNode::low_state_callback, this, std::placeholders::_1), 10);
            initialize_onnx_runtime();
            initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "Ray estimation controller initialized");
        } catch (const std::exception &e) {
            RCLCPP_ERROR(
                this->get_logger(), "Failed to initialize ray estimation controller: %s", e.what());
        }
    }

    void initialize_onnx_runtime() {
        // Create session options
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Enable TensorRT
        OrtTensorRTProviderOptionsV2 *tensorrt_options = nullptr;
        Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&tensorrt_options));

        std::vector<const char *> option_keys = {
            "trt_fp16_enable",
            "trt_engine_cache_enable",
            "trt_engine_cache_path",
        };
        std::vector<const char *> option_values = {
            "1",                  // Enable FP16
            "1",                  // Enable engine caching
            "./trt_engine_cache", // Specify a path to store the engine cache
        };
        Ort::GetApi().UpdateTensorRTProviderOptions(
            tensorrt_options, option_keys.data(), option_values.data(), option_keys.size());

        session_options.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);
        Ort::GetApi().ReleaseTensorRTProviderOptions(tensorrt_options);

        try {
            RCLCPP_INFO(
                this->get_logger(), "Loading ray estimation policy from %s",
                config_.ray_estimation_policy_path.c_str());
            ray_estimation_session_ = std::make_unique<Ort::Session>(
                env_, config_.ray_estimation_policy_path.c_str(), session_options);

            // Detect IMU input channels from model
            detect_imu_channels();

            RCLCPP_INFO(this->get_logger(), "Ray estimation ONNX model loaded successfully");
            RCLCPP_INFO(this->get_logger(), "Detected IMU channels: %d", config_.imu_channels);
        } catch (const Ort::Exception &e) {
            RCLCPP_ERROR(
                this->get_logger(), "Ray estimation ONNX initialization error: %s", e.what());
            throw;
        }
    }

    void detect_imu_channels() {
        // Get input info to determine the expected shape
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = ray_estimation_session_->GetInputCount();

        for (size_t i = 0; i < num_inputs; i++) {
            auto input_name_ptr = ray_estimation_session_->GetInputNameAllocated(i, allocator);
            std::string input_name(input_name_ptr.get());

            if (input_name == "imu_data" || input_name == "gravity") {
                config_.imu_input_name = input_name;
                auto type_info = ray_estimation_session_->GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                auto shape = tensor_info.GetShape();

                // Shape should be [batch, history_len, channels]
                if (shape.size() >= 3) {
                    config_.imu_channels = static_cast<int>(shape[2]);
                    config_.imu_history_len = static_cast<int>(shape[1]);
                    RCLCPP_INFO(
                        this->get_logger(), "Detected input '%s' with shape [%d, %d, %d]",
                        input_name.c_str(), static_cast<int>(shape[0]), config_.imu_history_len,
                        config_.imu_channels);
                }
            } else if (input_name == "grid") {
                auto type_info = ray_estimation_session_->GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                auto shape = tensor_info.GetShape();

                // Shape should be [batch, history_len, theta_bins, phi_bins]
                if (shape.size() >= 4) {
                    config_.spherical_grid_history_len = static_cast<int>(shape[1]);
                    RCLCPP_INFO(
                        this->get_logger(), "Detected grid input with shape [%d, %d, %d, %d]",
                        static_cast<int>(shape[0]), config_.spherical_grid_history_len,
                        static_cast<int>(shape[2]), static_cast<int>(shape[3]));
                }
            }
        }

        // Prompt user to confirm detected parameters
        RCLCPP_INFO(this->get_logger(), "\n=== Detected Model Parameters ===");
        RCLCPP_INFO(this->get_logger(), "IMU input name: %s", config_.imu_input_name.c_str());
        RCLCPP_INFO(this->get_logger(), "IMU channels: %d", config_.imu_channels);
        RCLCPP_INFO(this->get_logger(), "IMU history length: %d", config_.imu_history_len);
        RCLCPP_INFO(
            this->get_logger(), "Spherical grid history length: %d",
            config_.spherical_grid_history_len);
        RCLCPP_INFO(this->get_logger(), "=================================");
        RCLCPP_INFO(this->get_logger(), "Press ENTER to continue...");

        std::cin.get();
    }

    void low_state_callback(const void *msg) {
        low_state_ = *(unitree_go::msg::dds_::LowState_ *)msg;
        auto quat = Quaternionf(
            low_state_.imu_state().quaternion()[0], low_state_.imu_state().quaternion()[1],
            low_state_.imu_state().quaternion()[2], low_state_.imu_state().quaternion()[3]);
        Vector3f gravity_orientation = get_gravity_orientation_eigen(quat);

        Vector3f angular_velocity(
            low_state_.imu_state().gyroscope()[0], low_state_.imu_state().gyroscope()[1],
            low_state_.imu_state().gyroscope()[2]);

        for (int i = gravity_history_.size() - 1; i > 0; --i) {
            gravity_history_[i] = gravity_history_[i - 1];
        }
        gravity_history_[0] = gravity_orientation;

        for (int i = angular_velocity_history_.size() - 1; i > 0; --i) {
            angular_velocity_history_[i] = angular_velocity_history_[i - 1];
        }
        angular_velocity_history_[0] = angular_velocity;
    }

    void pointcloud_callback(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg) {
        std::vector<Vector3f> points;
        points.reserve(msg->point_num);

        for (int i = 0; i < msg->point_num; ++i) {
            auto &point = msg->points[i];
            if (std::abs(point.x) < 1e-3 && std::abs(point.y) < 1e-3 && std::abs(point.z) < 1e-3) {
                continue;
            }

            Vector3f p(point.x, point.y, point.z);

            // Remove points too close to origin
            if (p.norm() < config_.min_point_distance) {
                continue;
            }

            // Remove points inside configured cube
            if (p.x() >= config_.filter_cube_x_min && p.x() <= config_.filter_cube_x_max &&
                p.y() >= config_.filter_cube_y_min && p.y() <= config_.filter_cube_y_max &&
                p.z() >= config_.filter_cube_z_min && p.z() <= config_.filter_cube_z_max) {
                continue;
            }

            // Apply rotation to compensate for lidar pitch angle
            p = lidar_rotation_ * p;

            // Apply translation to compensate for lidar position offset
            p = p + lidar_offset_;

            const auto p_norm = p.norm();
            if (p_norm > config_.max_range || p_norm < 0.05 ||
                (p_norm < 0.3 && p.x() < -0.05 && std::abs(p.y()) < 0.2 && std::abs(p.z()) < 0.2)) {
                continue;
            }
            points.push_back(p);
        }

        // Store points for visualization
        latest_points_ = points;

        VectorXf spherical_grid = project_points_to_spherical_grid(
            points, config_.phi_range_min, config_.phi_range_max, config_.theta_range_min,
            config_.theta_range_max, config_.phi_res_deg, config_.theta_res_deg, config_.max_range);
        spherical_grid /= 3.0f;

        // Shift history: oldest at index 0, newest at last index
        for (int i = 0; i < spherical_grid_history_.size() - 1; ++i) {
            spherical_grid_history_[i] = spherical_grid_history_[i + 1];
        }
        spherical_grid_history_[spherical_grid_history_.size() - 1] = spherical_grid;
    }

    void run_prediction() {
        RCLCPP_INFO(this->get_logger(), "Running ray estimation prediction");
        try {
            // Visualize the data being used for inference if enabled
            if (config_.enable_visualization) {
                // Visualize point cloud
                if (pointcloud_viz_publisher_ && !latest_points_.empty()) {
                    publish_pointcloud_visualization(latest_points_);
                }

                // Visualize spherical grid
                if (grid_pointcloud_publisher_ && !spherical_grid_history_.empty()) {
                    const auto &current_grid =
                        spherical_grid_history_[spherical_grid_history_.size() - 1];
                    publish_grid_visualization(current_grid);
                }
            }

            std::vector<float> estimated_rays = run_ray_estimation_policy();
            RCLCPP_INFO(this->get_logger(), "Estimated rays");
            auto rays_msg = std::make_unique<std_msgs::msg::Float32MultiArray>();
            rays_msg->data = estimated_rays;
            estimated_rays_publisher_->publish(std::move(rays_msg));
            RCLCPP_INFO(this->get_logger(), "Published estimated rays");

            // Visualize rays if enabled
            if (config_.enable_visualization && ray_markers_publisher_) {
                publish_ray_visualization(estimated_rays);
            }
        } catch (const Ort::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Ray estimation ONNX inference error: %s", e.what());
        }
    }

    std::vector<float> run_ray_estimation_policy() {
        RCLCPP_INFO(this->get_logger(), "Running ray estimation policy inference");
        std::vector<float> spherical_grid_flat;
        int grid_size = config_.theta_bins * config_.phi_bins;
        spherical_grid_flat.reserve(config_.spherical_grid_history_len * grid_size);
        for (const auto &grid : spherical_grid_history_) {
            spherical_grid_flat.insert(
                spherical_grid_flat.end(), grid.data(), grid.data() + grid.size());
        }
        RCLCPP_INFO(this->get_logger(), "Flattened spherical grid");

        std::vector<float> imu_data_flat;
        imu_data_flat.reserve(config_.imu_history_len * config_.imu_channels);

        if (config_.imu_channels == 3) {
            // Old model: only gravity
            for (size_t i = 0; i < config_.imu_history_len; ++i) {
                imu_data_flat.push_back(gravity_history_[i].x());
                imu_data_flat.push_back(gravity_history_[i].y());
                imu_data_flat.push_back(gravity_history_[i].z());
            }
            RCLCPP_INFO(this->get_logger(), "Flattened IMU data (gravity only)");
        } else if (config_.imu_channels == 6) {
            // New model: gravity + angular velocity
            for (size_t i = 0; i < config_.imu_history_len; ++i) {
                imu_data_flat.push_back(gravity_history_[i].x());
                imu_data_flat.push_back(gravity_history_[i].y());
                imu_data_flat.push_back(gravity_history_[i].z());
                imu_data_flat.push_back(angular_velocity_history_[i].x());
                imu_data_flat.push_back(angular_velocity_history_[i].y());
                imu_data_flat.push_back(angular_velocity_history_[i].z());
            }
            RCLCPP_INFO(this->get_logger(), "Flattened IMU data (gravity + angular velocity)");
        } else {
            RCLCPP_ERROR(
                this->get_logger(), "Unsupported IMU channels: %d (expected 3 or 6)",
                config_.imu_channels);
            return std::vector<float>(config_.num_estimated_rays, 0.0f);
        }

        const std::vector<int64_t> spherical_grid_dims = {
            1, config_.spherical_grid_history_len, config_.theta_bins, config_.phi_bins};
        const std::vector<int64_t> imu_data_dims = {
            1, config_.imu_history_len, config_.imu_channels};
        RCLCPP_INFO(this->get_logger(), "Prepared input dimensions");

        const std::array<const char *, 2> input_names = {"grid", config_.imu_input_name.c_str()};
        const std::array<const char *, 1> output_names = {"predicted_rays"};

        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        std::vector<Ort::Value> input_tensors;
        RCLCPP_INFO(this->get_logger(), "Creating input tensors");
        RCLCPP_INFO(this->get_logger(), "grid size: %zu", spherical_grid_flat.size());
        RCLCPP_INFO(this->get_logger(), "imu_data size: %zu", imu_data_flat.size());

        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float *>(spherical_grid_flat.data()),
                spherical_grid_flat.size(), spherical_grid_dims.data(),
                spherical_grid_dims.size()));

        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float *>(imu_data_flat.data()), imu_data_flat.size(),
                imu_data_dims.data(), imu_data_dims.size()));

        RCLCPP_INFO(this->get_logger(), "Running ONNX inference");
        auto output_tensors = ray_estimation_session_->Run(
            Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
            input_tensors.size(), output_names.data(), output_names.size());

        RCLCPP_INFO(this->get_logger(), "ONNX inference completed");
        float *rays_ptr = output_tensors[0].GetTensorMutableData<float>();
        std::vector<float> estimated_rays(rays_ptr, rays_ptr + config_.num_estimated_rays);

        // Apply smooth ray filtering
        for (auto &ray : estimated_rays) {
            float blend_factor;
            if (ray < config_.ray_filter_threshold - config_.ray_filter_smoothness) {
                // Below transition region: full multiplier applied
                blend_factor = 1.0f;
            } else if (ray > config_.ray_filter_threshold + config_.ray_filter_smoothness) {
                // Above transition region: no multiplier applied
                blend_factor = 0.0f;
            } else {
                // Inside transition region: smooth interpolation using cosine
                float normalized =
                    (ray - (config_.ray_filter_threshold - config_.ray_filter_smoothness)) /
                    (2.0f * config_.ray_filter_smoothness);
                blend_factor = 0.5f * (1.0f + std::cos(normalized * M_PI));
            }
            // Apply weighted combination: original ray when blend_factor=0, multiplied ray when
            // blend_factor=1
            ray =
                ray * (1.0f - blend_factor) + (ray * config_.ray_filter_multiplier) * blend_factor;
        }

        RCLCPP_INFO(this->get_logger(), "Estimation finished");
        return estimated_rays;
    }

    void publish_pointcloud_visualization(const std::vector<Vector3f> &points) {
        auto msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
        msg->header.stamp = this->get_clock()->now();
        msg->header.frame_id = "base_link";

        msg->height = 1;
        msg->width = points.size();
        msg->is_dense = true;
        msg->is_bigendian = false;

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
        msg->row_step = msg->point_step * points.size();
        msg->data.resize(msg->row_step);

        for (size_t i = 0; i < points.size(); ++i) {
            memcpy(&msg->data[i * 12 + 0], &points[i].x(), sizeof(float));
            memcpy(&msg->data[i * 12 + 4], &points[i].y(), sizeof(float));
            memcpy(&msg->data[i * 12 + 8], &points[i].z(), sizeof(float));
        }

        pointcloud_viz_publisher_->publish(std::move(msg));
    }

    void publish_ray_visualization(const std::vector<float> &rays) {
        auto marker_array = std::make_unique<visualization_msgs::msg::MarkerArray>();
        int num_rays = rays.size();
        std::vector<float> angles(num_rays);
        for (int i = 0; i < num_rays; ++i) {
            angles[i] = -M_PI + (2.0f * M_PI * i) / num_rays;
        }

        for (int i = 0; i < num_rays; ++i) {
            // Sphere marker at endpoint
            visualization_msgs::msg::Marker sphere;
            sphere.header.frame_id = "base_link";
            sphere.header.stamp = this->get_clock()->now();
            sphere.type = visualization_msgs::msg::Marker::SPHERE;
            sphere.action = visualization_msgs::msg::Marker::ADD;
            sphere.id = i;
            sphere.pose.position.x = rays[i] * std::cos(angles[i]);
            sphere.pose.position.y = rays[i] * std::sin(angles[i]);
            sphere.pose.position.z = 0.0;
            sphere.pose.orientation.w = 1.0;
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.05;
            sphere.color.r = 1.0;
            sphere.color.g = 0.0;
            sphere.color.b = 0.0;
            sphere.color.a = 1.0;
            marker_array->markers.push_back(sphere);

            // Line marker
            visualization_msgs::msg::Marker line;
            line.header.frame_id = "base_link";
            line.header.stamp = this->get_clock()->now();
            line.type = visualization_msgs::msg::Marker::LINE_LIST;
            line.action = visualization_msgs::msg::Marker::ADD;
            line.id = i + num_rays;
            geometry_msgs::msg::Point p1, p2;
            p1.x = p1.y = p1.z = 0.0;
            p2.x = rays[i] * std::cos(angles[i]);
            p2.y = rays[i] * std::sin(angles[i]);
            p2.z = 0.0;
            line.points = {p1, p2};
            line.scale.x = 0.01;
            line.color.r = 1.0;
            line.color.g = 0.2;
            line.color.b = 0.2;
            line.color.a = 0.6;
            marker_array->markers.push_back(line);
        }

        ray_markers_publisher_->publish(std::move(marker_array));
    }

    void publish_grid_visualization(const VectorXf &grid) {
        auto msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
        msg->header.stamp = this->get_clock()->now();
        msg->header.frame_id = "base_link";

        float phi_min = config_.phi_range_min * M_PI / 180.0f;
        float phi_max = config_.phi_range_max * M_PI / 180.0f;
        float theta_min = config_.theta_range_min * M_PI / 180.0f;
        float theta_max = config_.theta_range_max * M_PI / 180.0f;

        std::vector<float> points_data;
        points_data.reserve(config_.theta_bins * config_.phi_bins * 4);

        for (int theta_idx = 0; theta_idx < config_.theta_bins; ++theta_idx) {
            for (int phi_idx = 0; phi_idx < config_.phi_bins; ++phi_idx) {
                float phi = phi_min + (phi_idx + 0.5f) * (phi_max - phi_min) / config_.phi_bins;
                float theta =
                    theta_min + (theta_idx + 0.5f) * (theta_max - theta_min) / config_.theta_bins;
                float distance = grid[theta_idx * config_.phi_bins + phi_idx] * config_.max_range;

                float x = distance * std::cos(theta) * std::cos(phi);
                float y = distance * std::cos(theta) * std::sin(phi);
                float z = distance * std::sin(theta);

                float normalized_dist = distance / config_.max_range;
                uint8_t r = static_cast<uint8_t>(normalized_dist * 255);
                uint8_t g = static_cast<uint8_t>(0.3f * 255);
                uint8_t b = static_cast<uint8_t>((1.0f - normalized_dist) * 255);
                uint32_t rgb = (r << 16) | (g << 8) | b;
                float rgb_float;
                memcpy(&rgb_float, &rgb, sizeof(float));

                points_data.push_back(x);
                points_data.push_back(y);
                points_data.push_back(z);
                points_data.push_back(rgb_float);
            }
        }

        msg->height = 1;
        msg->width = config_.theta_bins * config_.phi_bins;
        msg->is_dense = true;
        msg->is_bigendian = false;

        sensor_msgs::msg::PointField fx, fy, fz, frgb;
        fx.name = "x";
        fx.offset = 0;
        fx.datatype = sensor_msgs::msg::PointField::FLOAT32;
        fx.count = 1;
        fy.name = "y";
        fy.offset = 4;
        fy.datatype = sensor_msgs::msg::PointField::FLOAT32;
        fy.count = 1;
        fz.name = "z";
        fz.offset = 8;
        fz.datatype = sensor_msgs::msg::PointField::FLOAT32;
        fz.count = 1;
        frgb.name = "rgb";
        frgb.offset = 12;
        frgb.datatype = sensor_msgs::msg::PointField::FLOAT32;
        frgb.count = 1;
        msg->fields = {fx, fy, fz, frgb};

        msg->point_step = 16;
        msg->row_step = msg->point_step * msg->width;
        msg->data.resize(msg->row_step);
        memcpy(msg->data.data(), points_data.data(), points_data.size() * sizeof(float));

        grid_pointcloud_publisher_->publish(std::move(msg));
    }

    RayEstimationConfig config_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr lidar_subscription_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr estimated_rays_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr frequency_publisher_;

    // Visualization publishers (only created if enabled)
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_viz_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr ray_markers_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grid_pointcloud_publisher_;

    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber_;
    unitree_go::msg::dds_::LowState_ low_state_;

    Ort::Env env_;
    std::unique_ptr<Ort::Session> ray_estimation_session_;

    std::vector<VectorXf> spherical_grid_history_;
    std::vector<Vector3f> gravity_history_;
    std::vector<Vector3f> angular_velocity_history_;

    // Store latest point cloud for visualization
    std::vector<Vector3f> latest_points_;

    // Rotation matrix for lidar pitch compensation
    AngleAxisf lidar_rotation_;

    // Translation vector for lidar position offset
    Vector3f lidar_offset_;

    bool initialized_;

    // Frequency monitoring
    std::chrono::steady_clock::time_point last_loop_time_;
    float estimated_frequency_ = 0.0f;
    int frequency_publish_counter_ = 0;
};

int main(int argc, char *argv[]) {
    cxxopts::Options options("RayEstimationNode", "Unitree Go2 Ray Estimation");
    options.add_options()(
        "n,net", "network interface", cxxopts::value<std::string>()->default_value("lo"))(
        "p,policy", "ray estimation policy", cxxopts::value<std::string>()->default_value(""))(
        "v,viz", "enable visualization",
        cxxopts::value<bool>()->default_value("false"))("h,help", "Print usage");
    auto args = options.parse(argc, argv);

    auto net_interface = args["net"].as<std::string>();
    int net_idx = net_interface == "lo" ? 1 : 0;
    ChannelFactory::Instance()->Init(net_idx, net_interface);

    rclcpp::init(argc, argv);

    RayEstimationConfig config;
    config.enable_visualization = args["viz"].as<bool>();

    YAML::Node yaml_conf = YAML::LoadFile("config.yaml");

    // Load lidar pitch angle
    if (yaml_conf["lidar_pitch_angle_deg"]) {
        config.lidar_pitch_angle_deg = yaml_conf["lidar_pitch_angle_deg"].as<float>();
    }

    // Load lidar offset
    if (yaml_conf["lidar_offset_x"]) {
        config.lidar_offset_x = yaml_conf["lidar_offset_x"].as<float>();
    }
    if (yaml_conf["lidar_offset_y"]) {
        config.lidar_offset_y = yaml_conf["lidar_offset_y"].as<float>();
    }
    if (yaml_conf["lidar_offset_z"]) {
        config.lidar_offset_z = yaml_conf["lidar_offset_z"].as<float>();
    }

    if (yaml_conf["spherical_config"]) {
        config.phi_range_min = yaml_conf["spherical_config"]["phi_range_min"].as<float>();
        config.phi_range_max = yaml_conf["spherical_config"]["phi_range_max"].as<float>();
        config.theta_range_min = yaml_conf["spherical_config"]["theta_range_min"].as<float>();
        config.theta_range_max = yaml_conf["spherical_config"]["theta_range_max"].as<float>();
        config.phi_res_deg = yaml_conf["spherical_config"]["phi_res_deg"].as<int>();
        config.theta_res_deg = yaml_conf["spherical_config"]["theta_res_deg"].as<int>();
        config.max_range = yaml_conf["spherical_config"]["max_range"].as<float>();
        config.phi_bins = (config.phi_range_max - config.phi_range_min) / config.phi_res_deg;
        config.theta_bins =
            (config.theta_range_max - config.theta_range_min) / config.theta_res_deg;
    }
    if (!args["policy"].as<std::string>().empty()) {
        config.ray_predictor_model = args["policy"].as<std::string>();
    }

    config.init();
    auto node = std::make_shared<RayEstimationNode>(config);

    std::thread spin_thread([&]() {
        rclcpp::executors::MultiThreadedExecutor executor;
        executor.add_node(node);
        executor.spin();
        rclcpp::shutdown();
    });
    spin_thread.detach();

    node->run_pred_loop();

    return 0;
}
