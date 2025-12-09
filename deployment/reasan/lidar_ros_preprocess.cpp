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
#include "livox_ros_driver2/msg/custom_msg.hpp"
#include "livox_ros_driver2/msg/custom_point.hpp"
#include "mocap4r2_msgs/msg/rigid_bodies.hpp"
#include "nav_msgs/msg/odometry.hpp" // Added for odometry messages
#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

// Eigen headers
#include <Eigen/Core>
#include <Eigen/Geometry>

// Open3D headers
#include <open3d/Open3D.h>

// argument parser
#include "cxxopts.hpp"
#include "yaml-cpp/yaml.h"

// ONNX Runtime headers
#include "onnxruntime_cxx_api.h"

#include "advanced_gamepad.hpp"

// Unitree SDK headers
#include "unitree/common/json/json.hpp"
#include "unitree/common/time/time_tool.hpp"
#include "unitree/idl/go2/LowCmd_.hpp"
#include "unitree/idl/go2/LowState_.hpp"
#include "unitree/idl/go2/SportModeState_.hpp"
#include "unitree/robot/b2/motion_switcher/motion_switcher_api.hpp"
#include "unitree/robot/b2/motion_switcher/motion_switcher_client.hpp"
#include "unitree/robot/channel/channel_factory.hpp"
#include "unitree/robot/channel/channel_publisher.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"
#include "unitree/robot/go2/sport/sport_client.hpp"
#include "unitree/idl/go2/WirelessController_.hpp"

using namespace std::chrono_literals;
using namespace unitree::common;
using namespace unitree::robot::b2;
using namespace unitree::robot;
using namespace unitree;
using namespace Eigen;

std::atomic<bool> g_stop_control_signal(false);
std::atomic<bool> g_emergency_stop_mode(false);

void interrupte_handler(int signal_num)
{
    static bool first_interrupt = true;
    if (first_interrupt)
    {
        g_emergency_stop_mode.store(true, std::memory_order_relaxed);
        first_interrupt = false;
        fmt::print("\n=== EMERGENCY STOP ACTIVATED ===\n");
        fmt::print("Switching to locomotion policy with zero velocity commands\n");
        fmt::print("Robot will attempt to stand in place\n");
        fmt::print("Press Ctrl+C again to fully stop the program\n");
    }
    else
    {
        g_stop_control_signal.store(true, std::memory_order_relaxed);
        fmt::print("Second Ctrl+C detected - stopping control...\n");
    }
}

template <typename Iterable>
auto enumerate(Iterable&& iterable)
{
    using Iterator = decltype(std::begin(std::declval<Iterable>()));
    using T = decltype(*std::declval<Iterator>());

    struct Enumerated
    {
        std::size_t index;
        T element;
    };

    struct Enumerator
    {
        Iterator iterator;
        std::size_t index;

        auto operator!=(const Enumerator& other) const { return iterator != other.iterator; }

        auto&operator++()
        {
            ++iterator;
            ++index;
            return *this;
        }

        auto operator*() const { return Enumerated{index, *iterator}; }
    };

    struct Wrapper
    {
        Iterable& iterable;

        [[nodiscard]] auto begin() const { return Enumerator{std::begin(iterable), 0U}; }

        [[nodiscard]] auto end() const { return Enumerator{std::end(iterable), 0U}; }
    };

    return Wrapper{std::forward<Iterable>(iterable)};
}

// Ray source enum
enum class RaySource
{
    LIDAR = 1,
};

/**
 * @brief Convert Cartesian coordinates to spherical coordinates (r, theta, phi)
 * @param point Vector3d containing Cartesian coordinates
 * @return Vector3d representing (r, theta, phi)
 */
Vector3d cartesian_to_spherical(const Vector3f& point)
{
    float r = point.norm();
    // azimuthal angle: std::atan2, -pi~pi, 0 at x=1,y=0, counter-clockwise
    float theta = std::atan2(point.y(), point.x());
    // polar angle with epsilon to prevent division by zero
    // std::acos: -1.0~1.0 -> pi~0.0
    float phi = std::acos(point.z() / (r + 1e-10));
    return Vector3d(r, theta, phi);
}

/**
 * @brief Get gravity orientation from quaternion (equivalent to Python's get_gravity_orientation)
 * @param quat Quaternion representing orientation
 * @return Vector3f representing gravity orientation
 */
Vector3f get_gravity_orientation_eigen(const Quaternionf& quat)
{
    Vector3f gravity_orientation;
    gravity_orientation.x() = 2.0f * (-quat.z() * quat.x() + quat.w() * quat.y());
    gravity_orientation.y() = -2.0f * (quat.z() * quat.y() + quat.w() * quat.x());
    gravity_orientation.z() = 1.0f - 2.0f * (quat.w() * quat.w() + quat.z() * quat.z());

    return gravity_orientation;
}

uint32_t crc32_core(uint32_t* ptr, uint32_t len)
{
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; i++)
    {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
            {
                CRC32 <<= 1;
            }

            if (data & xbit)
                CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }

    return CRC32;
}

void pad_open3d_pointcloud(open3d::geometry::PointCloud& pcd, size_t n)
{
    size_t current_size = pcd.points_.size();
    if (current_size < n)
    {
        size_t num_to_add = n - current_size;
        Eigen::Vector3d zero_point(0.0, 0.0, 0.0);
        for (size_t i = 0; i < num_to_add; ++i)
        {
            pcd.points_.push_back(zero_point);
        }
        if (pcd.HasColors())
        {
            Eigen::Vector3d black_color(0.0, 0.0, 0.0);
            for (size_t i = 0; i < num_to_add; ++i)
            {
                pcd.colors_.push_back(black_color);
            }
        }
    }
}

// Configuration for Go2 robot (similar to Go2Config in Python)
struct Go2Config
{
    int control_freq = 50;
    float control_dt = 0.02f;

    std::string lowcmd_topic = "rt/lowcmd";
    std::string lowstate_topic = "rt/lowstate";

    std::string loco_policy_path;
    std::string high_policy_path;

    std::string loco_policy_name = "loco_policy";
    std::string high_policy_name = "high_policy";

    float kp = 32.0f;
    float kd = 1.0f;

    bool loco_only = false;

    bool manual_rotation = false;

    // Joint index mapping
    std::array<int, 12> joint2motor_idx = {3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8};

    // Default joint angles
    VectorXf default_angles;
    VectorXf crouch_angles;

    // ordered joint names in simulation (model input)
    std::vector<std::string> sim_joint_names = {
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    };

    // lowstate.motor_state/lowcmd.motor_cmd
    // Joint 0 ：Hip， body joint
    // Joint 1 ：Thigh，thigh joint
    // Joint 2 ：Calf，calf joint
    // FR_0 -> 0 , FR_1 -> 1  , FR_2 -> 2
    // FL_0 -> 3 , FL_1 -> 4  , FL_2 -> 5
    // RR_0 -> 6 , RR_1 -> 7  , RR_2 -> 8
    // RL_0 -> 9 , RL_1 -> 10 , RL_2 -> 11
    std::vector<std::string> motor_joint_names = {
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FL_hip_joint",
        "FL_thigh_joint", "FL_calf_joint", "RR_hip_joint", "RR_thigh_joint",
        "RR_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    };

    std::unordered_map<std::string, float> default_joint_angles = {
        {"FL_hip_joint", 0.1}, {"RL_hip_joint", 0.1}, {"FR_hip_joint", -0.1},
        {"RR_hip_joint", -0.1}, {"FL_thigh_joint", 0.8}, {"RL_thigh_joint", 1.0},
        {"FR_thigh_joint", 0.8}, {"RR_thigh_joint", 1.0}, {"FL_calf_joint", -1.5},
        {"RL_calf_joint", -1.5}, {"FR_calf_joint", -1.5}, {"RR_calf_joint", -1.5},
    };
    std::unordered_map<std::string, float> crouch_joint_angles = {
        {"FL_hip_joint", 0.125}, {"FL_thigh_joint", 1.23}, {"FL_calf_joint", -2.70},
        {"FR_hip_joint", -0.125}, {"FR_thigh_joint", 1.23}, {"FR_calf_joint", -2.70},
        {"RL_hip_joint", 0.47}, {"RL_thigh_joint", 1.25}, {"RL_calf_joint", -2.72},
        {"RR_hip_joint", -0.47}, {"RR_thigh_joint", 1.25}, {"RR_calf_joint", -2.72},
    };

    // Joint limits
    std::vector<std::pair<float, float>> joint_limits = {
        {-1.0472, 1.0472}, {-1.0472, 1.0472}, {-1.0472, 1.0472}, {-1.0472, 1.0472},
        {-1.5708, 3.4907}, {-1.5708, 3.4907}, {-0.5236, 4.5379}, {-0.5236, 4.5379},
        {-2.7227, -0.8378}, {-2.7227, -0.8378}, {-2.7227, -0.8378}, {-2.7227, -0.8378},
    };

    // Observation scales
    float lin_vel_scale = 2.0f;
    float ang_vel_scale = 0.25f;
    float dof_pos_scale = 1.0f;
    float dof_vel_scale = 0.05f;
    std::array<float, 3> cmd_scale = {lin_vel_scale, lin_vel_scale, ang_vel_scale};
    float obs_clip = 100.0f;

    // Action scales
    float action_scale = 0.25f;
    float action_clip = 100.0f;

    // Network dimensions
    int num_loco_actions = 12;
    int num_high_actions = 3;
    int num_high_cmd = 3;
    int num_proprio = 45;

    // PointNet++ config
    float voxel_size = 0.1f;
    int outlier_nb_neighbors = 20;
    double outlier_std_ratio = 1.0;
    int num_total_points = 400;
    int num_level1_centroids = 64;
    float level1_radius = 0.4f;
    int level1_max_neighbors = 16;
    int num_level2_centroids = 16;
    float level2_radius = 0.8f;
    int level2_max_neighbors = 8;
    int num_ray_obs = (64 * 3) + (64 * 16 * 3) + (16 * 3) + (16 * 8 * 3) + (16 * 8); // 3824

    // Ray source
    RaySource ray_obs_source = RaySource::LIDAR;

    bool use_sim = false;

    bool use_avoidance_policy = false;

    float high_lin_vel_max = 1.0f;
    float high_lin_vel_lateral_max = 0.5f;
    float high_ang_vel_max = 1.0f;

    std::vector<Vector3f> goals = {{2.5f, 0.0f, 0.0f}, {-2.5f, 0.0f, 0.0f}};

    float bound_x_min = -10.0f;
    float bound_x_max = 10.0f;
    float bound_y_min = -10.0f;
    float bound_y_max = 10.0f;

    void init()
    {
        fmt::print("joint2motor_idx: ");
        for (auto&& [i, name] : enumerate(sim_joint_names))
        {
            auto index = std::find(motor_joint_names.begin(), motor_joint_names.end(), name) -
                motor_joint_names.begin();
            joint2motor_idx.at(i) = index;
            fmt::print("{} ", index);
        }
        fmt::print("\nshould be:       3 0 9 6 4 1 10 7 5 2 11 8\n");

        default_angles.resize(num_loco_actions);
        crouch_angles.resize(num_loco_actions);
        for (int i = 0; i < num_loco_actions; ++i)
        {
            default_angles(i) = default_joint_angles.at(sim_joint_names.at(i));
            crouch_angles(i) = crouch_joint_angles.at(sim_joint_names.at(i));
        }

        // Get the executable directory
        char result[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
        std::string exe_path = std::string(result, (count > 0) ? count : 0);
        std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();

        // Set model paths
        loco_policy_path =
            (exe_dir / "./model" / fmt::format("{}.onnx", loco_policy_name)).string();
        high_policy_path =
            (exe_dir / "./model" / fmt::format("{}.onnx", high_policy_name)).string();

        // Modify joint limits to reduce range
        for (auto& limit : joint_limits)
        {
            float range = limit.second - limit.first;
            limit.first += 0.05f * range;
            limit.second -= 0.05f * range;
            if (limit.second <= limit.first)
            {
                throw std::runtime_error("Joint limits are invalid after modification.");
            }
        }
    }
};

/**
 * @brief Go2Control node combining LiDAR processing with robot control
 */
class Go2Control : public rclcpp::Node
{
public:
    Go2Control(Go2Config config, bool viz_only = false)
        : Node("go2_control")
          , config_(config)
          , viz_only_(viz_only)
          , running_(false)
          , initialized_(false)
          , slow_start_counter_(0)
          , is_rotating_to_goal_(false)
    {
        // Initialize ONNX environment once
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Go2Control");

        // Create LiDAR subscription
        lidar_subscription_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
            "/livox/lidar", 10,
            std::bind(&Go2Control::pointcloud_callback, this, std::placeholders::_1));

        // Create marker publisher for visualization
        ray_marker_publisher_ =
            this->create_publisher<visualization_msgs::msg::MarkerArray>("/ray_visualization", 10);

        // Create raw point cloud publisher for visualization-only mode
        if (viz_only_)
        {
            raw_pointcloud_publisher_ =
                this->create_publisher<visualization_msgs::msg::MarkerArray>("/raw_pointcloud", 10);
        }

        // Create ROS2 odometry subscription to replace unitree highstate subscriber
        odom_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/go2_odom", 10, std::bind(&Go2Control::odom_callback, this, std::placeholders::_1));

        // Load configuration
        load_config();

        // Initialize controller
        initialize_controller();

        RCLCPP_INFO(
            this->get_logger(), "Go2Control initialized (viz_only: %s)",
            viz_only_ ? "true" : "false");
    }

    ~Go2Control()
    {
        // Ensure we send zero commands when shutting down
        fmt::print("shutting down...\n");
        if (initialized_ && !viz_only_)
        {
            create_zero_cmd();
            send_cmd();
            RCLCPP_INFO(this->get_logger(), "Robot control shut down");
        }
    }

    void run_control_loop()
    {
        fmt::print("attempting to start control loop...\n");

        if (!initialized_)
        {
            fmt::print("quit control loop.\n");
            return;
        }

        if (!viz_only_)
        {
            start_control_sequence();
        }

        rclcpp::Rate rate(config_.control_freq);
        while (true)
        {
            auto loop_start = std::chrono::steady_clock::now();

            if (g_stop_control_signal.load(std::memory_order_relaxed))
            {
                fmt::print("stopping control...\n");
                break;
            }
            run_controller();

            auto loop_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> loop_duration = loop_end - loop_start;
            static int log_counter = 0;
            if (log_counter++ % 50 == 0)
            {
                // Log every second at 50Hz
                fmt::print("Control loop iteration time: {:.10f}s\n", loop_duration.count());
            }

            rate.sleep();
        }

        if (!viz_only_)
        {
            stop_control_sequence();
        }
    }

private:
    void load_config()
    {
        // Initialize controller state variables
        qj_.resize(config_.num_loco_actions);
        qj_.setZero();
        dqj_.resize(config_.num_loco_actions);
        dqj_.setZero();
        loco_action_.resize(config_.num_loco_actions, 0.0f);
        high_action_.resize(config_.num_high_actions, 0.0f);
        target_dof_pos_ = config_.default_angles;
        cmd_ = {0.0f, 0.0f, 0.0f};
        cmd_base_height_ = 0.34f;

        // initialize pointnet++ hierarchy (centroids and grouped points)
        l1_centroids_.resize(config_.num_level1_centroids * 3);
        l1_grouped_points_.resize(config_.num_level1_centroids * config_.level1_max_neighbors * 3);
        l2_centroids_.resize(config_.num_level2_centroids * 3);
        l2_grouped_points_.resize(config_.num_level2_centroids * config_.level2_max_neighbors * 3);
        l2_grouped_indices_.resize(config_.num_level2_centroids * config_.level2_max_neighbors);
        std::fill(l1_centroids_.begin(), l1_centroids_.end(), 0.0f);
        std::fill(l1_grouped_points_.begin(), l1_grouped_points_.end(), 0.0f);
        std::fill(l2_centroids_.begin(), l2_centroids_.end(), 0.0f);
        std::fill(l2_grouped_points_.begin(), l2_grouped_points_.end(), 0.0f);
        std::fill(l2_grouped_indices_.begin(), l2_grouped_indices_.end(), 0);

        proprio_obs_.resize(config_.num_proprio);
        proprio_obs_.setZero();

        init_pos_set_ = false;
        init_pos_ = {0.0f, 0.0f, 0.0f};
        goal_pos_ = {0.0f, 0.0f, 0.0f};
        curr_pos_ = {0.0f, 0.0f, 0.0f};
        curr_quat_ = {1.0f, 0.0f, 0.0f, 0.0f};

        // Initialize goals
        goals_.clear();
        for (const auto& goal : config_.goals)
        {
            goals_.emplace_back(goal);
        }
        goal_counter_ = 0;

        RCLCPP_INFO(this->get_logger(), "Configuration loaded");
    }

    void initialize_controller()
    {
        fmt::print("initializing controller...\n");
        try
        {
            // Initialize command publisher and state subscribers
            lowcmd_publisher_ = std::make_unique<ChannelPublisher<unitree_go::msg::dds_::LowCmd_>>(
                config_.lowcmd_topic);
            lowcmd_publisher_->InitChannel();

            lowstate_subscriber_ =
                std::make_unique<ChannelSubscriber<unitree_go::msg::dds_::LowState_>>(
                    config_.lowstate_topic);
            lowstate_subscriber_->InitChannel(
                std::bind(&Go2Control::low_state_callback, this, std::placeholders::_1), 10);

            // subscribe to wireless controller
            wireless_controller_.reset(
                new ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>(
                    "rt/wirelesscontroller"));
            wireless_controller_->InitChannel(
                std::bind(&Go2Control::wireless_controller_callback, this, std::placeholders::_1),
                10);

            // Initialize ONNX models
            initialize_onnx_runtime();

            // Initialize the command
            init_cmd_go();
            // Wait for initial state data
            if (!viz_only_)
            {
                wait_for_low_state();
            }

            // Initialize robot clients
            if (!config_.use_sim && !viz_only_)
            {
                sport_client_ = std::make_unique<unitree::robot::go2::SportClient>();
                motion_switcher_client_ = std::make_unique<MotionSwitcherClient>();

                sport_client_->SetTimeout(5.0f);
                sport_client_->Init();
                sport_client_->StandDown();
                std::this_thread::sleep_for(2s);

                motion_switcher_client_->SetTimeout(5.0f);
                motion_switcher_client_->Init();

                fmt::print("switching to release mode.\n");
                if (motion_switcher_client_->ReleaseMode() == 0)
                {
                    fmt::print("switch to release mode success.\n");
                }
                else
                {
                    throw std::runtime_error("switch to release mode failed.");
                }
                std::this_thread::sleep_for(5s);
            }

            initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "Controller initialized");
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize controller: %s", e.what());
        }
        fmt::print("controller initialized.\n");
    }

    void initialize_onnx_runtime()
    {
        fmt::print("initializing ONNX Runtime...\n");

        // Create session options
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Enable TensorRT
        OrtTensorRTProviderOptionsV2* tensorrt_options = nullptr;
        Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&tensorrt_options));

        /// NOTE: will fp16 affect policy performance?
        std::vector<const char*> option_keys = {
            "trt_fp16_enable",
            "trt_engine_cache_enable",
            "trt_engine_cache_path",
        };
        std::vector<const char*> option_values = {
            "1", // Enable FP16
            "1", // Enable engine caching
            "/tmp/trt_engine_cache", // Specify a path to store the engine cache
        };
        Ort::GetApi().UpdateTensorRTProviderOptions(
            tensorrt_options, option_keys.data(), option_values.data(), option_keys.size());

        session_options.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);
        Ort::GetApi().ReleaseTensorRTProviderOptions(tensorrt_options);

        try
        {
            // Create ONNX sessions
            if (config_.loco_only)
            {
                RCLCPP_INFO(
                    this->get_logger(), "Locomotion-only mode: Loading only loco policy from %s",
                    config_.loco_policy_path.c_str());

                loco_policy_session_ = std::make_unique<Ort::Session>(
                    env_, config_.loco_policy_path.c_str(), session_options);

                // Initialize LSTM hidden and cell states for loco policy only
                loco_policy_h_.resize(256);
                loco_policy_h_.setZero();

                loco_policy_c_.resize(256);
                loco_policy_c_.setZero();
            }
            else
            {
                RCLCPP_INFO(
                    this->get_logger(), "Loading ONNX models from %s and %s",
                    config_.loco_policy_path.c_str(), config_.high_policy_path.c_str());

                fmt::print("making loco policy session...\n");
                loco_policy_session_ = std::make_unique<Ort::Session>(
                    env_, config_.loco_policy_path.c_str(), session_options);

                fmt::print("making high policy session...\n");
                high_policy_session_ = std::make_unique<Ort::Session>(
                    env_, config_.high_policy_path.c_str(), session_options);

                fmt::print("session created.\n");

                // Initialize LSTM hidden and cell states for both policies
                loco_policy_h_.resize(256);
                loco_policy_h_.setZero();

                loco_policy_c_.resize(256);
                loco_policy_c_.setZero();

                high_policy_h_.resize(256);
                high_policy_h_.setZero();

                high_policy_c_.resize(256);
                high_policy_c_.setZero();
            }

            RCLCPP_INFO(this->get_logger(), "ONNX models loaded successfully");
        }
        catch (const Ort::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "ONNX initialization error: %s", e.what());
            throw;
        }
    }

    void low_state_callback(const void* msg)
    {
        low_state_ = *(unitree_go::msg::dds_::LowState_*)msg;
    }

    void wireless_controller_callback(const void* msg)
    {
        wireless_controller_msg_ = *(unitree_go::msg::dds_::WirelessController_*)msg;
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        // Replace high_state_callback with functionality to handle odometry messages
        if (config_.loco_only)
        {
            // For loco-only mode, extract command from Odometry message twist
            cmd_(0) = msg->twist.twist.linear.x * 2.0; // forward velocity
            cmd_(1) = msg->twist.twist.linear.y * 2.0; // lateral velocity
            cmd_(2) = msg->twist.twist.angular.z * 0.25; // yaw speed as angular command
            // Use z position as base height, clamped to appropriate range
            cmd_base_height_ = std::clamp(msg->pose.pose.position.z, 0.22, 0.37);

            // Log commands periodically
            static int log_counter = 0;
            if (log_counter++ % 100 == 0)
            {
                RCLCPP_INFO(
                    this->get_logger(), "Loco-only commands: [%.2f, %.2f, %.2f]", cmd_[0], cmd_[1],
                    cmd_[2]);
            }
        }
        else
        {
            if (!init_pos_set_)
            {
                init_pos_set_ = true;
                init_pos_(0) = msg->pose.pose.position.x;
                init_pos_(1) = msg->pose.pose.position.y;
                init_pos_(2) = msg->pose.pose.position.z;
                goal_pos_ = goals_[goal_counter_];
            }

            auto px = msg->pose.pose.position.x;
            auto py = msg->pose.pose.position.y;
            auto pz = msg->pose.pose.position.z;
            if (std::isnan(px) || std::isnan(py) || std::isnan(pz))
            {
                px = curr_pos_(0);
                py = curr_pos_(1);
                pz = curr_pos_(2);
            }
            curr_pos_(0) = px;
            curr_pos_(1) = py;
            curr_pos_(2) = pz;

            // quaternion conversion: wxyz (ROS uses xyzw order, so we need to reorder)
            auto w = msg->pose.pose.orientation.w;
            auto x = msg->pose.pose.orientation.x;
            auto y = msg->pose.pose.orientation.y;
            auto z = msg->pose.pose.orientation.z;
            if (std::isnan(w) || std::isnan(x) || std::isnan(y) || std::isnan(z))
            {
                w = curr_quat_.w();
                x = curr_quat_.x();
                y = curr_quat_.y();
                z = curr_quat_.z();
            }
            curr_quat_.w() = w;
            curr_quat_.x() = x;
            curr_quat_.y() = y;
            curr_quat_.z() = z;
        }
    }

    void wait_for_low_state()
    {
        if (config_.use_sim)
        {
            fmt::print("Simulation mode, skipping wait for low state.\n");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Waiting for robot state...");
        auto start = std::chrono::steady_clock::now();
        while (low_state_.tick() == 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - start)
                .count();
            if (elapsed > 5)
            {
                RCLCPP_WARN(this->get_logger(), "Timeout waiting for robot state");
                break;
            }
        }
        RCLCPP_INFO(this->get_logger(), "Successfully connected to the robot");
    }

    void init_cmd_go()
    {
        low_cmd_.head()[0] = 0xFE;
        low_cmd_.head()[1] = 0xEF;
        low_cmd_.level_flag() = 0xFF;
        low_cmd_.gpio() = 0;

        float PosStopF = 2.146e9f;
        float VelStopF = 16000.0f;

        for (int i = 0; i < low_cmd_.motor_cmd().size(); i++)
        {
            low_cmd_.motor_cmd()[i].mode() = 0x0A;
            low_cmd_.motor_cmd()[i].q() = PosStopF;
            low_cmd_.motor_cmd()[i].dq() = VelStopF;
            low_cmd_.motor_cmd()[i].kp() = 0;
            low_cmd_.motor_cmd()[i].kd() = 0;
            low_cmd_.motor_cmd()[i].tau() = 0;
        }
    }

    void create_zero_cmd()
    {
        for (int i = 0; i < 12; i++)
        {
            low_cmd_.motor_cmd()[i].q() = 0;
            low_cmd_.motor_cmd()[i].dq() = 0;
            low_cmd_.motor_cmd()[i].kp() = 0;
            low_cmd_.motor_cmd()[i].kd() = 0;
            low_cmd_.motor_cmd()[i].tau() = 0;
        }
    }

    void create_damping_cmd()
    {
        for (int i = 0; i < 12; i++)
        {
            low_cmd_.motor_cmd()[i].q() = 0;
            low_cmd_.motor_cmd()[i].dq() = 0;
            low_cmd_.motor_cmd()[i].kp() = 0;
            low_cmd_.motor_cmd()[i].kd() = 8;
            low_cmd_.motor_cmd()[i].tau() = 0;
        }
    }

    void send_cmd()
    {
        if (initialized_)
        {
            low_cmd_.crc() = crc32_core(
                (uint32_t*)&low_cmd_, (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
            lowcmd_publisher_->Write(low_cmd_);
        }
    }

    void zero_torque_state()
    {
        RCLCPP_INFO(this->get_logger(), "Entering zero torque state");
        create_zero_cmd();
        send_cmd();
        std::this_thread::sleep_for(1s);
    }

    void move_to_default_pos()
    {
        RCLCPP_INFO(this->get_logger(), "Moving to default position");

        // Move time 2s
        float total_time = 2.0f;
        int num_steps = static_cast<int>(total_time / config_.control_dt);

        // Record the current positions
        std::vector<float> init_dof_pos(12, 0.0f);
        for (int i = 0; i < 12; i++)
        {
            init_dof_pos[i] = low_state_.motor_state()[config_.joint2motor_idx[i]].q();
        }

        // Move to default position
        for (int step = 0; step < num_steps; step++)
        {
            float alpha = static_cast<float>(step) / num_steps;
            for (int j = 0; j < 12; j++)
            {
                int motor_idx = config_.joint2motor_idx[j];
                float target_pos = config_.default_angles[j];
                low_cmd_.motor_cmd()[motor_idx].q() =
                    init_dof_pos[j] * (1 - alpha) + target_pos * alpha;
                low_cmd_.motor_cmd()[motor_idx].dq() = 0;
                low_cmd_.motor_cmd()[motor_idx].kp() = 35.0f;
                low_cmd_.motor_cmd()[motor_idx].kd() = 0.5f;
                low_cmd_.motor_cmd()[motor_idx].tau() = 0;
            }
            send_cmd();
            std::this_thread::sleep_for(
                std::chrono::milliseconds(static_cast<int>(config_.control_dt * 1000)));
        }
    }

    void default_pos_state()
    {
        RCLCPP_INFO(this->get_logger(), "Entering default position state");

        for (int i = 0; i < 100; i++)
        {
            for (int j = 0; j < 12; j++)
            {
                int motor_idx = config_.joint2motor_idx[j];
                low_cmd_.motor_cmd()[motor_idx].q() = config_.default_angles[j];
                low_cmd_.motor_cmd()[motor_idx].dq() = 0;
                low_cmd_.motor_cmd()[motor_idx].kp() = 45.0f;
                low_cmd_.motor_cmd()[motor_idx].kd() = 0.5f;
                low_cmd_.motor_cmd()[motor_idx].tau() = 0;
            }
            send_cmd();
            std::this_thread::sleep_for(
                std::chrono::milliseconds(static_cast<int>(config_.control_dt * 1000)));
        }
    }

    void move_to_crouch_pose()
    {
        RCLCPP_INFO(this->get_logger(), "Moving to crouching position");

        // Move time 2s
        float total_time = 2.0f;
        int num_steps = static_cast<int>(total_time / config_.control_dt);

        // Record the current positions
        std::vector<float> init_dof_pos(12, 0.0f);
        for (int i = 0; i < 12; i++)
        {
            init_dof_pos[i] = low_state_.motor_state()[config_.joint2motor_idx[i]].q();
        }

        // Move to crouch position
        for (int step = 0; step < num_steps; step++)
        {
            float alpha = static_cast<float>(step) / num_steps;
            for (int j = 0; j < 12; j++)
            {
                int motor_idx = config_.joint2motor_idx[j];
                float target_pos = config_.crouch_angles[j];
                low_cmd_.motor_cmd()[motor_idx].q() =
                    init_dof_pos[j] * (1 - alpha) + target_pos * alpha;
                low_cmd_.motor_cmd()[motor_idx].dq() = 0;
                low_cmd_.motor_cmd()[motor_idx].kp() = 35.0f;
                low_cmd_.motor_cmd()[motor_idx].kd() = 0.5f;
                low_cmd_.motor_cmd()[motor_idx].tau() = 0;
            }
            send_cmd();
            std::this_thread::sleep_for(
                std::chrono::milliseconds(static_cast<int>(config_.control_dt * 1000)));
        }
    }

    void start_control_sequence()
    {
        if (!initialized_ || running_)
        {
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Starting control sequence");

        // Enter zero torque state
        zero_torque_state();

        // Move to default position
        move_to_default_pos();

        // Enter default position state
        default_pos_state();

        // move_to_crouch_pose();

        running_ = true;
        RCLCPP_INFO(this->get_logger(), "Control sequence started, robot ready");
    }

    void stop_control_sequence()
    {
        if (!initialized_ || !running_)
        {
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Stopping control sequence");

        // Move to crouch position
        move_to_crouch_pose();

        // Enter damping state
        create_damping_cmd();
        send_cmd();

        running_ = false;
        RCLCPP_INFO(this->get_logger(), "Robot stopped");
    }

    void run_controller(bool update_action = true, bool update_goal = true)
    {
        // fmt::print("\n\n===> run control <===\n");

        // emergency stop with controller
        gamepad_.Update(wireless_controller_msg_);
        if (gamepad_.B.on_press)
        {
            fmt::print("emergency stop mode activated by wireless controller.\n");
            g_emergency_stop_mode.store(true, std::memory_order_relaxed);
        }

        if (!config_.loco_only && update_goal)
        {
            // Update goal if needed
            float distance = std::sqrt(
                std::pow(goal_pos_[0] - curr_pos_[0], 2) +
                std::pow(goal_pos_[1] - curr_pos_[1], 2));
            if (distance < 0.5f)
            {
                goal_counter_ = (goal_counter_ + 1) % goals_.size();
                goal_pos_ = goals_[goal_counter_];
                if (config_.manual_rotation)
                {
                    is_rotating_to_goal_ = true;
                }
            }
        }

        if (update_action)
        {
            // Update joint positions and velocities
            for (int i = 0; i < 12; i++)
            {
                qj_[i] = low_state_.motor_state()[config_.joint2motor_idx[i]].q();
                dqj_[i] = low_state_.motor_state()[config_.joint2motor_idx[i]].dq();
            }

            // Get IMU state
            auto quat = Quaternionf(
                low_state_.imu_state().quaternion()[0], low_state_.imu_state().quaternion()[1],
                low_state_.imu_state().quaternion()[2], low_state_.imu_state().quaternion()[3]);
            auto ang_vel = std::array<float, 3>{
                low_state_.imu_state().gyroscope()[0], low_state_.imu_state().gyroscope()[1],
                low_state_.imu_state().gyroscope()[2]
            };
            auto gravity_orientation = get_gravity_orientation_eigen(quat);

            // Create observation
            std::vector<float> qj_obs(qj_.begin(), qj_.end());
            std::vector<float> dqj_obs(dqj_.begin(), dqj_.end());

            // Scale observations
            for (size_t i = 0; i < qj_obs.size(); i++)
            {
                qj_obs[i] = (qj_obs[i] - config_.default_angles[i]) * config_.dof_pos_scale;
                dqj_obs[i] = dqj_obs[i] * config_.dof_vel_scale;
            }
            for (size_t i = 0; i < 3; i++)
            {
                ang_vel[i] = ang_vel[i] * config_.ang_vel_scale;
            }

            // Convert goal to body frame
            // Create quaternion from w,x,y,z
            float qw = curr_quat_.w();
            float qx = curr_quat_.x();
            float qy = curr_quat_.y();
            float qz = curr_quat_.z();
            // fmt::print("curr_quat_: {} {} {} {}\n", curr_quat_.w(), curr_quat_.x(),
            // curr_quat_.y(),
            //            curr_quat_.z());
            // fmt::print("curr_pos_: {} {} {}\n", curr_pos_[0], curr_pos_[1], curr_pos_[2]);

            if (!config_.loco_only)
            {
                // Compute rotation matrix elements
                float n = qw * qw + qx * qx + qy * qy + qz * qz;
                float s = n > 0 ? 2.0f / n : 0.0f;

                float R11 = 1.0f - s * (qy * qy + qz * qz);
                float R12 = s * (qx * qy - qw * qz);
                float R13 = s * (qx * qz + qw * qy);
                float R21 = s * (qx * qy + qw * qz);
                float R22 = 1.0f - s * (qx * qx + qz * qz);
                float R23 = s * (qy * qz - qw * qx);
                float R31 = s * (qx * qz - qw * qy);
                float R32 = s * (qy * qz + qw * qx);
                float R33 = 1.0f - s * (qx * qx + qy * qy);

                // Calculate goal in body frame
                float dx = goal_pos_[0] - curr_pos_[0];
                float dy = goal_pos_[1] - curr_pos_[1];
                float dz = goal_pos_[2] - curr_pos_[2];

                // Rotate to body frame
                float goal_x_b = R11 * dx + R21 * dy + R31 * dz;
                float goal_y_b = R12 * dx + R22 * dy + R32 * dz;

                cmd_(0) = std::atan2(goal_y_b, goal_x_b);
                cmd_(1) = goal_x_b;
                cmd_(2) = goal_y_b;
                // fmt::print("cmd: {} {} {}\n", cmd_(0), cmd_(1), cmd_(2));

                // Log position and goal information (every 50 iterations)
                static int log_counter = 0;
                if (log_counter++ % 100 == 0)
                {
                    RCLCPP_INFO(
                        this->get_logger(),
                        "cmd: [%.2f, %.2f, %.2f] | pos: [%.2f, %.2f, %.2f] | goal: [%.2f, "
                        "%.2f, %.2f]",
                        cmd_[0], cmd_[1], cmd_[2], curr_pos_[0], curr_pos_[1], curr_pos_[2],
                        goal_pos_[0], goal_pos_[1], goal_pos_[2]);
                }
            }

            // Fill proprio observation
            proprio_obs_[0] = ang_vel[0];
            proprio_obs_[1] = ang_vel[1];
            proprio_obs_[2] = ang_vel[2];
            proprio_obs_[3] = gravity_orientation[0];
            proprio_obs_[4] = gravity_orientation[1];
            proprio_obs_[5] = gravity_orientation[2];
            proprio_obs_[6] = cmd_[0];
            proprio_obs_[7] = cmd_[1];
            proprio_obs_[8] = cmd_[2];

            for (int i = 0; i < 12; i++)
            {
                proprio_obs_[9 + i] = qj_obs[i];
                proprio_obs_[21 + i] = dqj_obs[i];
                proprio_obs_[33 + i] = loco_action_[i];
            }

            // Clip proprio observation
            for (auto& val : proprio_obs_)
            {
                val = std::clamp(val, -config_.obs_clip, config_.obs_clip);
            }

            // Set ray observation
            if (!config_.loco_only)
            {
                if (config_.ray_obs_source == RaySource::LIDAR)
                {
                    // ray_obs_ is updated in pointcloud_callback, no need to do anything here
                }
            }

            // Run neural network inference
            try
            {
                if (config_.loco_only || g_emergency_stop_mode.load(std::memory_order_relaxed))
                {
                    // In loco-only mode or emergency stop mode
                    if (g_emergency_stop_mode.load(std::memory_order_relaxed))
                    {
                        // Emergency stop: use zero velocity commands
                        cmd_[0] = 0.0f;
                        cmd_[1] = 0.0f;
                        cmd_[2] = 0.0f;

                        // Log emergency mode status periodically
                        static int emergency_log_counter = 0;
                        if (emergency_log_counter++ % 250 == 0)
                        {
                            // Every 5 seconds at 50Hz
                            RCLCPP_WARN(
                                this->get_logger(),
                                "EMERGENCY STOP MODE: Using zero velocity commands");
                        }
                    }
                    // For normal loco-only mode, cmd_ is already set in odom_callback

                    // Update proprio obs with current commands
                    proprio_obs_[6] = cmd_[0];
                    proprio_obs_[7] = cmd_[1];
                    proprio_obs_[8] = cmd_[2];

                    // Run loco policy directly
                    std::vector<float> loco_action = run_loco_policy();

                    // Process loco action
                    for (auto& val : loco_action)
                    {
                        if (std::isnan(val) || std::isinf(val))
                        {
                            val = 0.0f;
                        }
                    }

                    // Apply smoothing
                    std::vector<float> previous_action = loco_action_;
                    loco_action_ = loco_action;

                    for (size_t i = 0; i < loco_action.size(); i++)
                    {
                        loco_action[i] = 0.8f * loco_action[i] + 0.2f * previous_action[i];
                    }

                    // Scale actions
                    for (size_t i = 0; i < loco_action.size(); i++)
                    {
                        float action_scaled = loco_action[i] * config_.action_scale;
                        if (i < 4)
                        {
                            action_scaled *= 0.5f; // Reduce hip joint action
                        }
                        target_dof_pos_[i] = config_.default_angles[i] + action_scaled;
                    }
                }
                else
                {
                    // Standard dual policy mode
                    // Run high policy
                    if (config_.use_avoidance_policy)
                    {
                        cmd_.setZero();
                    }
                    std::vector<float> high_action = run_high_policy();

                    if (is_rotating_to_goal_)
                    {
                        // Manual rotation logic
                        float goal_direction_world =
                            std::atan2(goal_pos_[1] - curr_pos_[1], goal_pos_[0] - curr_pos_[0]);
                        float current_yaw = std::atan2(
                            2.0f *
                            (curr_quat_.w() * curr_quat_.z() + curr_quat_.x() * curr_quat_.y()),
                            1.0f - 2.0f * (curr_quat_.y() * curr_quat_.y() +
                                curr_quat_.z() * curr_quat_.z()));
                        float yaw_error = goal_direction_world - current_yaw;
                        // Normalize angle to [-pi, pi]
                        if (yaw_error > M_PI)
                            yaw_error -= 2.0f * M_PI;
                        if (yaw_error < -M_PI)
                            yaw_error += 2.0f * M_PI;

                        if (std::abs(yaw_error) < 0.1f)
                        {
                            // 0.1 rad ~= 5.7 degrees
                            is_rotating_to_goal_ = false;
                            high_action[0] = 0.0f;
                            high_action[1] = 0.0f;
                            high_action[2] = 0.0f;
                        }
                        else
                        {
                            high_action[0] = 0.0f; // No forward velocity
                            high_action[1] = 0.0f; // No lateral velocity
                            high_action[2] = std::clamp(yaw_error, -0.5f, 0.5f);
                        }
                    }
                    else
                    {
                        // Clip high action
                        if (slow_start_counter_ < 200)
                        {
                            high_action[0] = std::clamp(high_action[0], -1.5f, 1.5f);
                            high_action[1] = std::clamp(high_action[1], -1.5f, 1.5f);
                            high_action[2] = std::clamp(high_action[2], -0.25f, 0.25f);
                            high_action[3] = std::clamp(high_action[3], 0.20f, 0.40f);
                            slow_start_counter_++;
                        }
                        else
                        {
                            high_action[0] = std::clamp(
                                high_action[0], -config_.high_lin_vel_max * 2.0f,
                                config_.high_lin_vel_max * 2.0f);
                            high_action[1] = std::clamp(
                                high_action[1], -config_.high_lin_vel_lateral_max * 2.0f,
                                config_.high_lin_vel_lateral_max * 2.0f);
                            high_action[2] = std::clamp(
                                high_action[2], -config_.high_ang_vel_max * 0.25f,
                                config_.high_ang_vel_max * 0.25f);
                        }
                    }

                    // Update proprio obs with high action
                    proprio_obs_[6] = high_action[0];
                    proprio_obs_[7] = high_action[1];
                    proprio_obs_[8] = high_action[2];

                    high_action_ = high_action;

                    // Run loco policy
                    std::vector<float> loco_action = run_loco_policy();

                    // Process loco action
                    for (auto& val : loco_action)
                    {
                        if (std::isnan(val) || std::isinf(val))
                        {
                            val = 0.0f;
                        }
                    }

                    // Apply smoothing
                    std::vector<float> previous_action = loco_action_;
                    loco_action_ = loco_action;

                    for (size_t i = 0; i < loco_action.size(); i++)
                    {
                        loco_action[i] = 0.8f * loco_action[i] + 0.2f * previous_action[i];
                    }

                    // Scale actions
                    for (size_t i = 0; i < loco_action.size(); i++)
                    {
                        float action_scaled = loco_action[i] * config_.action_scale;
                        if (i < 4)
                        {
                            action_scaled *= 0.5f; // Reduce hip joint action
                        }
                        target_dof_pos_[i] = config_.default_angles[i] + action_scaled;
                    }
                }
            }
            catch (const Ort::Exception& e)
            {
                RCLCPP_ERROR(this->get_logger(), "ONNX inference error: %s", e.what());
            }
        }

        // Build low command
        for (int i = 0; i < 12; i++)
        {
            int motor_idx = config_.joint2motor_idx[i];
            // target_dof_pos_[i] = config_.crouch_angles[i];
            low_cmd_.motor_cmd()[motor_idx].q() = target_dof_pos_[i];
            low_cmd_.motor_cmd()[motor_idx].dq() = 0;
            low_cmd_.motor_cmd()[motor_idx].kp() = config_.kp;
            low_cmd_.motor_cmd()[motor_idx].kd() = config_.kd;
            low_cmd_.motor_cmd()[motor_idx].tau() = 0;
        }

        // Send the command
        if (!viz_only_)
        {
            send_cmd();
        }
    }

    std::vector<float> run_high_policy()
    {
        // allocate proprio buffer
        std::vector<float> proprio_in(
            config_.num_proprio + config_.num_high_actions + config_.num_high_cmd);

        // fill in proprio input
        std::copy(proprio_obs_.begin(), proprio_obs_.end(), proprio_in.begin());
        std::copy(cmd_.begin(), cmd_.end(), proprio_in.begin() + config_.num_proprio);
        std::copy(
            high_action_.begin(), high_action_.end(),
            proprio_in.begin() + config_.num_proprio + config_.num_high_cmd);
        if (proprio_obs_.size() + cmd_.size() + high_action_.size() != proprio_in.size())
        {
            RCLCPP_ERROR(
                this->get_logger(), "Proprio input size mismatch: expected %zu, got %zu",
                proprio_in.size(), proprio_obs_.size() + cmd_.size() + high_action_.size());
        }

        // Input dimensions and shapes
        const std::vector<int64_t> proprio_dims = {
            1, config_.num_proprio + config_.num_high_actions + config_.num_high_cmd
        };
        const std::vector<int64_t> l1_centroids_dims = {1, config_.num_level1_centroids * 3};
        const std::vector<int64_t> l1_grouped_points_dims = {
            1, config_.num_level1_centroids * config_.level1_max_neighbors * 3
        };
        const std::vector<int64_t> l2_centroids_dims = {1, config_.num_level2_centroids * 3};
        const std::vector<int64_t> l2_grouped_points_dims = {
            1, config_.num_level2_centroids * config_.level2_max_neighbors * 3
        };
        const std::vector<int64_t> l2_grouped_indices_dims = {
            1, config_.num_level2_centroids * config_.level2_max_neighbors
        };
        const std::vector<int64_t> h_dims = {1, 1, 256};
        const std::vector<int64_t> c_dims = {1, 1, 256};

        // Define input names
        const std::array<const char*, 8> input_names = {
            "proprio_in",
            "l1_centroids_in",
            "l1_group_points_in",
            "l2_centroids_in",
            "l2_group_points_in",
            "l2_group_indices",
            "h_in",
            "c_in"
        };
        const std::array<const char*, 3> output_names = {"actions", "h_out", "c_out"};

        // Create input tensors
        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        std::vector<Ort::Value> input_tensors;

        // allocate input on cpu
        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float*>(proprio_in.data()), proprio_in.size(),
                proprio_dims.data(), proprio_dims.size()));

        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, l1_centroids_.data(), l1_centroids_.size(), l1_centroids_dims.data(),
                l1_centroids_dims.size()));

        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, l1_grouped_points_.data(), l1_grouped_points_.size(),
                l1_grouped_points_dims.data(), l1_grouped_points_dims.size()));

        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, l2_centroids_.data(), l2_centroids_.size(), l2_centroids_dims.data(),
                l2_centroids_dims.size()));

        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, l2_grouped_points_.data(), l2_grouped_points_.size(),
                l2_grouped_points_dims.data(), l2_grouped_points_dims.size()));

        input_tensors.push_back(
            Ort::Value::CreateTensor<int32_t>(
                memory_info, l2_grouped_indices_.data(), l2_grouped_indices_.size(),
                l2_grouped_indices_dims.data(), l2_grouped_indices_dims.size()));

        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, high_policy_h_.data(), high_policy_h_.size(), h_dims.data(),
                h_dims.size()));

        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, high_policy_c_.data(), high_policy_c_.size(), h_dims.data(),
                h_dims.size()));

        // Run inference
        auto output_tensors = high_policy_session_->Run(
            Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
            input_tensors.size(), output_names.data(), output_names.size());

        // Extract action
        float* actions_ptr = output_tensors[0].GetTensorMutableData<float>();
        std::vector<float> actions(actions_ptr, actions_ptr + config_.num_high_actions);

        // Update hidden and cell states
        float* h_out_ptr = output_tensors[1].GetTensorMutableData<float>();
        float* c_out_ptr = output_tensors[2].GetTensorMutableData<float>();

        std::copy_n(h_out_ptr, high_policy_h_.size(), high_policy_h_.begin());
        std::copy_n(c_out_ptr, high_policy_c_.size(), high_policy_c_.begin());

        return actions;
    }

    std::vector<float> run_loco_policy()
    {
        // Input dimensions and shapes
        const std::vector<int64_t> input_dims = {1, static_cast<int64_t>(proprio_obs_.size())};
        const std::vector<int64_t> h_dims = {1, 1, 256};

        // Define input names
        const std::array<const char*, 3> input_names = {"obs", "h_in", "c_in"};
        const std::array<const char*, 3> output_names = {"actions", "h_out", "c_out"};

        // Create input tensors
        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, proprio_obs_.data(), proprio_obs_.size(), input_dims.data(),
                input_dims.size()));

        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, loco_policy_h_.data(), loco_policy_h_.size(), h_dims.data(),
                h_dims.size()));

        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, loco_policy_c_.data(), loco_policy_c_.size(), h_dims.data(),
                h_dims.size()));

        // Run inference
        auto output_tensors = loco_policy_session_->Run(
            Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
            input_tensors.size(), output_names.data(), output_names.size());

        // Extract action
        float* actions_ptr = output_tensors[0].GetTensorMutableData<float>();
        std::vector<float> actions(actions_ptr, actions_ptr + config_.num_loco_actions);

        // Update hidden and cell states
        float* h_out_ptr = output_tensors[1].GetTensorMutableData<float>();
        float* c_out_ptr = output_tensors[2].GetTensorMutableData<float>();

        std::copy_n(h_out_ptr, loco_policy_h_.size(), loco_policy_h_.begin());
        std::copy_n(c_out_ptr, loco_policy_c_.size(), loco_policy_c_.begin());

        return actions;
    }

    void pointcloud_callback(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
    {
        auto preprocess_start = std::chrono::steady_clock::now();

        // Process points
        std::vector<Eigen::Vector3d> points;
        points.reserve(msg->point_num);

        for (int i = 0; i < msg->point_num; ++i)
        {
            auto& point = msg->points[i];
            if (std::abs(point.x) < 1e-3 && std::abs(point.y) < 1e-3 && std::abs(point.z) < 1e-3)
            {
                continue;
            }
            Eigen::Vector3d p(point.x, point.y, point.z);
            if (p.norm() > 3.0)
            {
                continue;
            }
            points.push_back(p);
        }

        if (points.empty())
        {
            return;
        }

        auto pcd = std::make_shared<open3d::geometry::PointCloud>(points);

        if (viz_only_)
        {
            publish_raw_pointcloud(*pcd);
        }

        auto preprocess_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> preprocess_duration = preprocess_end - preprocess_start;
        fmt::print("{:.4f}s\n", preprocess_duration.count());

        // 1. Outlier rejection
        auto [pcd_filtered, ind] =
            pcd->RemoveStatisticalOutliers(config_.outlier_nb_neighbors, config_.outlier_std_ratio);

        preprocess_end = std::chrono::steady_clock::now();
        preprocess_duration = preprocess_end - preprocess_start;
        fmt::print("{:.4f}s\n", preprocess_duration.count());

        // 2. Voxel downsampling
        auto pcd_downsampled = pcd_filtered->VoxelDownSample(config_.voxel_size);
        pad_open3d_pointcloud(*pcd_downsampled, config_.num_total_points);

        preprocess_end = std::chrono::steady_clock::now();
        preprocess_duration = preprocess_end - preprocess_start;
        fmt::print("{:.4f}s\n", preprocess_duration.count());

        // 3. Farthest point sampling
        auto pcd_fps = pcd_downsampled->FarthestPointDownSample(config_.num_total_points);
        if (pcd_fps->points_.size() < config_.num_total_points)
        {
            // Pad if not enough points
            if (!pcd_fps->points_.empty())
            {
                while (pcd_fps->points_.size() < config_.num_total_points)
                {
                    pcd_fps->points_.push_back(Eigen::Vector3d::Zero());
                }
            }
            else
            {
                // If no points at all, fill with zeros
                pcd_fps->points_.resize(config_.num_total_points, Eigen::Vector3d::Zero());
            }
        }

        preprocess_end = std::chrono::steady_clock::now();
        preprocess_duration = preprocess_end - preprocess_start;
        fmt::print("{:.4f}s\n", preprocess_duration.count());

        // 4. Apply virtual boundaries
        // apply_virtual_boundaries(*pcd_fps);

        // 5. Two-level grouping
        // Level 1
        auto pcd_l1_centroids = pcd_fps->FarthestPointDownSample(config_.num_level1_centroids);
        if (pcd_l1_centroids->points_.size() < config_.num_level1_centroids)
        {
            if (!pcd_l1_centroids->points_.empty())
            {
                while (pcd_l1_centroids->points_.size() < config_.num_level1_centroids)
                {
                    pcd_l1_centroids->points_.push_back(pcd_l1_centroids->points_.front());
                }
            }
            else
            {
                pcd_l1_centroids->points_.resize(
                    config_.num_level1_centroids, Eigen::Vector3d::Zero());
            }
        }

        open3d::geometry::KDTreeFlann kdtree_fps(*pcd_fps);
        std::vector<Eigen::Vector3f> l1_grouped_points;
        l1_grouped_points.reserve(config_.num_level1_centroids * config_.level1_max_neighbors);

        for (const auto& centroid : pcd_l1_centroids->points_)
        {
            std::vector<int> indices;
            std::vector<double> dists_sq;
            kdtree_fps.SearchRadius(centroid, config_.level1_radius, indices, dists_sq);

            std::vector<Eigen::Vector3f> neighbors;
            for (int idx : indices)
            {
                neighbors.push_back(pcd_fps->points_[idx].cast<float>());
            }
            if (neighbors.size() > config_.level1_max_neighbors)
            {
                neighbors.resize(config_.level1_max_neighbors);
            }
            else if (neighbors.size() < config_.level1_max_neighbors)
            {
                Eigen::Vector3f first_point =
                    neighbors.empty() ? Eigen::Vector3f::Zero() : neighbors.front();
                while (neighbors.size() < config_.level1_max_neighbors)
                {
                    neighbors.push_back(first_point);
                }
            }
            l1_grouped_points.insert(l1_grouped_points.end(), neighbors.begin(), neighbors.end());
        }

        // Level 2
        auto pcd_l2_centroids =
            pcd_l1_centroids->FarthestPointDownSample(config_.num_level2_centroids);
        if (pcd_l2_centroids->points_.size() < config_.num_level2_centroids)
        {
            if (!pcd_l2_centroids->points_.empty())
            {
                while (pcd_l2_centroids->points_.size() < config_.num_level2_centroids)
                {
                    pcd_l2_centroids->points_.push_back(pcd_l2_centroids->points_.front());
                }
            }
            else
            {
                pcd_l2_centroids->points_.resize(
                    config_.num_level2_centroids, Eigen::Vector3d::Zero());
            }
        }

        open3d::geometry::KDTreeFlann kdtree_l1_centroids(*pcd_l1_centroids);
        std::vector<Eigen::Vector3f> l2_grouped_points;
        std::vector<float> l2_grouped_indices;
        l2_grouped_points.reserve(config_.num_level2_centroids * config_.level2_max_neighbors);
        l2_grouped_indices.reserve(config_.num_level2_centroids * config_.level2_max_neighbors);

        for (const auto& centroid : pcd_l2_centroids->points_)
        {
            std::vector<int> indices;
            std::vector<double> dists_sq;
            kdtree_l1_centroids.SearchRadius(centroid, config_.level2_radius, indices, dists_sq);

            std::vector<Eigen::Vector3f> neighbors;
            std::vector<float> neighbor_indices;
            for (int idx : indices)
            {
                neighbors.push_back(pcd_l1_centroids->points_[idx].cast<float>());
                neighbor_indices.push_back(static_cast<float>(idx));
            }

            if (neighbors.size() > config_.level2_max_neighbors)
            {
                neighbors.resize(config_.level2_max_neighbors);
                neighbor_indices.resize(config_.level2_max_neighbors);
            }
            else if (neighbors.size() < config_.level2_max_neighbors)
            {
                Eigen::Vector3f first_point =
                    neighbors.empty() ? Eigen::Vector3f::Zero() : neighbors.front();
                float first_idx = neighbor_indices.empty() ? 0.0f : neighbor_indices.front();
                while (neighbors.size() < config_.level2_max_neighbors)
                {
                    neighbors.push_back(first_point);
                    neighbor_indices.push_back(first_idx);
                }
            }
            l2_grouped_points.insert(l2_grouped_points.end(), neighbors.begin(), neighbors.end());
            l2_grouped_indices.insert(
                l2_grouped_indices.end(), neighbor_indices.begin(), neighbor_indices.end());
        }

        // 6. Update point cloud observation
        // L1 centroids
        l1_centroids_.clear();
        for (const auto& p : pcd_l1_centroids->points_)
        {
            l1_centroids_.push_back(p.x());
            l1_centroids_.push_back(p.y());
            l1_centroids_.push_back(p.z());
        }
        // L1 grouped points
        l1_grouped_points_.clear();
        for (const auto& p : l1_grouped_points)
        {
            l1_grouped_points_.push_back(p.x());
            l1_grouped_points_.push_back(p.y());
            l1_grouped_points_.push_back(p.z());
        }
        // L2 centroids
        l2_centroids_.clear();
        for (const auto& p : pcd_l2_centroids->points_)
        {
            l2_centroids_.push_back(p.x());
            l2_centroids_.push_back(p.y());
            l2_centroids_.push_back(p.z());
        }
        // L2 grouped points
        l2_grouped_points_.clear();
        for (const auto& p : l2_grouped_points)
        {
            l2_grouped_points_.push_back(p.x());
            l2_grouped_points_.push_back(p.y());
            l2_grouped_points_.push_back(p.z());
        }
        // L2 grouped indices
        l2_grouped_indices_.clear();
        for (const auto& idx : l2_grouped_indices)
        {
            l2_grouped_indices_.push_back(idx);
        }

        preprocess_end = std::chrono::steady_clock::now();
        preprocess_duration = preprocess_end - preprocess_start;
        fmt::print("Point cloud preprocessing time: {:.4f}s\n", preprocess_duration.count());

        publish_visualization(*pcd_fps, *pcd_l1_centroids, *pcd_l2_centroids);
    }

    void apply_virtual_boundaries(open3d::geometry::PointCloud& pcd)
    {
        if (!init_pos_set_)
        {
            return; // Can't apply boundaries without position information
        }

        // Using current quaternion to get rotation matrix
        Eigen::Matrix3f R = curr_quat_.toRotationMatrix();

        for (auto& point : pcd.points_)
        {
            Eigen::Vector3f point_world = R * point.cast<float>() + curr_pos_;

            point_world.x() = std::clamp(point_world.x(), config_.bound_x_min, config_.bound_x_max);
            point_world.y() = std::clamp(point_world.y(), config_.bound_y_min, config_.bound_y_max);

            // Transform back to body frame
            point = (R.transpose() * (point_world - curr_pos_)).cast<double>();
        }
    }

    void publish_visualization(
        const open3d::geometry::PointCloud& pcd_fps,
        const open3d::geometry::PointCloud& pcd_l1,
        const open3d::geometry::PointCloud& pcd_l2)
    {
        visualization_msgs::msg::MarkerArray marker_array;
        auto now = this->get_clock()->now();

        auto create_marker = [&](int id, const std::string& ns, const Eigen::Vector3d& pos,
                                 const std::array<float, 4>& color, double scale)
        {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "odom";
            marker.header.stamp = now;
            marker.ns = ns;
            marker.id = id;
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.position.x = pos.x();
            marker.pose.position.y = pos.y();
            marker.pose.position.z = pos.z();
            marker.pose.orientation.w = 1.0;
            marker.scale.x = marker.scale.y = marker.scale.z = scale;
            marker.color.r = color[0];
            marker.color.g = color[1];
            marker.color.b = color[2];
            marker.color.a = color[3];
            marker.lifetime = rclcpp::Duration(0, 200000000); // 0.2 seconds
            return marker;
        };

        // FPS points (blue)
        for (size_t i = 0; i < pcd_fps.points_.size(); ++i)
        {
            marker_array.markers.push_back(
                create_marker(i, "fps_points", pcd_fps.points_[i], {0.3, 0.3, 1.0, 1.0}, 0.05));
        }

        // L1 centroids (green)
        for (size_t i = 0; i < pcd_l1.points_.size(); ++i)
        {
            marker_array.markers.push_back(
                create_marker(i, "l1_centroids", pcd_l1.points_[i], {0.0, 1.0, 0.0, 1.0}, 0.1));
        }

        // L2 centroids (red)
        for (size_t i = 0; i < pcd_l2.points_.size(); ++i)
        {
            marker_array.markers.push_back(
                create_marker(i, "l2_centroids", pcd_l2.points_[i], {1.0, 0.0, 0.0, 1.0}, 0.15));
        }

        ray_marker_publisher_->publish(marker_array);
    }

    void publish_raw_pointcloud(const open3d::geometry::PointCloud& pcd)
    {
        visualization_msgs::msg::MarkerArray marker_array;

        // Sample points to avoid overwhelming the visualization (take every 10th point)
        for (size_t i = 0; i < pcd.points_.size(); i += 1)
        {
            const auto& point = pcd.points_[i];

            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "odom";
            marker.header.stamp = this->get_clock()->now();
            marker.ns = "raw_pointcloud";
            marker.id = static_cast<int>(i);
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;

            marker.pose.position.x = point.x();
            marker.pose.position.y = point.y();
            marker.pose.position.z = point.z();
            marker.pose.orientation.w = 1.0;

            marker.scale.x = 0.05;
            marker.scale.y = 0.05;
            marker.scale.z = 0.05;

            // Color points white
            marker.color.r = 0.4;
            marker.color.g = 0.4;
            marker.color.b = 0.4;
            marker.color.a = 0.7;

            marker.lifetime = rclcpp::Duration(0, 200000000); // 0.2 seconds

            marker_array.markers.push_back(marker);
        }

        if (raw_pointcloud_publisher_)
        {
            raw_pointcloud_publisher_->publish(marker_array);
        }
    }

    // Parameters
    // Data storage
    // ROS2 objects
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr lidar_subscription_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr ray_marker_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr raw_pointcloud_publisher_;
    rclcpp::TimerBase::SharedPtr visualization_timer_;

    // Control objects
    rclcpp::TimerBase::SharedPtr control_timer_;

    // Go2 Configuration
    Go2Config config_;
    bool viz_only_;

    // ONNX Runtime
    Ort::Env env_;
    std::unique_ptr<Ort::Session> loco_policy_session_;
    std::unique_ptr<Ort::Session> high_policy_session_;

    // LSTM states
    VectorXf loco_policy_h_;
    VectorXf loco_policy_c_;
    VectorXf high_policy_h_;
    VectorXf high_policy_c_;

    VectorXf ray_distances_;

    // Controller state
    VectorXf qj_;
    VectorXf dqj_;
    std::vector<float> loco_action_;
    std::vector<float> high_action_;
    VectorXf target_dof_pos_;
    Vector3f cmd_;
    VectorXf proprio_obs_;

    // sampled points (FPS)
    std::vector<float> l1_centroids_;
    std::vector<float> l1_grouped_points_;
    std::vector<float> l2_centroids_;
    std::vector<float> l2_grouped_points_;
    std::vector<int32_t> l2_grouped_indices_;

    // Position tracking
    bool init_pos_set_ = false;
    Vector3f init_pos_;
    Vector3f goal_pos_;
    Vector3f curr_pos_;
    Quaternionf curr_quat_;
    std::vector<Vector3f> goals_;
    size_t goal_counter_;

    // Unitree SDK
    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher_;
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber_;
    ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_> wireless_controller_;
    std::unique_ptr<unitree::robot::go2::SportClient> sport_client_;
    std::unique_ptr<MotionSwitcherClient> motion_switcher_client_;
    unitree_go::msg::dds_::LowCmd_ low_cmd_;
    unitree_go::msg::dds_::LowState_ low_state_;
    unitree_go::msg::dds_::SportModeState_ high_state_;
    unitree_go::msg::dds_::WirelessController_ wireless_controller_msg_;

    Gamepad gamepad_;

    // Controller status
    bool running_;
    bool initialized_;
    int slow_start_counter_;
    float cmd_base_height_;
    bool is_rotating_to_goal_;
};

int main(int argc, char* argv[])
{
    signal(SIGINT, interrupte_handler);

    // parse command line arguments
    cxxopts::Options options("Go2Control", "Unitree Go2 Control");
    options.add_options() //
        ("n,net", "network interface", cxxopts::value<std::string>()->default_value("lo")) //
        ("l,loco", "only locomotion policy", cxxopts::value<bool>()->default_value("false")) //
        ("v,viz", "visualization only mode", cxxopts::value<bool>()->default_value("false")) //
        ("hp,highpolicy", "high-level policy", cxxopts::value<std::string>()->default_value("")) //
        ("lp,locopolicy", "locomotion policy", cxxopts::value<std::string>()->default_value("")) //
        ("a,avoid", "run avoidance policy", cxxopts::value<bool>()->default_value("false")) //
        ("h,help", "Print usage");
    auto args = options.parse(argc, argv);

    auto net_interface = args["net"].as<std::string>();
    int net_idx = net_interface == "lo" ? 1 : 0;
    bool viz_only = args["viz"].as<bool>();

    // initialize unitree sdk
    ChannelFactory::Instance()->Init(net_idx, net_interface);

    // initialize ros
    rclcpp::init(argc, argv);

    Go2Config config;

    YAML::Node yaml_conf = YAML::LoadFile("config.yaml");

    config.use_sim = net_interface == "lo";
    config.use_avoidance_policy = args["avoid"].as<bool>();
    config.kp = yaml_conf["kp"].as<float>();
    config.kd = yaml_conf["kd"].as<float>();
    fmt::print("Using kp: {}, kd: {}\n", config.kp, config.kd);
    config.high_lin_vel_max = yaml_conf["linear_velocity_clip"].as<float>();
    config.high_lin_vel_lateral_max = yaml_conf["linear_velocity_lateral_clip"].as<float>();
    config.high_ang_vel_max = yaml_conf["angular_velocity_clip"].as<float>();
    fmt::print(
        "Using high-level linear velocity max: {}, lateral max: {}, angular max: {}\n",
        config.high_lin_vel_max, config.high_lin_vel_lateral_max, config.high_ang_vel_max);
    if (yaml_conf["pointnet_config"])
    {
        config.voxel_size = yaml_conf["pointnet_config"]["voxel_size"].as<float>();
        config.outlier_nb_neighbors =
            yaml_conf["pointnet_config"]["outlier_nb_neighbors"].as<int>();
        config.outlier_std_ratio = yaml_conf["pointnet_config"]["outlier_std_ratio"].as<double>();
        config.num_total_points = yaml_conf["pointnet_config"]["num_total_points"].as<int>();
        config.num_level1_centroids =
            yaml_conf["pointnet_config"]["num_level1_centroids"].as<int>();
        config.level1_radius = yaml_conf["pointnet_config"]["level1_radius"].as<float>();
        config.level1_max_neighbors =
            yaml_conf["pointnet_config"]["level1_max_neighbors"].as<int>();
        config.num_level2_centroids =
            yaml_conf["pointnet_config"]["num_level2_centroids"].as<int>();
        config.level2_radius = yaml_conf["pointnet_config"]["level2_radius"].as<float>();
        config.level2_max_neighbors =
            yaml_conf["pointnet_config"]["level2_max_neighbors"].as<int>();
        fmt::print(
            "Using PointNet config: voxel_size: {}, num_total_points: {}\n", config.voxel_size,
            config.num_total_points);
    }
    if (yaml_conf["manual_rotation"])
    {
        config.manual_rotation = yaml_conf["manual_rotation"].as<bool>();
        fmt::print("Using manual rotation: {}\n", config.manual_rotation);
    }
    if (yaml_conf["boundary"])
    {
        config.bound_x_min = yaml_conf["boundary"]["x_min"].as<float>();
        config.bound_x_max = yaml_conf["boundary"]["x_max"].as<float>();
        config.bound_y_min = yaml_conf["boundary"]["y_min"].as<float>();
        config.bound_y_max = yaml_conf["boundary"]["y_max"].as<float>();
        fmt::print(
            "Using boundary: x_min: {}, x_max: {}, y_min: {}, y_max: {}\n", config.bound_x_min,
            config.bound_x_max, config.bound_y_min, config.bound_y_max);
    }
    if (yaml_conf["goal_points"])
    {
        config.goals.clear();
        fmt::print("Using goal points:\n");
        for (const auto& goal : yaml_conf["goal_points"])
        {
            config.goals.emplace_back(goal[0].as<float>(), goal[1].as<float>(), 0.0);
            fmt::print("  - ({}, {})\n", goal[0].as<float>(), goal[1].as<float>());
        }
    }
    config.loco_only = args["loco"].as<bool>();
    if (!args["highpolicy"].as<std::string>().empty())
    {
        config.high_policy_name = args["highpolicy"].as<std::string>();
        fmt::print("Using high-level policy: {}\n", config.high_policy_name);
    }
    if (!args["locopolicy"].as<std::string>().empty())
    {
        config.loco_policy_name = args["locopolicy"].as<std::string>();
        fmt::print("Using locomotion policy: {}\n", config.loco_policy_name);
    }
    fmt::print("Using loco_only mode: {}\n", config.loco_only);

    if (config.loco_only)
    {
        fmt::print("Running in locomotion-only mode\n");
        fmt::print("Commands will be received through SportModeState channel\n");
        fmt::print("Use the pygame_control.py script to send commands\n");
    }
    else
    {
        fmt::print("Running with both high-level and locomotion policies\n");
    }

    config.ray_obs_source = RaySource::LIDAR;

    config.init();
    auto node = std::make_shared<Go2Control>(config, viz_only);

    // spin ros in a separate thread
    std::thread spin_thread([&]()
    {
        rclcpp::executors::MultiThreadedExecutor executor;
        executor.add_node(node);
        executor.spin();
        rclcpp::shutdown();
    });
    spin_thread.detach();

    // run control loop
    node->run_control_loop();

    return 0;
}
