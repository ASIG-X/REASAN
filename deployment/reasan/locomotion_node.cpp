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
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"

// Eigen headers
#include <Eigen/Core>
#include <Eigen/Geometry>

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
#include "unitree/idl/go2/WirelessController_.hpp"
#include "unitree/robot/b2/motion_switcher/motion_switcher_api.hpp"
#include "unitree/robot/b2/motion_switcher/motion_switcher_client.hpp"
#include "unitree/robot/channel/channel_factory.hpp"
#include "unitree/robot/channel/channel_publisher.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"
#include "unitree/robot/go2/sport/sport_client.hpp"

using namespace std::chrono_literals;
using namespace unitree::common;
using namespace unitree::robot::b2;
using namespace unitree::robot;
using namespace unitree;
using namespace Eigen;

std::atomic<bool> g_stop_control_signal(false);
std::atomic<bool> g_emergency_stop_mode(false);

void interrupt_handler(int signal_num) {
    static bool first_interrupt = true;
    if (first_interrupt) {
        g_emergency_stop_mode.store(true, std::memory_order_relaxed);
        first_interrupt = false;
        fmt::print("\n=== EMERGENCY STOP ACTIVATED ===\n");
        fmt::print("Switching to locomotion policy with zero velocity commands\n");
        fmt::print("Robot will attempt to stand in place\n");
        fmt::print("Press Ctrl+C again to fully stop the program\n");
    } else {
        g_stop_control_signal.store(true, std::memory_order_relaxed);
        fmt::print("Second Ctrl+C detected - stopping control...\n");
    }
}

/**
 * @brief Get gravity orientation from quaternion (equivalent to Python's get_gravity_orientation)
 * @param quat Quaternion representing orientation
 * @return Vector3f representing gravity orientation
 */
Vector3f get_gravity_orientation_eigen(const Quaternionf &quat) {
    Vector3f gravity_orientation;
    gravity_orientation.x() = 2.0f * (-quat.z() * quat.x() + quat.w() * quat.y());
    gravity_orientation.y() = -2.0f * (quat.z() * quat.y() + quat.w() * quat.x());
    gravity_orientation.z() = 1.0f - 2.0f * (quat.w() * quat.w() + quat.z() * quat.z());

    return gravity_orientation;
}

uint32_t crc32_core(uint32_t *ptr, uint32_t len) {
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; i++) {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++) {
            if (CRC32 & 0x80000000) {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            } else {
                CRC32 <<= 1;
            }

            if (data & xbit)
                CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }

    return CRC32;
}

template <typename Iterable> auto enumerate(Iterable &&iterable) {
    using Iterator = decltype(std::begin(std::declval<Iterable>()));
    using T = decltype(*std::declval<Iterator>());

    struct Enumerated {
        std::size_t index;
        T element;
    };

    struct Enumerator {
        Iterator iterator;
        std::size_t index;

        auto operator!=(const Enumerator &other) const { return iterator != other.iterator; }

        auto &operator++() {
            ++iterator;
            ++index;
            return *this;
        }

        auto operator*() const { return Enumerated{index, *iterator}; }
    };

    struct Wrapper {
        Iterable &iterable;

        [[nodiscard]] auto begin() const { return Enumerator{std::begin(iterable), 0U}; }

        [[nodiscard]] auto end() const { return Enumerator{std::end(iterable), 0U}; }
    };

    return Wrapper{std::forward<Iterable>(iterable)};
}

// Configuration for Go2 robot (locomotion-specific parts)
struct LocomotionConfig {
    int control_freq = 50;
    float control_dt = 0.02f;

    std::string lowcmd_topic = "rt/lowcmd";
    std::string lowstate_topic = "rt/lowstate";

    std::string loco_policy_path;
    std::string loco_policy_name = "loco_policy";

    float kp = 32.0f;
    float kd = 1.0f;

    // Joint index mapping
    std::array<int, 12> joint2motor_idx = {3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8};

    // Default joint angles
    VectorXf default_angles;
    VectorXf crouch_angles;

    // ordered joint names in simulation (model input)
    std::vector<std::string> sim_joint_names = {
        "FL_hip_joint",   "FR_hip_joint",   "RL_hip_joint",   "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint",  "FR_calf_joint",  "RL_calf_joint",  "RR_calf_joint",
    };

    std::vector<std::string> motor_joint_names = {
        "FR_hip_joint",   "FR_thigh_joint", "FR_calf_joint",  "FL_hip_joint",
        "FL_thigh_joint", "FL_calf_joint",  "RR_hip_joint",   "RR_thigh_joint",
        "RR_calf_joint",  "RL_hip_joint",   "RL_thigh_joint", "RL_calf_joint",
    };

    std::unordered_map<std::string, float> default_joint_angles = {
        {"FL_hip_joint", 0.1},   {"RL_hip_joint", 0.1},   {"FR_hip_joint", -0.1},
        {"RR_hip_joint", -0.1},  {"FL_thigh_joint", 0.8}, {"RL_thigh_joint", 1.0},
        {"FR_thigh_joint", 0.8}, {"RR_thigh_joint", 1.0}, {"FL_calf_joint", -1.5},
        {"RL_calf_joint", -1.5}, {"FR_calf_joint", -1.5}, {"RR_calf_joint", -1.5},
    };

    std::unordered_map<std::string, float> crouch_joint_angles = {
        {"FL_hip_joint", 0.125},  {"FL_thigh_joint", 1.23}, {"FL_calf_joint", -2.70},
        {"FR_hip_joint", -0.125}, {"FR_thigh_joint", 1.23}, {"FR_calf_joint", -2.70},
        {"RL_hip_joint", 0.47},   {"RL_thigh_joint", 1.25}, {"RL_calf_joint", -2.72},
        {"RR_hip_joint", -0.47},  {"RR_thigh_joint", 1.25}, {"RR_calf_joint", -2.72},
    };

    // Joint limits
    std::vector<std::pair<float, float>> joint_limits = {
        {-1.0472, 1.0472},  {-1.0472, 1.0472},  {-1.0472, 1.0472},  {-1.0472, 1.0472},
        {-1.5708, 3.4907},  {-1.5708, 3.4907},  {-0.5236, 4.5379},  {-0.5236, 4.5379},
        {-2.7227, -0.8378}, {-2.7227, -0.8378}, {-2.7227, -0.8378}, {-2.7227, -0.8378},
    };

    // Observation scales
    float lin_vel_scale = 2.0f;
    float ang_vel_scale = 0.25f;
    float dof_pos_scale = 1.0f;
    float dof_vel_scale = 0.05f;
    std::array<float, 3> cmd_scale = {lin_vel_scale, lin_vel_scale, ang_vel_scale};
    float obs_clip = 100.0f;

    float forward_clip = 2.5;
    float lateral_clip = 1.5;
    float angular_clip = 3.0;

    // Frequency monitoring
    float frequency_ema_alpha = 0.1f;
    int frequency_publish_interval = 50;

    // Action scales
    float action_scale = 0.25f;
    float action_clip = 100.0f;

    // Network dimensions
    int num_loco_actions = 12;
    int num_proprio = 45;

    bool use_sim = false;

    void init() {
        fmt::print("joint2motor_idx: ");
        for (auto &&[i, name] : enumerate(sim_joint_names)) {
            auto index = std::find(motor_joint_names.begin(), motor_joint_names.end(), name) -
                         motor_joint_names.begin();
            joint2motor_idx.at(i) = index;
            fmt::print("{} ", index);
        }
        fmt::print("\nshould be:       3 0 9 6 4 1 10 7 5 2 11 8\n");

        default_angles.resize(num_loco_actions);
        crouch_angles.resize(num_loco_actions);
        for (int i = 0; i < num_loco_actions; ++i) {
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

        // Modify joint limits to reduce range
        for (auto &limit : joint_limits) {
            float range = limit.second - limit.first;
            limit.first += 0.05f * range;
            limit.second -= 0.05f * range;
            if (limit.second <= limit.first) {
                throw std::runtime_error("Joint limits are invalid after modification.");
            }
        }

        // Load frequency monitoring config
        YAML::Node yaml_conf = YAML::LoadFile("config.yaml");
        if (yaml_conf["frequency_monitoring"]) {
            frequency_ema_alpha = yaml_conf["frequency_monitoring"]["ema_alpha"].as<float>(0.1f);
            frequency_publish_interval =
                yaml_conf["frequency_monitoring"]["publish_interval"].as<int>(50);
        }
    }
};

/**
 * @brief Locomotion control node - handles low-level joint control and locomotion policy
 */
class LocomotionNode : public rclcpp::Node {
  public:
    LocomotionNode(LocomotionConfig config)
        : Node("locomotion_node")
        , config_(config)
        , running_(false)
        , initialized_(false) {
        // Initialize ONNX environment once
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LocomotionNode");

        // Create command subscription for high-level commands
        command_subscription_ = this->create_subscription<geometry_msgs::msg::Point>(
            "/high_level_command", 10,
            std::bind(&LocomotionNode::command_callback, this, std::placeholders::_1));

        // Create frequency publisher
        frequency_publisher_ = this->create_publisher<std_msgs::msg::Float32>(
            "/locomotion_node/control_frequency", 10);
        last_loop_time_ = std::chrono::steady_clock::now();

        // Load configuration
        load_config();

        // Initialize controller
        initialize_controller();

        RCLCPP_INFO(this->get_logger(), "LocomotionNode initialized");
    }

    ~LocomotionNode() {
        // Ensure we send zero commands when shutting down
        fmt::print("shutting down locomotion node...\n");
        if (initialized_) {
            create_zero_cmd();
            send_cmd();
            RCLCPP_INFO(this->get_logger(), "Locomotion control shut down");
        }
    }

    void run_control_loop() {
        fmt::print("attempting to start locomotion control loop...\n");

        if (!initialized_) {
            fmt::print("quit locomotion control loop.\n");
            return;
        }

        start_control_sequence();

        rclcpp::Rate rate(config_.control_freq);
        int i = 0;

        while (true) {
            auto loop_start = std::chrono::steady_clock::now();

            if (g_stop_control_signal.load(std::memory_order_relaxed)) {
                fmt::print("stopping locomotion control...\n");
                break;
            }
            run_controller();

            auto loop_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> loop_duration = loop_end - loop_start;

            // Update and publish frequency
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

            static int log_counter = 0;
            if (log_counter++ % 50 == 0) {
                fmt::print("Locomotion control iteration time: {:.10f}s\n", loop_duration.count());
                fmt::print(
                    "Current cmd: x: {:.2f}, y: {:.2f}, z: {:.2f}\n", cmd_(0), cmd_(1), cmd_(2));
            }

            rate.sleep();
        }

        stop_control_sequence();
    }

  private:
    void load_config() {
        // Initialize controller state variables
        qj_.resize(config_.num_loco_actions);
        qj_.setZero();
        dqj_.resize(config_.num_loco_actions);
        dqj_.setZero();
        loco_action_.resize(config_.num_loco_actions, 0.0f);
        target_dof_pos_ = config_.default_angles;
        cmd_ = {0.0f, 0.0f, 0.0f};

        proprio_obs_.resize(config_.num_proprio);
        proprio_obs_.setZero();

        RCLCPP_INFO(this->get_logger(), "Locomotion configuration loaded");
    }

    void initialize_controller() {
        fmt::print("initializing locomotion controller...\n");
        try {
            // Initialize command publisher and state subscribers
            lowcmd_publisher_ = std::make_unique<ChannelPublisher<unitree_go::msg::dds_::LowCmd_>>(
                config_.lowcmd_topic);
            lowcmd_publisher_->InitChannel();

            lowstate_subscriber_ =
                std::make_unique<ChannelSubscriber<unitree_go::msg::dds_::LowState_>>(
                    config_.lowstate_topic);
            lowstate_subscriber_->InitChannel(
                std::bind(&LocomotionNode::low_state_callback, this, std::placeholders::_1), 10);

            // subscribe to wireless controller
            wireless_controller_.reset(
                new ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>(
                    "rt/wirelesscontroller"));
            wireless_controller_->InitChannel(
                std::bind(
                    &LocomotionNode::wireless_controller_callback, this, std::placeholders::_1),
                10);

            // Initialize ONNX models
            initialize_onnx_runtime();

            // Initialize the command
            init_cmd_go();

            // Wait for initial state data
            wait_for_low_state();

            // Initialize robot clients
            if (!config_.use_sim) {
                sport_client_ = std::make_unique<unitree::robot::go2::SportClient>();
                motion_switcher_client_ = std::make_unique<MotionSwitcherClient>();

                sport_client_->SetTimeout(5.0f);
                sport_client_->Init();
                sport_client_->StandDown();
                std::this_thread::sleep_for(2s);

                motion_switcher_client_->SetTimeout(5.0f);
                motion_switcher_client_->Init();

                fmt::print("switching to release mode.\n");
                if (motion_switcher_client_->ReleaseMode() == 0) {
                    fmt::print("switch to release mode success.\n");
                } else {
                    throw std::runtime_error("switch to release mode failed.");
                }
                std::this_thread::sleep_for(5s);
            }

            initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "Locomotion controller initialized");
        } catch (const std::exception &e) {
            RCLCPP_ERROR(
                this->get_logger(), "Failed to initialize locomotion controller: %s", e.what());
        }
        fmt::print("locomotion controller initialized.\n");
    }

    void initialize_onnx_runtime() {
        fmt::print("initializing ONNX Runtime for locomotion...\n");

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
                this->get_logger(), "Loading locomotion policy from %s",
                config_.loco_policy_path.c_str());

            loco_policy_session_ = std::make_unique<Ort::Session>(
                env_, config_.loco_policy_path.c_str(), session_options);

            // Initialize LSTM hidden and cell states for loco policy
            loco_policy_h_.resize(256);
            loco_policy_h_.setZero();

            loco_policy_c_.resize(256);
            loco_policy_c_.setZero();

            RCLCPP_INFO(this->get_logger(), "Locomotion ONNX model loaded successfully");
        } catch (const Ort::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Locomotion ONNX initialization error: %s", e.what());
            throw;
        }
    }

    void command_callback(const geometry_msgs::msg::Point::SharedPtr msg) {
        // Extract commands from Point message: x=forward, y=lateral, z=angular
        cmd_(0) = msg->x;
        cmd_(1) = msg->y;
        cmd_(2) = msg->z;

        // clip velocity command
        cmd_(0) = std::clamp(cmd_(0), -config_.forward_clip, config_.forward_clip);
        cmd_(1) = std::clamp(cmd_(1), -config_.lateral_clip, config_.lateral_clip);
        cmd_(2) = std::clamp(cmd_(2), -config_.angular_clip, config_.angular_clip);

        // In emergency stop mode, override with zero commands
        if (g_emergency_stop_mode.load(std::memory_order_relaxed)) {
            cmd_(0) = 0.0f;
            cmd_(1) = 0.0f;
            cmd_(2) = 0.0f;
        }

        // Log commands periodically
        static int log_counter = 0;
        if (log_counter++ % 100 == 0) {
            RCLCPP_INFO(
                this->get_logger(), "Received commands: [%.2f, %.2f, %.2f]", cmd_[0], cmd_[1],
                cmd_[2]);
        }
    }

    void low_state_callback(const void *msg) {
        low_state_ = *(unitree_go::msg::dds_::LowState_ *)msg;
    }

    void wireless_controller_callback(const void *msg) {
        wireless_controller_msg_ = *(unitree_go::msg::dds_::WirelessController_ *)msg;
    }

    void wait_for_low_state() {
        if (config_.use_sim) {
            fmt::print("Simulation mode, skipping wait for low state.\n");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Waiting for robot state...");
        auto start = std::chrono::steady_clock::now();
        while (low_state_.tick() == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
            if (elapsed > 5) {
                RCLCPP_WARN(this->get_logger(), "Timeout waiting for robot state");
                break;
            }
        }
        RCLCPP_INFO(this->get_logger(), "Successfully connected to the robot");
    }

    void init_cmd_go() {
        low_cmd_.head()[0] = 0xFE;
        low_cmd_.head()[1] = 0xEF;
        low_cmd_.level_flag() = 0xFF;
        low_cmd_.gpio() = 0;

        float PosStopF = 2.146e9f;
        float VelStopF = 16000.0f;

        for (int i = 0; i < low_cmd_.motor_cmd().size(); i++) {
            low_cmd_.motor_cmd()[i].mode() = 0x0A;
            low_cmd_.motor_cmd()[i].q() = PosStopF;
            low_cmd_.motor_cmd()[i].dq() = VelStopF;
            low_cmd_.motor_cmd()[i].kp() = 0;
            low_cmd_.motor_cmd()[i].kd() = 0;
            low_cmd_.motor_cmd()[i].tau() = 0;
        }
    }

    void create_zero_cmd() {
        for (int i = 0; i < 12; i++) {
            low_cmd_.motor_cmd()[i].q() = 0;
            low_cmd_.motor_cmd()[i].dq() = 0;
            low_cmd_.motor_cmd()[i].kp() = 0;
            low_cmd_.motor_cmd()[i].kd() = 0;
            low_cmd_.motor_cmd()[i].tau() = 0;
        }
    }

    void create_damping_cmd() {
        for (int i = 0; i < 12; i++) {
            low_cmd_.motor_cmd()[i].q() = 0;
            low_cmd_.motor_cmd()[i].dq() = 0;
            low_cmd_.motor_cmd()[i].kp() = 0;
            low_cmd_.motor_cmd()[i].kd() = 8;
            low_cmd_.motor_cmd()[i].tau() = 0;
        }
    }

    void send_cmd() {
        if (initialized_) {
            low_cmd_.crc() = crc32_core(
                (uint32_t *)&low_cmd_, (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
            lowcmd_publisher_->Write(low_cmd_);
        }
    }

    void zero_torque_state() {
        RCLCPP_INFO(this->get_logger(), "Entering zero torque state");
        create_zero_cmd();
        send_cmd();
        std::this_thread::sleep_for(1s);
    }

    void move_to_default_pos() {
        RCLCPP_INFO(this->get_logger(), "Moving to default position");

        // Move time 2s
        float total_time = 2.0f;
        int num_steps = static_cast<int>(total_time / config_.control_dt);

        // Record the current positions
        std::vector<float> init_dof_pos(12, 0.0f);
        for (int i = 0; i < 12; i++) {
            init_dof_pos[i] = low_state_.motor_state()[config_.joint2motor_idx[i]].q();
        }

        // Move to default position
        for (int step = 0; step < num_steps; step++) {
            float alpha = static_cast<float>(step) / num_steps;
            for (int j = 0; j < 12; j++) {
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

    void default_pos_state() {
        RCLCPP_INFO(this->get_logger(), "Entering default position state");

        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 12; j++) {
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

    void move_to_crouch_pose() {
        RCLCPP_INFO(this->get_logger(), "Moving to crouching position");

        // Move time 2s
        float total_time = 2.0f;
        int num_steps = static_cast<int>(total_time / config_.control_dt);

        // Record the current positions
        std::vector<float> init_dof_pos(12, 0.0f);
        for (int i = 0; i < 12; i++) {
            init_dof_pos[i] = low_state_.motor_state()[config_.joint2motor_idx[i]].q();
        }

        // Move to crouch position
        for (int step = 0; step < num_steps; step++) {
            float alpha = static_cast<float>(step) / num_steps;
            for (int j = 0; j < 12; j++) {
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

    void start_control_sequence() {
        if (!initialized_ || running_) {
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Starting locomotion control sequence");

        // Enter zero torque state
        zero_torque_state();

        // Move to default position
        move_to_default_pos();

        // Enter default position state
        default_pos_state();

        running_ = true;
        RCLCPP_INFO(this->get_logger(), "Locomotion control sequence started, robot ready");
    }

    void stop_control_sequence() {
        if (!initialized_ || !running_) {
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Stopping locomotion control sequence");

        move_to_default_pos();

        move_to_crouch_pose();

        // Enter damping state
        create_damping_cmd();
        send_cmd();

        running_ = false;
        RCLCPP_INFO(this->get_logger(), "Locomotion stopped");
    }

    void run_controller() {
        // emergency stop with controller
        gamepad_.Update(wireless_controller_msg_);
        if (gamepad_.B.on_press) {
            if (!g_emergency_stop_mode.load(std::memory_order_relaxed)) {
                fmt::print("emergency stop mode activated by wireless controller.\n");
                g_emergency_stop_mode.store(true, std::memory_order_relaxed);
            } else {
                fmt::print("exiting emergency stop mode by wireless controller.\n");
                g_stop_control_signal.store(true, std::memory_order_relaxed);
            }
        }

        // Update joint positions and velocities
        for (int i = 0; i < 12; i++) {
            qj_[i] = low_state_.motor_state()[config_.joint2motor_idx[i]].q();
            dqj_[i] = low_state_.motor_state()[config_.joint2motor_idx[i]].dq();
        }

        // Get IMU state
        auto quat = Quaternionf(
            low_state_.imu_state().quaternion()[0], low_state_.imu_state().quaternion()[1],
            low_state_.imu_state().quaternion()[2], low_state_.imu_state().quaternion()[3]);
        auto ang_vel = std::array<float, 3>{
            low_state_.imu_state().gyroscope()[0], low_state_.imu_state().gyroscope()[1],
            low_state_.imu_state().gyroscope()[2]};
        auto gravity_orientation = get_gravity_orientation_eigen(quat);

        // Create observation
        std::vector<float> qj_obs(qj_.begin(), qj_.end());
        std::vector<float> dqj_obs(dqj_.begin(), dqj_.end());

        // Scale observations
        for (size_t i = 0; i < qj_obs.size(); i++) {
            qj_obs[i] = (qj_obs[i] - config_.default_angles[i]) * config_.dof_pos_scale;
            dqj_obs[i] = dqj_obs[i] * config_.dof_vel_scale;
        }
        for (size_t i = 0; i < 3; i++) {
            ang_vel[i] = ang_vel[i] * config_.ang_vel_scale;
        }

        // Fill proprio observation
        proprio_obs_[0] = ang_vel[0];
        proprio_obs_[1] = ang_vel[1];
        proprio_obs_[2] = ang_vel[2];
        proprio_obs_[3] = gravity_orientation[0];
        proprio_obs_[4] = gravity_orientation[1];
        proprio_obs_[5] = gravity_orientation[2];
        proprio_obs_[6] = cmd_[0] * 2.0f;
        proprio_obs_[7] = cmd_[1] * 2.0f;
        proprio_obs_[8] = cmd_[2] * 0.25f;

        for (int i = 0; i < 12; i++) {
            proprio_obs_[9 + i] = qj_obs[i];
            proprio_obs_[21 + i] = dqj_obs[i];
            proprio_obs_[33 + i] = loco_action_[i];
        }

        // Clip proprio observation
        for (auto &val : proprio_obs_) {
            val = std::clamp(val, -config_.obs_clip, config_.obs_clip);
        }

        // Run neural network inference
        try {
            // Run loco policy
            std::vector<float> loco_action = run_loco_policy();

            // Process loco action
            for (auto &val : loco_action) {
                if (std::isnan(val) || std::isinf(val)) {
                    val = 0.0f;
                }
            }

            // Apply smoothing
            std::vector<float> previous_action = loco_action_;
            loco_action_ = loco_action;

            for (size_t i = 0; i < loco_action.size(); i++) {
                loco_action[i] = 0.8f * loco_action[i] + 0.2f * previous_action[i];
            }

            // Scale actions
            for (size_t i = 0; i < loco_action.size(); i++) {
                float action_scaled = loco_action[i] * config_.action_scale;
                if (i < 4) {
                    action_scaled *= 0.5f; // Reduce hip joint action
                }
                target_dof_pos_[i] = config_.default_angles[i] + action_scaled;
            }
        } catch (const Ort::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Locomotion ONNX inference error: %s", e.what());
        }

        // Build low command
        for (int i = 0; i < 12; i++) {
            int motor_idx = config_.joint2motor_idx[i];
            low_cmd_.motor_cmd()[motor_idx].q() = target_dof_pos_[i];
            low_cmd_.motor_cmd()[motor_idx].dq() = 0;
            low_cmd_.motor_cmd()[motor_idx].kp() = config_.kp;
            low_cmd_.motor_cmd()[motor_idx].kd() = config_.kd;
            low_cmd_.motor_cmd()[motor_idx].tau() = 0;
        }

        // Send the command
        send_cmd();
    }

    std::vector<float> run_loco_policy() {
        // Input dimensions and shapes
        const std::vector<int64_t> input_dims = {1, static_cast<int64_t>(proprio_obs_.size())};
        const std::vector<int64_t> h_dims = {1, 1, 256};

        // Define input names
        const std::array<const char *, 3> input_names = {"obs", "h_in", "c_in"};
        const std::array<const char *, 3> output_names = {"actions", "h_out", "c_out"};

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
        float *actions_ptr = output_tensors[0].GetTensorMutableData<float>();
        std::vector<float> actions(actions_ptr, actions_ptr + config_.num_loco_actions);

        // Update hidden and cell states
        float *h_out_ptr = output_tensors[1].GetTensorMutableData<float>();
        float *c_out_ptr = output_tensors[2].GetTensorMutableData<float>();

        std::copy_n(h_out_ptr, loco_policy_h_.size(), loco_policy_h_.begin());
        std::copy_n(c_out_ptr, loco_policy_c_.size(), loco_policy_c_.begin());

        return actions;
    }

  private:
    // Go2 Configuration
    LocomotionConfig config_;

    // ROS2 objects
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr command_subscription_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr frequency_publisher_;

    // ONNX Runtime
    Ort::Env env_;
    std::unique_ptr<Ort::Session> loco_policy_session_;

    // LSTM states
    VectorXf loco_policy_h_;
    VectorXf loco_policy_c_;

    // Controller state
    VectorXf qj_;
    VectorXf dqj_;
    std::vector<float> loco_action_;
    VectorXf target_dof_pos_;
    Vector3f cmd_;
    VectorXf proprio_obs_;

    // Unitree SDK
    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher_;
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber_;
    ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_> wireless_controller_;
    std::unique_ptr<unitree::robot::go2::SportClient> sport_client_;
    std::unique_ptr<MotionSwitcherClient> motion_switcher_client_;
    unitree_go::msg::dds_::LowCmd_ low_cmd_{};
    unitree_go::msg::dds_::LowState_ low_state_{};
    unitree_go::msg::dds_::WirelessController_ wireless_controller_msg_{};

    Gamepad gamepad_;

    // Controller status
    bool running_;
    bool initialized_;

    // Frequency monitoring
    std::chrono::steady_clock::time_point last_loop_time_;
    float estimated_frequency_ = 0.0f;
    int frequency_publish_counter_ = 0;
};

int main(int argc, char *argv[]) {
    signal(SIGINT, interrupt_handler);

    // parse command line arguments
    cxxopts::Options options("LocomotionNode", "Unitree Go2 Locomotion Control");
    options.add_options()                                                                        //
        ("n,net", "network interface", cxxopts::value<std::string>()->default_value("lo"))       //
        ("lp,locopolicy", "locomotion policy", cxxopts::value<std::string>()->default_value("")) //
        ("h,help", "Print usage");
    auto args = options.parse(argc, argv);

    auto net_interface = args["net"].as<std::string>();
    int net_idx = net_interface == "lo" ? 1 : 0;

    // initialize unitree sdk
    ChannelFactory::Instance()->Init(net_idx, net_interface);

    // initialize ros
    rclcpp::init(argc, argv);

    LocomotionConfig config;

    YAML::Node yaml_conf = YAML::LoadFile("config.yaml");

    config.use_sim = net_interface == "lo";
    config.kp = yaml_conf["kp"].as<float>();
    config.kd = yaml_conf["kd"].as<float>();
    config.forward_clip = yaml_conf["linear_velocity_clip"].as<float>();
    config.lateral_clip = yaml_conf["linear_velocity_lateral_clip"].as<float>();
    config.angular_clip = yaml_conf["angular_velocity_clip"].as<float>();
    fmt::print("Using kp: {}, kd: {}\n", config.kp, config.kd);

    // Load frequency monitoring config
    YAML::Node freq_conf = yaml_conf["frequency_monitoring"];
    if (freq_conf) {
        config.frequency_ema_alpha = freq_conf["ema_alpha"].as<float>(0.1f);
        config.frequency_publish_interval = freq_conf["publish_interval"].as<int>(50);
    }

    if (!args["locopolicy"].as<std::string>().empty()) {
        config.loco_policy_name = args["locopolicy"].as<std::string>();
        fmt::print("Using locomotion policy: {}\n", config.loco_policy_name);
    }

    config.init();
    auto node = std::make_shared<LocomotionNode>(config);

    // spin ros in a separate thread
    std::thread spin_thread([&]() {
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