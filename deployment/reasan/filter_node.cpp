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
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

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

struct FilterConfig {
    int control_freq = 50;
    std::string filter_policy_path;
    std::string filter_policy_name = "filter_policy";
    int num_estimated_rays = 180;
    int num_actions = 3;
    int num_proprio_obs = 12;
    int lstm_num_layers = 1;
    int lstm_hidden_size = 256;

    // Frequency monitoring
    float frequency_ema_alpha = 0.1f;
    int frequency_publish_interval = 50;

    void init() {
        char result[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
        std::string exe_path = std::string(result, (count > 0) ? count : 0);
        std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();

        filter_policy_path =
            (exe_dir / "./model" / fmt::format("{}.onnx", filter_policy_name)).string();

        // Load frequency monitoring config
        YAML::Node yaml_conf = YAML::LoadFile("config.yaml");
        if (yaml_conf["frequency_monitoring"]) {
            frequency_ema_alpha = yaml_conf["frequency_monitoring"]["ema_alpha"].as<float>(0.1f);
            frequency_publish_interval =
                yaml_conf["frequency_monitoring"]["publish_interval"].as<int>(50);
        }
    }
};

class FilterNode : public rclcpp::Node {
  public:
    FilterNode(FilterConfig config)
        : Node("filter_node")
        , config_(config)
        , initialized_(false) {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "FilterNode");

        estimated_rays_subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/estimated_rays", 10,
            std::bind(&FilterNode::estimated_rays_callback, this, std::placeholders::_1));

        navigation_vel_cmd_subscription_ = this->create_subscription<geometry_msgs::msg::Point>(
            "/navigation_vel_cmd", 10,
            std::bind(&FilterNode::navigation_vel_cmd_callback, this, std::placeholders::_1));

        // Add low state subscription for IMU data
        lowstate_subscriber_ =
            std::make_unique<ChannelSubscriber<unitree_go::msg::dds_::LowState_>>("rt/lowstate");
        lowstate_subscriber_->InitChannel(
            std::bind(&FilterNode::low_state_callback, this, std::placeholders::_1), 10);

        command_publisher_ =
            this->create_publisher<geometry_msgs::msg::Point>("/high_level_command", 10);

        frequency_publisher_ =
            this->create_publisher<std_msgs::msg::Float32>("/filter_node/control_frequency", 10);
        last_loop_time_ = std::chrono::steady_clock::now();

        load_config();
        initialize_controller();

        RCLCPP_INFO(this->get_logger(), "FilterNode initialized");
    }

    void run_control_loop() {
        if (!initialized_) {
            RCLCPP_ERROR(this->get_logger(), "FilterNode not initialized. Exiting.");
            return;
        }

        rclcpp::Rate rate(config_.control_freq);
        int i = 0;
        while (rclcpp::ok()) {
            auto loop_start = std::chrono::steady_clock::now();

            if (i++ % 50 == 0) {
                RCLCPP_INFO(
                    this->get_logger(), "Current cmd: x: %.2f, y: %.2f, z: %.2f",
                    navigation_vel_cmd_[0], navigation_vel_cmd_[1], navigation_vel_cmd_[2]);
            }
            run_controller();

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
        estimated_rays_.resize(config_.num_estimated_rays, 0.0f);
        navigation_vel_cmd_ = {0.0f, 0.0f, 0.0f};
        high_actions_ = {0.0f, 0.0f, 0.0f};

        // Initialize LSTM states
        filter_policy_h_.resize(config_.lstm_num_layers * config_.lstm_hidden_size, 0.0f);
        filter_policy_c_.resize(config_.lstm_num_layers * config_.lstm_hidden_size, 0.0f);

        RCLCPP_INFO(this->get_logger(), "Filter configuration loaded");
    }

    void initialize_controller() {
        try {
            initialize_onnx_runtime();
            initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "Filter controller initialized");
        } catch (const std::exception &e) {
            RCLCPP_ERROR(
                this->get_logger(), "Failed to initialize filter controller: %s", e.what());
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
                this->get_logger(), "Loading filter policy from %s",
                config_.filter_policy_path.c_str());
            filter_session_ = std::make_unique<Ort::Session>(
                env_, config_.filter_policy_path.c_str(), session_options);
            RCLCPP_INFO(this->get_logger(), "Filter policy ONNX model loaded successfully");
        } catch (const Ort::Exception &e) {
            RCLCPP_ERROR(
                this->get_logger(), "Filter policy ONNX initialization error: %s", e.what());
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

    void navigation_vel_cmd_callback(const geometry_msgs::msg::Point::SharedPtr msg) {
        navigation_vel_cmd_[0] = msg->x;
        navigation_vel_cmd_[1] = msg->y;
        navigation_vel_cmd_[2] = msg->z;
    }

    void low_state_callback(const void *msg) {
        low_state_ = *(unitree_go::msg::dds_::LowState_ *)msg;
    }

    void run_controller() {
        try {
            std::vector<float> modified_action = run_filter_policy();
            auto command_msg = std::make_unique<geometry_msgs::msg::Point>();
            command_msg->x = modified_action[0];
            command_msg->y = modified_action[1];
            command_msg->z = modified_action[2];
            command_publisher_->publish(std::move(command_msg));
        } catch (const Ort::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Filter policy ONNX inference error: %s", e.what());
        }
    }

    std::vector<float> run_filter_policy() {
        // Create proprio observation (12 dimensions)
        std::vector<float> proprio_obs(config_.num_proprio_obs);

        // Get IMU data from low state
        auto quat = Quaternionf(
            low_state_.imu_state().quaternion()[0], // w
            low_state_.imu_state().quaternion()[1], // x
            low_state_.imu_state().quaternion()[2], // y
            low_state_.imu_state().quaternion()[3]  // z
        );

        // Base angular velocity * 0.25 (3 values)
        proprio_obs[0] = low_state_.imu_state().gyroscope()[0] * 0.25f;
        proprio_obs[1] = low_state_.imu_state().gyroscope()[1] * 0.25f;
        proprio_obs[2] = low_state_.imu_state().gyroscope()[2] * 0.25f;

        // Projected gravity in body frame (3 values)
        Vector3f gravity_orientation = get_gravity_orientation_eigen(quat);
        proprio_obs[3] = gravity_orientation[0];
        proprio_obs[4] = gravity_orientation[1];
        proprio_obs[5] = gravity_orientation[2];

        // Command (3 values: lin vel x/y + ang vel z, all in body frame)
        proprio_obs[6] = navigation_vel_cmd_[0];
        proprio_obs[7] = navigation_vel_cmd_[1];
        proprio_obs[8] = navigation_vel_cmd_[2];

        // High actions (3 values: last raw output from filter policy)
        proprio_obs[9] = high_actions_[0];
        proprio_obs[10] = high_actions_[1];
        proprio_obs[11] = high_actions_[2];

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
                memory_info, filter_policy_h_.data(), filter_policy_h_.size(), lstm_dims.data(),
                lstm_dims.size()));

        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info, filter_policy_c_.data(), filter_policy_c_.size(), lstm_dims.data(),
                lstm_dims.size()));

        auto output_tensors = filter_session_->Run(
            Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
            input_tensors.size(), output_names.data(), output_names.size());

        // Extract actions
        float *actions_ptr = output_tensors[0].GetTensorMutableData<float>();
        std::vector<float> actions(actions_ptr, actions_ptr + config_.num_actions);

        // Update LSTM hidden and cell states
        float *h_out_ptr = output_tensors[1].GetTensorMutableData<float>();
        float *c_out_ptr = output_tensors[2].GetTensorMutableData<float>();

        std::copy_n(h_out_ptr, filter_policy_h_.size(), filter_policy_h_.begin());
        std::copy_n(c_out_ptr, filter_policy_c_.size(), filter_policy_c_.begin());

        // Update high_actions_ for next iteration
        high_actions_ = actions;

        return actions;
    }

    /**
     * @brief Get gravity orientation from quaternion (equivalent to Python's
     * get_gravity_orientation)
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

    FilterConfig config_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr estimated_rays_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr navigation_vel_cmd_subscription_;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr command_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr frequency_publisher_;

    // Add Unitree SDK subscriber
    std::unique_ptr<ChannelSubscriber<unitree_go::msg::dds_::LowState_>> lowstate_subscriber_;
    unitree_go::msg::dds_::LowState_ low_state_;

    Ort::Env env_;
    std::unique_ptr<Ort::Session> filter_session_;

    std::vector<float> estimated_rays_;
    std::vector<float> navigation_vel_cmd_;
    std::vector<float> high_actions_;

    // LSTM states
    std::vector<float> filter_policy_h_;
    std::vector<float> filter_policy_c_;

    bool initialized_;

    // Frequency monitoring
    std::chrono::steady_clock::time_point last_loop_time_;
    float estimated_frequency_ = 0.0f;
    int frequency_publish_counter_ = 0;
};

int main(int argc, char *argv[]) {
    cxxopts::Options options("FilterNode", "Unitree Go2 Filter Node");
    options.add_options()(
        "n,net", "network interface", cxxopts::value<std::string>()->default_value("lo"))(
        "p,policy", "filter policy",
        cxxopts::value<std::string>()->default_value(""))("h,help", "Print usage");
    auto args = options.parse(argc, argv);

    // Initialize Unitree SDK
    auto net_interface = args["net"].as<std::string>();
    int net_idx = net_interface == "lo" ? 1 : 0;
    ChannelFactory::Instance()->Init(net_idx, net_interface);

    rclcpp::init(argc, argv);

    FilterConfig config;
    if (!args["policy"].as<std::string>().empty()) {
        config.filter_policy_name = args["policy"].as<std::string>();
    }

    config.init();
    auto node = std::make_shared<FilterNode>(config);

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
