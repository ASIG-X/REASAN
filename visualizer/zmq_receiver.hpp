#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <zmq.hpp>

// Data structures to hold received data
struct IMUData {
    float quaternion[4] = {1, 0, 0, 0};
    float gyroscope[3] = {0, 0, 0};
    float accelerometer[3] = {0, 0, 0};
    float rpy[3] = {0, 0, 0};
    float gravity[3] = {0, 0, -1};
    uint8_t temperature = 0;
};

struct MotorData {
    uint8_t mode = 0;
    float q = 0;
    float dq = 0;
    float ddq = 0;
    float tau_est = 0;
    uint8_t temperature = 0;
    uint32_t lost = 0;
};

struct BatteryData {
    uint8_t version_high = 0;
    uint8_t version_low = 0;
    uint8_t status = 0;
    uint8_t soc = 0;
    int32_t current = 0;
    uint16_t cycle = 0;
    uint8_t bq_ntc[2] = {0, 0};
    uint8_t mcu_ntc[2] = {0, 0};
    uint16_t cell_vol[15] = {0};
};

struct ControlData {
    uint8_t mode = 0; // 0 = FILTER_POLICY, 1 = WIRELESS_CONTROL
    float cmd[3] = {0, 0, 0};
    float nav_cmd[3] = {0, 0, 0};
};

class ZMQReceiver {
  public:
    ZMQReceiver(const std::string &address);
    ~ZMQReceiver();

    void start();
    void stop();

    // Thread-safe getters
    std::vector<float> getEstimatedRays();
    std::vector<float> getPointCloud(); // Returns XYZI format (4 floats per point)
    bool pointCloudHasIntensity();      // True if intensity data is available
    IMUData getIMUData();
    std::vector<MotorData> getMotorData();
    int16_t getFootForce(int index);
    BatteryData getBatteryData();
    ControlData getControlData();

    bool isConnected() const { return connected_; }

  private:
    void receiveLoop();
    void parseMessage(const zmq::message_t &msg);

    void parseRays(const uint8_t *data, size_t size);
    void parsePointCloud(const uint8_t *data, size_t size);
    void parseIMU(const uint8_t *data, size_t size);
    void parseMotors(const uint8_t *data, size_t size);
    void parseFootForce(const uint8_t *data, size_t size);
    void parseBattery(const uint8_t *data, size_t size);
    void parseControl(const uint8_t *data, size_t size);

    std::string address_;
    zmq::context_t context_;
    zmq::socket_t socket_;

    std::thread receiveThread_;
    std::atomic<bool> running_;
    std::atomic<bool> connected_;

    // Data storage with mutex protection
    std::mutex dataMutex_;
    std::vector<float> estimatedRays_;
    std::vector<float> pointCloud_; // XYZI format (4 floats per point)
    bool pointCloudHasIntensity_ = false;
    IMUData imuData_;
    std::vector<MotorData> motorData_;
    int16_t footForce_[4] = {0, 0, 0, 0};
    BatteryData batteryData_;
    ControlData controlData_;
};
