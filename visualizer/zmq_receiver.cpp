#include "zmq_receiver.hpp"
#include <cstdio>

ZMQReceiver::ZMQReceiver(const std::string &address)
    : address_(address)
    , context_(1)
    , socket_(context_, zmq::socket_type::sub)
    , running_(false)
    , connected_(false) {
    motorData_.resize(12);
}

ZMQReceiver::~ZMQReceiver() { stop(); }

void ZMQReceiver::start() {
    if (running_)
        return;

    try {
        socket_.connect(address_);
        socket_.set(zmq::sockopt::subscribe, ""); // Subscribe to all messages
        socket_.set(zmq::sockopt::rcvtimeo, 100); // 100ms timeout

        running_ = true;
        connected_ = true;
        receiveThread_ = std::thread(&ZMQReceiver::receiveLoop, this);

        printf("ZMQ connected to %s\n", address_.c_str());
    } catch (const zmq::error_t &e) {
        fprintf(stderr, "ZMQ connection error: %s\n", e.what());
        connected_ = false;
    }
}

void ZMQReceiver::stop() {
    running_ = false;
    if (receiveThread_.joinable()) {
        receiveThread_.join();
    }
    connected_ = false;
}

void ZMQReceiver::receiveLoop() {
    while (running_) {
        try {
            zmq::message_t msg;
            auto result = socket_.recv(msg, zmq::recv_flags::none);

            if (result.has_value() && msg.size() > 0) {
                parseMessage(msg);
            }
        } catch (const zmq::error_t &e) {
            if (e.num() != EAGAIN) { // Ignore timeout errors
                fprintf(stderr, "ZMQ receive error: %s\n", e.what());
            }
        }
    }
}

void ZMQReceiver::parseMessage(const zmq::message_t &msg) {
    if (msg.size() < 4)
        return;

    const uint8_t *data = static_cast<const uint8_t *>(msg.data());
    char msgType[5] = {0};
    std::memcpy(msgType, data, 4);

    if (std::strcmp(msgType, "RAYS") == 0) {
        parseRays(data, msg.size());
    } else if (std::strcmp(msgType, "PCLD") == 0) {
        parsePointCloud(data, msg.size());
    } else if (std::strcmp(msgType, "IMUS") == 0) {
        parseIMU(data, msg.size());
    } else if (std::strcmp(msgType, "MTRS") == 0) {
        parseMotors(data, msg.size());
    } else if (std::strcmp(msgType, "FOOT") == 0) {
        parseFootForce(data, msg.size());
    } else if (std::strcmp(msgType, "BATT") == 0) {
        parseBattery(data, msg.size());
    } else if (std::strcmp(msgType, "CTRL") == 0) {
        parseControl(data, msg.size());
    }
}

void ZMQReceiver::parseRays(const uint8_t *data, size_t size) {
    if (size < 8)
        return;

    uint32_t numRays;
    std::memcpy(&numRays, data + 4, sizeof(uint32_t));

    if (size < 8 + numRays * sizeof(float))
        return;

    std::lock_guard<std::mutex> lock(dataMutex_);
    estimatedRays_.resize(numRays);
    std::memcpy(estimatedRays_.data(), data + 8, numRays * sizeof(float));
}

void ZMQReceiver::parsePointCloud(const uint8_t *data, size_t size) {
    if (size < 8)
        return;

    uint32_t numPoints;
    std::memcpy(&numPoints, data + 4, sizeof(uint32_t));

    // Support both old (XYZ = 3 floats) and new (XYZI = 4 floats) formats
    size_t expectedSizeXYZI = 8 + numPoints * 4 * sizeof(float);
    size_t expectedSizeXYZ = 8 + numPoints * 3 * sizeof(float);

    std::lock_guard<std::mutex> lock(dataMutex_);

    if (size >= expectedSizeXYZI) {
        // New format: XYZI (4 floats per point)
        pointCloud_.resize(numPoints * 4);
        std::memcpy(pointCloud_.data(), data + 8, numPoints * 4 * sizeof(float));
        pointCloudHasIntensity_ = true;
    } else if (size >= expectedSizeXYZ) {
        // Old format: XYZ only (3 floats per point) - add default intensity
        pointCloud_.resize(numPoints * 4);
        const float *srcData = reinterpret_cast<const float *>(data + 8);
        for (uint32_t i = 0; i < numPoints; i++) {
            pointCloud_[i * 4 + 0] = srcData[i * 3 + 0];
            pointCloud_[i * 4 + 1] = srcData[i * 3 + 1];
            pointCloud_[i * 4 + 2] = srcData[i * 3 + 2];
            pointCloud_[i * 4 + 3] = 0.5f; // Default intensity
        }
        pointCloudHasIntensity_ = false;
    }
}

void ZMQReceiver::parseIMU(const uint8_t *data, size_t size) {
    // Expected: 4 (header) + 16 floats + 1 byte = 69 bytes
    if (size < 69)
        return;

    std::lock_guard<std::mutex> lock(dataMutex_);

    size_t offset = 4;

    // Quaternion (4 floats)
    std::memcpy(imuData_.quaternion, data + offset, 4 * sizeof(float));
    offset += 4 * sizeof(float);

    // Gyroscope (3 floats)
    std::memcpy(imuData_.gyroscope, data + offset, 3 * sizeof(float));
    offset += 3 * sizeof(float);

    // Accelerometer (3 floats)
    std::memcpy(imuData_.accelerometer, data + offset, 3 * sizeof(float));
    offset += 3 * sizeof(float);

    // RPY (3 floats)
    std::memcpy(imuData_.rpy, data + offset, 3 * sizeof(float));
    offset += 3 * sizeof(float);

    // Gravity (3 floats)
    std::memcpy(imuData_.gravity, data + offset, 3 * sizeof(float));
    offset += 3 * sizeof(float);

    // Temperature (1 byte)
    imuData_.temperature = data[offset];
}

void ZMQReceiver::parseMotors(const uint8_t *data, size_t size) {
    // Expected: 4 (header) + 12 * 22 bytes = 268 bytes
    const int numMotors = 12;
    const int bytesPerMotor = 22;

    if (size < 4 + numMotors * bytesPerMotor)
        return;

    std::lock_guard<std::mutex> lock(dataMutex_);

    size_t offset = 4;
    for (int i = 0; i < numMotors; i++) {
        motorData_[i].mode = data[offset++];

        std::memcpy(&motorData_[i].q, data + offset, sizeof(float));
        offset += sizeof(float);

        std::memcpy(&motorData_[i].dq, data + offset, sizeof(float));
        offset += sizeof(float);

        std::memcpy(&motorData_[i].ddq, data + offset, sizeof(float));
        offset += sizeof(float);

        std::memcpy(&motorData_[i].tau_est, data + offset, sizeof(float));
        offset += sizeof(float);

        motorData_[i].temperature = data[offset++];

        std::memcpy(&motorData_[i].lost, data + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
    }
}

void ZMQReceiver::parseFootForce(const uint8_t *data, size_t size) {
    // Expected: 4 (header) + 4 * 2 bytes = 12 bytes
    if (size < 12)
        return;

    std::lock_guard<std::mutex> lock(dataMutex_);

    size_t offset = 4;
    for (int i = 0; i < 4; i++) {
        std::memcpy(&footForce_[i], data + offset, sizeof(int16_t));
        offset += sizeof(int16_t);
    }
}

void ZMQReceiver::parseBattery(const uint8_t *data, size_t size) {
    // Expected: 4 + 1 + 1 + 1 + 1 + 4 + 2 + 2 + 2 + 30 = 48 bytes
    if (size < 48)
        return;

    std::lock_guard<std::mutex> lock(dataMutex_);

    size_t offset = 4;

    batteryData_.version_high = data[offset++];
    batteryData_.version_low = data[offset++];
    batteryData_.status = data[offset++];
    batteryData_.soc = data[offset++];

    std::memcpy(&batteryData_.current, data + offset, sizeof(int32_t));
    offset += sizeof(int32_t);

    std::memcpy(&batteryData_.cycle, data + offset, sizeof(uint16_t));
    offset += sizeof(uint16_t);

    batteryData_.bq_ntc[0] = data[offset++];
    batteryData_.bq_ntc[1] = data[offset++];

    batteryData_.mcu_ntc[0] = data[offset++];
    batteryData_.mcu_ntc[1] = data[offset++];

    for (int i = 0; i < 15; i++) {
        std::memcpy(&batteryData_.cell_vol[i], data + offset, sizeof(uint16_t));
        offset += sizeof(uint16_t);
    }
}

void ZMQReceiver::parseControl(const uint8_t *data, size_t size) {
    // Expected: 4 + 1 + 6 * 4 = 29 bytes
    if (size < 29)
        return;

    std::lock_guard<std::mutex> lock(dataMutex_);

    size_t offset = 4;

    controlData_.mode = data[offset++];

    for (int i = 0; i < 3; i++) {
        std::memcpy(&controlData_.cmd[i], data + offset, sizeof(float));
        offset += sizeof(float);
    }

    for (int i = 0; i < 3; i++) {
        std::memcpy(&controlData_.nav_cmd[i], data + offset, sizeof(float));
        offset += sizeof(float);
    }
}

// Thread-safe getters
std::vector<float> ZMQReceiver::getEstimatedRays() {
    std::lock_guard<std::mutex> lock(dataMutex_);
    return estimatedRays_;
}

std::vector<float> ZMQReceiver::getPointCloud() {
    std::lock_guard<std::mutex> lock(dataMutex_);
    return pointCloud_;
}

bool ZMQReceiver::pointCloudHasIntensity() {
    std::lock_guard<std::mutex> lock(dataMutex_);
    return pointCloudHasIntensity_;
}

IMUData ZMQReceiver::getIMUData() {
    std::lock_guard<std::mutex> lock(dataMutex_);
    return imuData_;
}

std::vector<MotorData> ZMQReceiver::getMotorData() {
    std::lock_guard<std::mutex> lock(dataMutex_);
    return motorData_;
}

int16_t ZMQReceiver::getFootForce(int index) {
    std::lock_guard<std::mutex> lock(dataMutex_);
    if (index >= 0 && index < 4) {
        return footForce_[index];
    }
    return 0;
}

BatteryData ZMQReceiver::getBatteryData() {
    std::lock_guard<std::mutex> lock(dataMutex_);
    return batteryData_;
}

ControlData ZMQReceiver::getControlData() {
    std::lock_guard<std::mutex> lock(dataMutex_);
    return controlData_;
}
