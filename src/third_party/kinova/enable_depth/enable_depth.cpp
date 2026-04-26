#include <iostream>
#include "RouterClient.h"
#include "TransportClientTcp.h"
#include "SessionManager.h"
#include "VisionConfigClientRpc.h"
#include "DeviceManagerClientRpc.h"
#include "SessionClientRpc.h"
#include "Common.pb.h"

namespace k_api = Kinova::Api;

int main(int argc, char* argv[]) {
    const std::string ip       = (argc > 1) ? argv[1] : "192.168.1.10";
    const std::string username = (argc > 2) ? argv[2] : "admin";
    const std::string password = (argc > 3) ? argv[3] : "admin";
    const int port = 10000;

    k_api::TransportClientTcp transport;
    k_api::RouterClient router(&transport, [](k_api::KError e) {
        std::cerr << "Router error: " << e.toString() << std::endl;
    });

    transport.connect(ip, port);

    auto session_info = k_api::Session::CreateSessionInfo();
    session_info.set_username(username);
    session_info.set_password(password);
    session_info.set_session_inactivity_timeout(60000);
    session_info.set_connection_inactivity_timeout(2000);

    k_api::SessionManager session_manager(&router);
    session_manager.CreateSession(session_info);
    std::cout << "Connected to " << ip << std::endl;

    // Find the vision module device ID
    k_api::DeviceManager::DeviceManagerClient device_manager(&router);
    auto devices = device_manager.ReadAllDevices();
    uint32_t vision_device_id = 0;
    for (const auto& handle : devices.device_handle()) {
        std::cout << "Device id=" << handle.device_identifier()
                  << " type=" << handle.device_type() << std::endl;
        if (handle.device_type() == k_api::Common::VISION) {
            vision_device_id = handle.device_identifier();
        }
    }

    if (vision_device_id == 0) {
        std::cerr << "No vision module found!" << std::endl;
        session_manager.CloseSession();
        transport.disconnect();
        return 1;
    }
    std::cout << "Vision module device_id=" << vision_device_id << std::endl;

    k_api::VisionConfig::VisionConfigClient vision(&router);

    // Read current depth settings
    k_api::VisionConfig::SensorIdentifier id;
    id.set_sensor(k_api::VisionConfig::SENSOR_DEPTH);
    auto current = vision.GetSensorSettings(id, vision_device_id);
    std::cout << "Current depth: resolution=" << current.resolution()
              << " fps=" << current.frame_rate()
              << " bitrate=" << current.bit_rate() << std::endl;

    // Also read color sensor settings for comparison
    k_api::VisionConfig::SensorIdentifier color_id;
    color_id.set_sensor(k_api::VisionConfig::SENSOR_COLOR);
    auto color = vision.GetSensorSettings(color_id, vision_device_id);
    std::cout << "Current color: resolution=" << color.resolution()
              << " fps=" << color.frame_rate()
              << " bitrate=" << color.bit_rate() << std::endl;

    session_manager.CloseSession();
    transport.disconnect();
    return 0;
}
