#include <events/mbed_events.h>
#include <mbed.h>
#include <ble/BLE.h>
#include <ble/Gap.h>

#include <tensorflow/lite/micro/micro_interpreter.h>

#include "model_settings.h"
#include "misc.h"

#pragma once

constexpr const char *DEVICE_NAME = "Arduino_33";
constexpr const char *UUID_SERVICE = "e31e5d86-e4ca-457b-88b5-0b55ed1940cb";
constexpr const char *UUID_CHAR = "e31e5d86-e4ca-457b-88b5-0b55ed1940cc";

void schedule_ble_events(BLE::OnEventsToProcessCallbackContext *context);

// Small custom BLE service to provide voice command characteristic.
class VoiceCmdService {
public:
    VoiceCmdService(BLE &ble_);

    void update_command(uint8_t cmd_);
private:
    BLE &ble;
    uint8_t cmd = SILENCE;
    ReadOnlyGattCharacteristic<uint8_t> characteristic{
        UUID_CHAR,
        &cmd,
        GattCharacteristic::BLE_GATT_CHAR_PROPERTIES_READ | 
        GattCharacteristic::BLE_GATT_CHAR_PROPERTIES_NOTIFY
    };
};

class VoiceCmd : ble::Gap::EventHandler {
public:
    VoiceCmd(BLE &ble_) : ble(ble_) {}

    void start();
private:
    // Callback triggered when the ble initialization process has finished.
    void on_init(BLE::InitializationCompleteCallbackContext *params);
    // Set BLE payload and advertise.
    void start_advertising();
    // Called every time the results of an audio recognition run are available.
    void inference();
    // Blink LED based on the recognized command.
    void respond(int32_t current_time, const Command &cmd);
    // Blinking with RGB when awaiting for connection.
    void waiting_blink();
    // Disconnect callback.
    void onDisconnectionComplete(const ble::DisconnectionCompleteEvent&) override;
    // Connect callback.
    void onConnectionComplete(const ble::ConnectionCompleteEvent &event) override;
    // Print MAC.
    void print_mac_address();

    BLE &ble;
    VoiceCmdService service{ble};
    UUID uuid = UUID_SERVICE;
    int respond_event;

    int32_t previous_time;
    int8_t *model_input_buffer;
    tflite::MicroInterpreter *interpreter;

    uint8_t adv_buffer[ble::LEGACY_ADVERTISING_MAX_SIZE];
    ble::AdvertisingDataBuilder adv_data_builder{adv_buffer};
};
