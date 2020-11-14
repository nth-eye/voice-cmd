#include "voice_cmd.h"

BLE &ble_device = BLE::Instance();
VoiceCmd voice_cmd_detector{ble_device};

// Redirect FileHandles to get printf() working
mbed::FileHandle *mbed::mbed_override_console(int) 
{
    return &Serial;
}

void setup()
{
    Serial.begin(9600);
    ble_device.onEventsToProcess(schedule_ble_events);
    voice_cmd_detector.start();
}

void loop()
{}
