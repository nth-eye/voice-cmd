#include <TensorFlowLite.h>
#include <Arduino.h>

#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/version.h>

#include "audio_provider.h"
#include "feature_provider.h"
#include "model.h"
#include "recognizer.h"
#include "voice_cmd.h"

namespace {

events::EventQueue event_queue{32 * EVENTS_EVENT_SIZE};

Array<int8_t, FEATURE_ELEMENT_COUNT> feature_buffer = {};
const auto model = tflite::GetModel(g_model);
const auto feature_provider = FeatureProvider(feature_buffer);
auto recognizer = Recognizer();

mbed::DigitalOut LED(digitalPinToPinName(LED_BUILTIN), LOW);
mbed::DigitalOut LED_R(digitalPinToPinName(LEDR), HIGH);
mbed::DigitalOut LED_G(digitalPinToPinName(LEDG), HIGH);
mbed::DigitalOut LED_B(digitalPinToPinName(LEDB), HIGH);

} // namespace

void schedule_ble_events(BLE::OnEventsToProcessCallbackContext *context) 
{
    event_queue.call(mbed::Callback<void()>(&context->ble, &BLE::processEvents));
}

VoiceCmdService::VoiceCmdService(BLE &ble_) : ble(ble_)
{
    MBED_ASSERT(cmd < N_LABELS);
    GattCharacteristic *char_table[] = { &characteristic };
    GattService vl_service(
        UUID_SERVICE,
        char_table,
        sizeof(char_table) / sizeof(GattCharacteristic *)
    );
    ble.gattServer().addService(vl_service);
}

void VoiceCmdService::update_command(uint8_t cmd_)
{
    MBED_ASSERT(cmd_ < N_LABELS);
    cmd = cmd_;
    ble.gattServer().write(characteristic.getValueHandle(), &cmd, 1);
}

void VoiceCmd::start() 
{
    using namespace std::chrono;

    ble.gap().setEventHandler(this);
    ble.init(this, &VoiceCmd::on_init);

    respond_event = event_queue.call_every(1s, this, &VoiceCmd::waiting_blink);
    event_queue.dispatch_forever();
}

void VoiceCmd::on_init(BLE::InitializationCompleteCallbackContext *params) 
{
    if (params->error != BLE_ERROR_NONE) {
        printf("Ble initialization failed\r\n");
        return;
    }
    // Check the model's version compatibility.
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model provided is schema version %lu not equal to supported version %d\r\n",
            model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    // Static reporter to initialize interpreter.
    static tflite::MicroErrorReporter static_reporter;
    // Create an area of memory to use for input, output, and intermediate arrays.
    static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
    // Pull in only the operation implementations we need.
    // This relies on a complete list of all the ops needed by this graph.
    // An easier approach is to just use the AllOpsResolver, but this will
    // incur some penalty in code space for op implementations that are not
    // needed by this graph.
    static tflite::MicroMutableOpResolver<4> micro_op_resolver(&static_reporter);
    // if (micro_op_resolver.AddConv2D() != kTfLiteOk) return;
    if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) return;
    if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) return;
    if (micro_op_resolver.AddSoftmax() != kTfLiteOk) return;
    if (micro_op_resolver.AddReshape() != kTfLiteOk) return;
    // static tflite::AllOpsResolver micro_op_resolver;
    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE, &static_reporter);
    interpreter = &static_interpreter;
    // Allocate memory from the tensor_arena for the model's tensors.
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors() failed\r\n");
        return;
    }
    TfLiteTensor *model_input = nullptr;
    // Get information about the memory area to use for the model's input.
    model_input = interpreter->input(0);
    if (model_input->dims->size != 2 || 
        model_input->dims->data[0] != 1 ||
        model_input->dims->data[1] != FEATURE_SLICE_COUNT * FEATURE_SLICE_SIZE ||
        model_input->type != kTfLiteInt8) 
    {
        printf("Bad input tensor parameters in model\r\n");
        return;
    }
    model_input_buffer = model_input->data.int8;
    previous_time = 0;

    if (init_micro_features() != kTfLiteOk) {
        printf("init_micro_features() failed\r\n");
        return;
    }

    if (init_audio_recording() != kTfLiteOk) {
        printf("init_audio_recording() failed\r\n");
        return;
    }
    print_mac_address();
    start_advertising();
}

void VoiceCmd::start_advertising() 
{
    // Create advertising parameters and payload
    ble::AdvertisingParameters adv_parameters(
        ble::advertising_type_t::CONNECTABLE_UNDIRECTED,
        ble::adv_interval_t(ble::millisecond_t(1000))
    );

    adv_data_builder.setFlags();
    adv_data_builder.setAppearance(ble::adv_data_appearance_t::UNKNOWN);
    adv_data_builder.setLocalServiceList(mbed::make_Span(&uuid, 1));
    adv_data_builder.setName(DEVICE_NAME);

    // Setup advertising
    ble_error_t error = ble.gap().setAdvertisingParameters(
        ble::LEGACY_ADVERTISING_HANDLE,
        adv_parameters
    );

    if (error) {
        printf("ble.gap().setAdvertisingParameters() failed\r\n");
        return;
    }

    error = ble.gap().setAdvertisingPayload(
        ble::LEGACY_ADVERTISING_HANDLE,
        adv_data_builder.getAdvertisingData()
    );

    if (error) {
        printf("ble.gap().setAdvertisingPayload() failed\r\n");
        return;
    }

    // Start advertising
    error = ble.gap().startAdvertising(ble::LEGACY_ADVERTISING_HANDLE);

    if (error) {
        printf("ble.gap().startAdvertising() failed\r\n");
        return;
    }
}

void VoiceCmd::inference() 
{
    // Fetch the spectrogram for the current time.
    const auto current_time = get_latest_audio_timestamp();
    const auto num_new_slices = feature_provider.populate_feature_data(previous_time, current_time);

    if (num_new_slices == -1) {
        printf("FeatureProvider::populate_feature_data() failed\r\n");
        return;
    }
    previous_time = current_time;
    // If no new audio samples have been received since last time, don't bother.
    if (!num_new_slices) 
        return;

    // Copy feature buffer to input tensor
    for (size_t i = 0; i < FEATURE_ELEMENT_COUNT; i++)
        model_input_buffer[i] = feature_buffer[i];

    // Run the model on the spectrogram input and make sure it succeeds.
    if (interpreter->Invoke() != kTfLiteOk) {
        printf("Invoke() failed\r\n");
        return;
    }

    // Obtain a pointer to the output tensor
    TfLiteTensor *output = interpreter->output(0);
    // Determine whether a command was recognized based on the output of inference
    TfLiteStatus process_status = kTfLiteOk;
    Command cmd = recognizer.process_results(*output, current_time, process_status);

    if (process_status != kTfLiteOk) {
        printf("RecognizeCommands::process_results() failed\r\n");
        return;
    }
    
    // printf("CMD: %u [%u] %u \r\n", cmd.found_command, cmd.score, cmd.is_new);
    respond(current_time, cmd);
}

void VoiceCmd::respond(int32_t current_time, const Command &cmd) 
{
    static int32_t last_cmd_time = 0;

    if (cmd.is_new && last_cmd_time < current_time - 1500) {

        printf("Heard %s [%d] %ld ms\n", LABELS[cmd.found_command], cmd.score, current_time);

        LED = LOW;
        LED_R = LED_G = LED_B = HIGH;

        // If we hear a command, light up the appropriate LED
        switch (cmd.found_command) {
            case SILENCE:   break;
            case UNKNOWN:   LED_B = LOW; break;	// Blue for unknown
            case ON:        LED_G = LOW; break;	// Green for on
            case OFF:       LED_R = LOW; break;	// Red for off
        }
        if (cmd.found_command != SILENCE) 
            last_cmd_time = current_time;

        service.update_command(cmd.found_command);
    }
    // If last_command_time is non-zero but was > 3 seconds ago, zero it and switch off the LED.
    if (last_cmd_time) {
        if (last_cmd_time < current_time - 1500) {
            last_cmd_time = 0;
            LED = LOW;
            LED_R = LED_G = LED_B = HIGH;
        }
        // If it is non-zero but < 3 seconds ago, do nothing.
        return;
    }
    // Otherwise, toggle the LED every time an inference is performed.
    LED = !LED;
}

void VoiceCmd::waiting_blink()
{
    LED_R = LED_G = LED_B = !LED_R;
}

void VoiceCmd::onDisconnectionComplete(const ble::DisconnectionCompleteEvent&)
{
    using namespace std::chrono;

    LED = LOW;
    ble.gap().startAdvertising(ble::LEGACY_ADVERTISING_HANDLE);
    event_queue.cancel(respond_event);
    respond_event = event_queue.call_every(1s, this, &VoiceCmd::waiting_blink);
}

void VoiceCmd::onConnectionComplete(const ble::ConnectionCompleteEvent &event)
{
    using namespace std::chrono;

    LED_R = LED_G = LED_B = HIGH;
    event_queue.cancel(respond_event);
    if (event.getStatus() == BLE_ERROR_NONE)
        respond_event = event_queue.call_every(200ms, this, &VoiceCmd::inference);
}

void VoiceCmd::print_mac_address()
{
    // Print out device MAC address to the console
    ble::own_address_type_t addr_type;
    ble::address_t addr;
    BLE::Instance().gap().getAddress(addr_type, addr);
    printf("DEVICE MAC ADDRESS: %02x:%02x:%02x:%02x:%02x:%02x\n",
            addr[5], addr[4], addr[3], addr[2], addr[1], addr[0]);
}