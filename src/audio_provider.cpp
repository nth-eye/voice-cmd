#include "audio_provider.h"
#include "model_settings.h"
#include "misc.h"
#include "PDM.h"

namespace {

constexpr size_t capture_buffer_size = DEFAULT_PDM_BUFFER_SIZE * 16;
constexpr size_t ring_buffer_mask = capture_buffer_size - 1;
// An internal buffer able to fit 16x our sample size.
int16_t capture_buffer[capture_buffer_size]; // FIXME + DEFAULT_PDM_BUFFER_SIZE

// A buffer that holds our output.
int16_t output_buffer[MAX_AUDIO_SAMPLE_SIZE];
// Mark as volatile so we can check in a while loop to see if any samples have arrived yet.
volatile int32_t latest_audio_timestamp = 0;

}

void callback_pdm()
{
    // Determine the index, in the history of all samples, of the last sample
    const int32_t start_sample_offset = latest_audio_timestamp * (AUDIO_SAMPLE_FREQUENCY / 1000);
    // Determine the index of this sample in our ring buffer
    const size_t capture_idx = start_sample_offset & ring_buffer_mask;
    // Read the data to the correct place in our buffer
    PDM.read(capture_buffer + capture_idx, DEFAULT_PDM_BUFFER_SIZE);
    // Calculate what timestamp the last audio sample represents.
    // This is how we let the outside world know that new audio data has arrived.
    latest_audio_timestamp += (DEFAULT_PDM_BUFFER_SIZE / (AUDIO_SAMPLE_FREQUENCY / 1000));
}

TfLiteStatus init_audio_recording()
{
    PDM.onReceive(callback_pdm);
    PDM.setGain(20);
    // Start listening for audio: MONO @ 16KHz.
    if (!PDM.begin(1, AUDIO_SAMPLE_FREQUENCY)) 
        return kTfLiteError;
    // Block until we have our first audio sample
    while (!latest_audio_timestamp);

    return kTfLiteOk;
}

TfLiteStatus get_audio_samples(
    size_t start_ms, 
    size_t duration_ms,
    size_t &audio_samples_size, 
    int16_t **audio_samples)
{
    // Determine the index, in the history of all samples, of the first sample we want.
    const size_t start_offset = start_ms * (AUDIO_SAMPLE_FREQUENCY / 1000);
    // Determine how many samples we want in total
    const size_t duration_sample_count = duration_ms * (AUDIO_SAMPLE_FREQUENCY / 1000);

    for (size_t i = 0; i < duration_sample_count; ++i) {
        // For each sample, transform its index in the history of all samples into
        // its index in capture_buffer
        const size_t capture_index = (start_offset + i) & ring_buffer_mask;
        // Write the sample to the output buffer
        output_buffer[i] = capture_buffer[capture_index];
    }

    // Set pointers to provide access to the audio
    audio_samples_size = MAX_AUDIO_SAMPLE_SIZE;
    *audio_samples = output_buffer;

    return kTfLiteOk;
}

int32_t get_latest_audio_timestamp() 
{ 
    return latest_audio_timestamp; 
}