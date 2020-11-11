#include <tensorflow/lite/c/common.h>

#pragma once

TfLiteStatus init_audio_recording();

// Expected to return 16-bit PCM sample data for a given point in time. The
// sample data itself should be used as quickly as possible by the caller, since
// to allow memory optimizations there are no guarantees that the samples won't
// be overwritten by new data in the future. In practice, implementations should
// ensure that there's a reasonable time allowed for clients to access the data
// before any reuse.
TfLiteStatus get_audio_samples(
    size_t start_ms, 
    size_t duration_ms,
    size_t &audio_samples_size, 
    int16_t **audio_samples);

// Returns the time that audio data was last captured in milliseconds. There's
// no contract about what time zero represents, the accuracy, or the granularity
// of the result. Subsequent calls will generally not return a lower value, but
// even that's not guaranteed if there's an overflow wraparound.
int32_t get_latest_audio_timestamp();