#include <cstddef>
#include <cstdint>

#pragma once

// The size of the input time series data we pass to the FFT to produce the
// frequency information. This has to be a power of two, and since we're dealing
// with 30ms of 16KHz inputs, which means 480 samples, this is the next value.
constexpr size_t MAX_AUDIO_SAMPLE_SIZE = 512;
constexpr size_t AUDIO_SAMPLE_FREQUENCY = 16000;

// The following values are derived from values used during model training.
constexpr size_t FEATURE_SLICE_SIZE = 40;
constexpr size_t FEATURE_SLICE_COUNT = 49;
constexpr size_t FEATURE_ELEMENT_COUNT = FEATURE_SLICE_SIZE * FEATURE_SLICE_COUNT;
constexpr size_t FEATURE_SLICE_STRIDE_MS = 20;
constexpr size_t FEATURE_SLICE_DURATION_MS = 30;

// The size of this will depend on the model you're using, and may need to be determined by experimentation.
constexpr size_t TENSOR_ARENA_SIZE = 126 * 1024; // ~144 is max before mbed crash and 126 is max before ble crash

enum : uint8_t {
    SILENCE,
    UNKNOWN,
    ON,
    OFF,
    N_LABELS // Don't modify, leave at the end of the enum.
};

constexpr const char *LABELS[N_LABELS] = {
    "silence",
    "unknown",
    "on",
    "off"
};
