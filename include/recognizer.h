#include <tensorflow/lite/c/common.h>

#include <limits>
#include "misc.h"

#pragma once

// Primitive decoding model for results from audio recognition model on a single window of samples.
class Recognizer {
public:
    Command process_results(
        const TfLiteTensor &latest_results, 
        const int32_t current_time_ms, 
        TfLiteStatus &status);
private:
    // Calculate the average score across all the results in the window.
    Array<int32_t, N_LABELS> calculate_average();
    size_t find_highest(const Array<int32_t, N_LABELS> &scores);

    static constexpr int32_t avg_window_duration_ms = 1000;
    static constexpr int32_t suppression_ms = 1500;
    static constexpr int32_t min_count = 3;
    static const uint8_t thresholds[N_LABELS];

    RingBuf<Result, FEATURE_SLICE_COUNT + 1> prev_results;
    uint8_t prev_top_idx = SILENCE;
    int32_t prev_top_time = std::numeric_limits<int32_t>::max(); // FIXME min()
};

