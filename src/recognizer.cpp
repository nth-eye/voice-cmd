#include <mbed.h>

#include "recognizer.h"

const uint8_t Recognizer::thresholds[N_LABELS] = {200, 215, 180, 180};

Array<int32_t, N_LABELS> Recognizer::calculate_average()
{
    Array<int32_t, N_LABELS> avg_scores;

    for (const auto &it : prev_results) {
        for (size_t i = 0; i < N_LABELS; ++i)
            avg_scores[i] += it.scores[i] + 128;
    }
    
    for (auto &it : avg_scores)
        it /= prev_results.size();

    return avg_scores;
}

Command Recognizer::process_results(
    const TfLiteTensor &latest_results, 
    const int32_t current_time_ms, 
    TfLiteStatus &status)
{
    if (latest_results.dims->size != 2 ||
        latest_results.dims->data[0] != 1 ||
        latest_results.dims->data[1] != N_LABELS) 
    {
        printf("The results for recognition should contain %d elements, but there are %d in an %d-dimensional shape \n",
            N_LABELS, latest_results.dims->data[1], latest_results.dims->size);
        status = kTfLiteError;
    }
    if (latest_results.type != kTfLiteInt8) {
        printf("The results for recognition should be int8 elements, but are %d \n",
            latest_results.type);
        status = kTfLiteError;
    }
    if (!prev_results.empty() && current_time_ms < prev_results.front().time) {
        printf("Results must be fed in increasing time order, but received a timestamp of %ld that was earlier than the previous one of %ld \n",
            current_time_ms, prev_results.front().time);
        status = kTfLiteError;
    }
    if (status != kTfLiteOk) 
        return Command();

    // Add the latest results to the head of the queue.
    prev_results.push_back({current_time_ms, latest_results.data.int8});

    // Prune any earlier results that are too old for the averaging window.
    const int64_t time_limit = current_time_ms - avg_window_duration_ms;

    while (!prev_results.empty() && prev_results.front().time < time_limit)
        prev_results.pop_front();

    // If there are too few results, assume the result will be unreliable and bail.
    const int64_t earliest_time = prev_results.front().time;
    const int64_t samples_duration = current_time_ms - earliest_time;

    if (prev_results.size() < min_count || samples_duration < (avg_window_duration_ms / 4)) {
        // printf("Prev results: %d Samples_duration: %d", prev_results.size() < min_count, samples_duration < (avg_window_duration_ms / 4));
        return {prev_top_idx, 0, false};
    }

    // Calculate the average score across all the results in the window.
    const auto scores = calculate_average();

    // Find the current highest scoring category.
    uint8_t top_index = 0;
    int32_t top_score = 0;

    for (size_t i = 0; i < N_LABELS; ++i) {
        if (scores[i] > top_score) {
            top_score = scores[i];
            top_index = i;
        }
    }
    // If we've recently had another label trigger, assume one that occurs too
    // soon afterwards is a bad result.
    int64_t time_since_last_top = current_time_ms - prev_top_time;

    // if (prev_top_idx == SILENCE || prev_top_time == std::numeric_limits<int32_t>::min())
    // 	time_since_last_top = std::numeric_limits<int32_t>::max();

    bool is_new_command = false;

    if (top_score > thresholds[top_index] && (top_index != prev_top_idx || time_since_last_top > suppression_ms)) {
        prev_top_idx = top_index;
        prev_top_time = current_time_ms;
        is_new_command = true;
    }

    return {top_index, top_score, is_new_command};
}
