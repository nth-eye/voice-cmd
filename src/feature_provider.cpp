#include <mbed.h>

#include "feature_provider.h"
#include "model_settings.h"

void FeatureProvider::shift_slices(const size_t slices_to_keep) const
{
	const size_t slices_to_drop = FEATURE_SLICE_COUNT - slices_to_keep;

	for (size_t dst_slice = 0; dst_slice < slices_to_keep; ++dst_slice) {

		int8_t *dst_slice_data = &feature_data[dst_slice * FEATURE_SLICE_SIZE];
		const int src_slice = dst_slice + slices_to_drop;
		const int8_t *src_slice_data = &feature_data[src_slice * FEATURE_SLICE_SIZE];

		for (size_t i = 0; i < FEATURE_SLICE_SIZE; ++i)
			dst_slice_data[i] = src_slice_data[i];
	}
}

int FeatureProvider::populate_feature_data(int32_t last_time_in_ms, int32_t time_in_ms) const
{
	// 1) Calculate how many time steps we need.
	const int last_step = last_time_in_ms / FEATURE_SLICE_STRIDE_MS;
	const int current_step = time_in_ms / FEATURE_SLICE_STRIDE_MS;
	size_t slices_needed = current_step - last_step;

	if (!last_time_in_ms || slices_needed > FEATURE_SLICE_COUNT)
		slices_needed = FEATURE_SLICE_COUNT;

	// 2) Determine how many slices to keep and then shift appropriately.
	const size_t slices_to_keep = FEATURE_SLICE_COUNT - slices_needed;
	
	if (slices_to_keep > 0) 
        shift_slices(slices_to_keep);

	// 3) 
	if (slices_needed > 0) {
		
		for (size_t new_slice = slices_to_keep; new_slice < FEATURE_SLICE_COUNT; ++new_slice) {

			const int new_step = (current_step - FEATURE_SLICE_COUNT + 1) + new_slice;
			const int32_t slice_start_ms = new_step * FEATURE_SLICE_STRIDE_MS;
			int16_t *audio_samples = nullptr;
			size_t audio_samples_size = 0;

			get_audio_samples(
				slice_start_ms,
				FEATURE_SLICE_DURATION_MS, 
				audio_samples_size,
				&audio_samples);

			if (audio_samples_size < MAX_AUDIO_SAMPLE_SIZE) {
				printf("Audio data size %d too small, want %d \n", audio_samples_size, MAX_AUDIO_SAMPLE_SIZE);
				return -1;
			}

			int8_t *new_slice_data = &feature_data[new_slice * FEATURE_SLICE_SIZE];
			size_t num_samples_read;

			auto generate_status = generate_micro_features(
				audio_samples, 
				audio_samples_size,
				FEATURE_SLICE_SIZE,
				new_slice_data,
				&num_samples_read);

			if (generate_status != kTfLiteOk) 
                return -1;
		}
	}
	return slices_needed;
}