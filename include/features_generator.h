#include <tensorflow/lite/c/common.h>

#pragma once

// Sets up any resources needed for the feature generation pipeline.
TfLiteStatus init_micro_features();

// Converts audio sample data into a more compact form that's appropriate for
// feeding into a neural network.
TfLiteStatus generate_micro_features(
	const int16_t *input, int input_size,
	int output_size, int8_t *output,
	size_t *num_samples_read);