#include "features_generator.h"
#include "audio_provider.h"
#include "misc.h"

#pragma once

class FeatureProvider {
public:
    FeatureProvider(Array<int8_t, FEATURE_ELEMENT_COUNT> &feature_data_) 
        : feature_data(feature_data_) 
    {}

    int populate_feature_data(
        int32_t last_time_in_ms, 
        int32_t time_in_ms) const;
private:
    void shift_slices(const size_t slices_to_keep) const;
    
    Array<int8_t, FEATURE_ELEMENT_COUNT> &feature_data;
};