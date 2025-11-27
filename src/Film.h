#pragma once
#include "long_march.h"

// Film class for accumulating ray tracing samples over time
// Used for progressive rendering when camera is stationary
class Film {
public:
    Film(grassland::graphics::Core* core, int width, int height);
    ~Film();

    // Reset accumulation (call when camera moves or scene changes)
    void Reset();

    // Get the accumulated color image (for display)
    grassland::graphics::Image* GetAccumulatedColorImage() const { return accumulated_color_image_.get(); }
    
    // Get the sample count image (for shader)
    grassland::graphics::Image* GetAccumulatedSamplesImage() const { return accumulated_samples_image_.get(); }
    
    // Get the final output image (averaged result)
    grassland::graphics::Image* GetOutputImage() const { return output_image_.get(); }

    // Get current sample count
    int GetSampleCount() const { return sample_count_; }

    // Increment sample count
    void IncrementSampleCount() { sample_count_++; }

    // Convert accumulated data to final output image (divide by sample count)
    void DevelopToOutput();

    // Resize the film (call when window resizes)
    void Resize(int width, int height);

    int GetWidth() const { return width_; }
    int GetHeight() const { return height_; }

private:
    grassland::graphics::Core* core_;
    int width_;
    int height_;
    int sample_count_; // Number of accumulated samples

    // Accumulated color (sum of all samples)
    std::unique_ptr<grassland::graphics::Image> accumulated_color_image_;
    
    // Accumulated sample count per pixel
    std::unique_ptr<grassland::graphics::Image> accumulated_samples_image_;
    
    // Final output image (accumulated_color / accumulated_samples)
    std::unique_ptr<grassland::graphics::Image> output_image_;

    void CreateImages();
};

