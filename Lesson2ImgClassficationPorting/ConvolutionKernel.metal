#include <metal_stdlib>
using namespace metal;

kernel void convolution_kernel(device float* input [[buffer(0)]],
                               device float* filter [[buffer(1)]],
                               device float* output [[buffer(2)]],
                               uint2 gid [[thread_position_in_grid]],
                               constant uint& inputWidth [[buffer(3)]],
                               constant uint& inputHeight [[buffer(4)]],
                               constant uint& filterWidth [[buffer(5)]],
                               constant uint& filterHeight [[buffer(6)]]) {
    // Calculate the center position of the filter
    int halfFilterWidth = filterWidth / 2;
    int halfFilterHeight = filterHeight / 2;

    // Initialize the output value
    float result = 0.0f;

    // Perform convolution operation
    for (int fy = 0; fy < filterHeight; fy++) {
        for (int fx = 0; fx < filterWidth; fx++) {
            int imgX = gid.x + fx - halfFilterWidth;
            int imgY = gid.y + fy - halfFilterHeight;

            // Check for boundary conditions
            if (imgX >= 0 && imgX < inputWidth && imgY >= 0 && imgY < inputHeight) {
                float imageValue = input[imgY * inputWidth + imgX];
                float filterValue = filter[fy * filterWidth + fx];
                result += imageValue * filterValue;
            }
        }
    }

    // Store the result in the output buffer
    output[gid.y * inputWidth + gid.x] = result;
}


// Metal kernel function for ReLU activation
kernel void relu_activation(device float* data [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
    float value = data[gid.y * gid.x];
    data[gid.y * gid.x] = max(0.0f, value); // Apply ReLU
}


// Metal kernel function for batch normalization
kernel void batch_normalization(device float* data [[buffer(0)]],
                                constant float& mean [[buffer(1)]],
                                constant float& variance [[buffer(2)]],
                                constant float& epsilon [[buffer(3)]],
                                constant float& gamma [[buffer(4)]],
                                constant float& beta [[buffer(5)]],
                                uint2 gid [[thread_position_in_grid]]) {
    float value = data[gid.y * gid.x];
    // Normalize the value
    float normalized = (value - mean) / sqrt(variance + epsilon);
    // Apply scale (gamma) and shift (beta)
    data[gid.y * gid.x] = gamma * normalized + beta;
}


// Metal kernel function for dropout
kernel void dropout(device float* data [[buffer(0)]],
                    constant float& dropProbability [[buffer(1)]],
                    constant uint& seed [[buffer(2)]],
                    uint2 gid [[thread_position_in_grid]]) {
    // Generate a random number
    float randomValue = fract(sin(dot(float2(gid), float2(12.9898, 78.233))) * 43758.5453);
    // Zero out the element based on the drop probability
    data[gid.y * gid.x] = (randomValue < dropProbability) ? 0.0f : data[gid.y * gid.x];
}

// Metal kernel function for max pooling
kernel void max_pooling(const device float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& inputWidth [[buffer(2)]],
                        constant uint& inputHeight [[buffer(3)]],
                        constant uint& poolSize [[buffer(4)]],
                        uint2 gid [[thread_position_in_grid]]) {
    float maxValue = -FLT_MAX;
    for (int py = 0; py < poolSize; py++) {
        for (int px = 0; px < poolSize; px++) {
            int x = gid.x * poolSize + px;
            int y = gid.y * poolSize + py;

            // Ensure we are within bounds
            if (x < inputWidth && y < inputHeight) {
                maxValue = max(maxValue, input[y * inputWidth + x]);
            }
        }
    }

    // Store the maximum value in the output buffer
    output[gid.y * inputWidth / poolSize + gid.x] = maxValue;
}
