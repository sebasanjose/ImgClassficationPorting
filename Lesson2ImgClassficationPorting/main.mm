#import <Metal/Metal.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

std::vector<std::vector<float>> loadCSV(const std::string& filePath) {
    std::vector<std::vector<float>> dataset;
    std::ifstream file(filePath);
    std::string line;

    // Check if file opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filePath << std::endl;
        return dataset;
    }

    // Skip the header row if it exists
    if (std::getline(file, line)) {
        // Assuming the first row is a header, skip it
    }

    // Process each line in the CSV
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream lineStream(line);
        std::string cell;

        // Read each cell separated by a comma
        while (std::getline(lineStream, cell, ',')) {
            try {
                row.push_back(std::stof(cell));
            } catch (const std::invalid_argument&) {
                std::cerr << "Warning: Non-numeric value found in the dataset, skipping: " << cell << std::endl;
                continue; // Skip non-numeric values
            }
        }

        if (!row.empty()) {
            dataset.push_back(row);
        }
    }

    return dataset;
}


std::vector<float> flattenDataset(const std::vector<std::vector<float>>& dataset) {
            std::vector<float> flattened;
            for (const auto& row : dataset) {
                flattened.insert(flattened.end(), row.begin(), row.end());
            }
            return flattened;
        }




int main(int argc, const char * argv[]) {
    @autoreleasepool {
        // Set up Metal device and command queue
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal is not supported on this device." << std::endl;
            return -1;
        }
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        
        // Load the compiled Metal shader
        NSError* error = nil;
        id<MTLLibrary> library = [device newDefaultLibrary];
        if (!library) {
            std::cerr << "Failed to load default Metal library." << std::endl;
            return -1;
        }
        
        // Create a compute pipeline state for the convolution kernel
        id<MTLFunction> convolutionFunction = [library newFunctionWithName:@"convolution_kernel"];
        if (!convolutionFunction) {
            std::cerr << "Failed to find the convolution kernel function." << std::endl;
            return -1;
        }
        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:convolutionFunction error:&error];
        if (!pipelineState) {
            std::cerr << "Failed to create pipeline state: " << [[error localizedDescription] UTF8String] << std::endl;
            return -1;
        }
        
        // Set up Metal buffers (example sizes and data for input, filter, output)
        const uint inputWidth = 28;
        const uint inputHeight = 28;
        const uint filterWidth = 3;
        const uint filterHeight = 3;
        const uint outputWidth = 28;
        const uint outputHeight = 28;
        
        std::vector<float> input(inputWidth * inputHeight, 1.0f); // Example input data
        std::vector<float> filter(filterWidth * filterHeight, 0.1f); // Example filter
        std::vector<float> output(outputWidth * outputHeight, 0.0f); // Output buffer
        
        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input.data() length:input.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> filterBuffer = [device newBufferWithBytes:filter.data() length:filter.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [device newBufferWithBytes:output.data() length:output.size() * sizeof(float) options:MTLResourceStorageModeShared];
        
        // Create a command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:pipelineState];
        
        // Set buffers for the compute encoder
        [computeEncoder setBuffer:inputBuffer offset:0 atIndex:0];
        [computeEncoder setBuffer:filterBuffer offset:0 atIndex:1];
        [computeEncoder setBuffer:outputBuffer offset:0 atIndex:2];
        [computeEncoder setBytes:&inputWidth length:sizeof(inputWidth) atIndex:3];
        [computeEncoder setBytes:&inputHeight length:sizeof(inputHeight) atIndex:4];
        [computeEncoder setBytes:&filterWidth length:sizeof(filterWidth) atIndex:5];
        [computeEncoder setBytes:&filterHeight length:sizeof(filterHeight) atIndex:6];
        
        // Set thread groups and grid sizes
        MTLSize gridSize = MTLSizeMake(outputWidth, outputHeight, 1);
        NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
        
        // Dispatch the compute kernel
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        
        // End encoding and commit the command buffer
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Read back the output
        float* outputData = (float*)outputBuffer.contents;
        std::cout << "Output data: " << std::endl;
        for (uint i = 0; i < outputWidth * outputHeight; i++) {
            std::cout << outputData[i] << " ";
            if ((i + 1) % outputWidth == 0) {
                std::cout << std::endl;
            }
        }
    }
    return 0;
}
