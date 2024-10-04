// metal_helpers.m

#import "metal_helpers.h"

MetalContext* metal_init(const char* shader_filename) {
    MetalContext* ctx = (MetalContext*)malloc(sizeof(MetalContext));
    ctx->device = MTLCreateSystemDefaultDevice();
    ctx->commandQueue = [ctx->device newCommandQueue];

    NSError* error = nil;

    // Load the Metal shader
    NSString* shaderPath = [NSString stringWithUTF8String:shader_filename];
    NSString* shaderSource = [NSString stringWithContentsOfFile:shaderPath encoding:NSUTF8StringEncoding error:&error];
    if (!shaderSource) {
        NSLog(@"Failed to load shader file: %@", error.localizedDescription);
        free(ctx);
        return NULL;
    }

    ctx->library = [ctx->device newLibraryWithSource:shaderSource options:nil error:&error];
    if (!ctx->library) {
        NSLog(@"Failed to compile shader library: %@", error.localizedDescription);
        free(ctx);
        return NULL;
    }

    // Create compute pipeline states
    id<MTLFunction> computeFunctionLoss = [ctx->library newFunctionWithName:@"compute_loss_and_update_weights"];
    ctx->computePipelineStateLoss = [ctx->device newComputePipelineStateWithFunction:computeFunctionLoss error:&error];
    if (!ctx->computePipelineStateLoss) {
        NSLog(@"Failed to create compute pipeline state for loss computation: %@", error.localizedDescription);
        free(ctx);
        return NULL;
    }

    id<MTLFunction> computeFunctionNextPixel = [ctx->library newFunctionWithName:@"choose_next_pixel"];
    ctx->computePipelineStateNextPixel = [ctx->device newComputePipelineStateWithFunction:computeFunctionNextPixel error:&error];
    if (!ctx->computePipelineStateNextPixel) {
        NSLog(@"Failed to create compute pipeline state for next pixel selection: %@", error.localizedDescription);
        free(ctx);
        return NULL;
    }

    return ctx;
}

void metal_cleanup(MetalContext* ctx) {
    // Release Metal objects
    ctx->computePipelineStateLoss = nil;
    ctx->computePipelineStateNextPixel = nil;
    ctx->library = nil;
    ctx->commandQueue = nil;
    ctx->device = nil;
    free(ctx);
}

double metal_compute_loss_and_update_weights(MetalContext* ctx, MetalComputeParams* params) {
    id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:ctx->computePipelineStateLoss];

    // Create buffers
    id<MTLBuffer> paramsBuffer = [ctx->device newBufferWithBytes:params length:sizeof(MetalComputeParams) options:MTLResourceStorageModeShared];
    double loss = 0.0;
    id<MTLBuffer> lossBuffer = [ctx->device newBufferWithBytes:&loss length:sizeof(double) options:MTLResourceStorageModeShared];

    [computeEncoder setBuffer:paramsBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:lossBuffer offset:0 atIndex:1];

    MTLSize gridSize = MTLSizeMake(1, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(1, 1, 1);

    [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];

    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Get the result
    double* lossPtr = (double*)[lossBuffer contents];
    loss = *lossPtr;

    return loss;
}

int metal_choose_next_pixel(MetalContext* ctx, MetalComputeParams* params) {
    id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:ctx->computePipelineStateNextPixel];

    // Create buffers
    id<MTLBuffer> paramsBuffer = [ctx->device newBufferWithBytes:params length:sizeof(MetalComputeParams) options:MTLResourceStorageModeShared];
    int next_idx = -1;
    id<MTLBuffer> nextIdxBuffer = [ctx->device newBufferWithBytes:&next_idx length:sizeof(int) options:MTLResourceStorageModeShared];

    [computeEncoder setBuffer:paramsBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:nextIdxBuffer offset:0 atIndex:1];

    MTLSize gridSize = MTLSizeMake(params->num_pixels, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(64, 1, 1); // Adjust as needed

    [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];

    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Get the result
    int* nextIdxPtr = (int*)[nextIdxBuffer contents];
    next_idx = *nextIdxPtr;

    return next_idx;
}