package eu.kiaru.limeseg.cudaHelper;

import jcuda.Pointer;
import jcuda.Sizeof;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

/**
 * Helper function to store Float Array on GPU and CPU side
 * @author Nicolas Chiaruttini
 */
public class FloatGpuArray extends GpuArray<float[]> {
    
    public FloatGpuArray() {
        this(defaultAllocatedSize);
    }
    
    public FloatGpuArray(int iniSize) {
        this.sizeOfObject=Sizeof.FLOAT;
        gpuArray = new Pointer();
        ensureAllocatedSize(iniSize, false, false);
    } 
    
    @Override
    public void push() {
        assert allocatedSize>=length;
        cudaMemcpy(this.gpuArray, Pointer.to(this.cpuArray), this.sizeOfObject*length, cudaMemcpyHostToDevice);
    }

    @Override
    public void pop() {
        assert allocatedSize>=length;
        cudaMemcpy(Pointer.to(this.cpuArray), this.gpuArray, this.sizeOfObject*length, cudaMemcpyDeviceToHost);
    }

    @Override
    public void ensureAllocatedSize(int nObjects, boolean keepGPUData, boolean keepCPUData) {
        if (allocatedSize>=nObjects) return;
        // new to reallocate
        allocatedSize = (int)(nObjects*GpuArray.MARGIN_MEM_SUP);        
        if (allocatedSize==0) {allocatedSize=defaultAllocatedSize;}
        if (keepCPUData) {         
            float[] newArray = new float[allocatedSize];
            System.arraycopy(cpuArray, 0, newArray, 0, length);
            cpuArray = newArray;
        } else {
            cpuArray = new float[allocatedSize];
        }
        if (keepGPUData) {
            Pointer newGpuArray = new Pointer();
            cudaMalloc(newGpuArray, allocatedSize*sizeOfObject);
            cudaMemcpy(newGpuArray, gpuArray, sizeOfObject*length, cudaMemcpyDeviceToDevice);
            cudaFree(gpuArray);
            gpuArray = newGpuArray;
        } else {
            cudaFree(gpuArray);
            gpuArray = new Pointer(); // not sure but that may prevent other problems
            cudaMalloc(gpuArray, allocatedSize*sizeOfObject);
        }
    }
    
}
