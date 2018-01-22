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
 * Helper function to store Integer Array on GPU and CPU side
 * @author Nicolas Chiaruttini
 */
public class IntGpuArray extends GpuArray<int[]>{
    
    public IntGpuArray() {
        this(defaultAllocatedSize);
    }
    
    public IntGpuArray(int iniSize) {
        this.sizeOfObject=Sizeof.INT;
        gpuArray = new Pointer();
        ensureAllocatedSize(iniSize, false, false);
    }  
    
    
    @Override
    public void push() {
        assert allocatedSize>=length;
        cudaMemcpy(gpuArray, Pointer.to(cpuArray), sizeOfObject*length, cudaMemcpyHostToDevice);
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
            int[] newArray = new int[allocatedSize];
            System.arraycopy(cpuArray, 0, newArray, 0, length);
            cpuArray = newArray;
        } else {
            cpuArray = new int[allocatedSize];
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
