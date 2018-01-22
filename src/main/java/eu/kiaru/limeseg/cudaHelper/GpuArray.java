package eu.kiaru.limeseg.cudaHelper;

import jcuda.Pointer;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMemset;

/**
 * Helper function to store Generic Array on GPU and CPU side
 * @author Nicolas Chiaruttini
 */
abstract public class GpuArray<T> {
    static float MARGIN_MEM_SUP=1.5f;
    static float MARGIN_MEM_INF=6f;
    int sizeOfObject;
    public int length;
    int allocatedSize;
    static int defaultAllocatedSize = 10;
    
    public static int CPU_AND_GPU=0;
    public static int GPU_ONLY=1;
    
    public T cpuArray;
    public Pointer gpuArray; 
    
    abstract public void push();
    abstract public void pop();
    abstract public void ensureAllocatedSize(int nObjects, boolean keepGPUData, boolean keepCPUData);
    
    public void setGpuMemToZero() {
        cudaMemset(gpuArray,0,length*sizeOfObject); 
    }
    
    public void freeMem() {
        cpuArray=null;
        cudaFree(gpuArray);
        gpuArray = new Pointer(); // took me ages to find out this bug
        allocatedSize=-1;
        length=-1;
    }
}
