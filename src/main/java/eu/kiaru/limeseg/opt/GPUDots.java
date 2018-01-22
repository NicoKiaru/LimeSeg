package eu.kiaru.limeseg.opt;

import java.util.ArrayList;

import eu.kiaru.limeseg.cudaHelper.FloatGpuArray;
import eu.kiaru.limeseg.cudaHelper.IntGpuArray;
import eu.kiaru.limeseg.struct.DotN;

import static java.util.stream.IntStream.range;

/**
 * Helper function to handle lost of dots by GPU
 * @author Nicolas Chiaruttini
 */
class GPUDots {
    
    final public static int
            PX=0,PY=1,PZ=2,
            NX=3,NY=4,NZ=5,
            FX=6,FY=7,FZ=8,
            MX=9,MY=10,MZ=11,
            RFX=12, RFY=13, RFZ=14,
            SUPER_DOT_RADIUS_SQUARED = 15,
            RELAXED=16,
            N_PARAMS_FLOAT=17;
    
    final public static int
          N_NEIGH=0,
          CELL_ID=1,
          HAS_CONVERGED=2,          
          ALL_NEIGHBORS_HAVE_CONVERGED=3,  
          ALL_NEIGHBORS_HAVE_CONVERGED_PREVIOUSLY=4,
          N_PARAMS_INT=5;           
    
    public boolean hasBeenPushed;
    
    IntGpuArray[] iGA_Int;
    FloatGpuArray[] iGA_Float;
    
    int numberOfDots;
    
    public GPUDots() {
        hasBeenPushed=false;
        numberOfDots=0;
        // Initialisation of gpu arrays
        iGA_Float = new FloatGpuArray[GPUDots.N_PARAMS_FLOAT];
        for (int i=0;i<GPUDots.N_PARAMS_FLOAT;i++) {
            iGA_Float[i] = new FloatGpuArray();
        }
        
        iGA_Int = new IntGpuArray[GPUDots.N_PARAMS_INT];
        for (int i=0;i<GPUDots.N_PARAMS_INT;i++) {
            iGA_Int[i] = new IntGpuArray();
        }
    }
    
    public void setAllNeighborsConvergenceTo1() {
        for(int index=0;index<iGA_Int[GPUDots.ALL_NEIGHBORS_HAVE_CONVERGED].length;index++) {
            iGA_Int[GPUDots.ALL_NEIGHBORS_HAVE_CONVERGED].cpuArray[index]=1;
        }
        iGA_Int[GPUDots.ALL_NEIGHBORS_HAVE_CONVERGED].push();
    }
    
    public void push(final ArrayList<DotN> dotList) {
        numberOfDots = dotList.size();
        // Mem allocation check
        for (int i=0;i<GPUDots.N_PARAMS_FLOAT;i++) {
            iGA_Float[i].ensureAllocatedSize(numberOfDots, false, false);
            iGA_Float[i].length=numberOfDots;
        }
        for (int i=0;i<GPUDots.N_PARAMS_INT;i++) {
            iGA_Int[i].ensureAllocatedSize(numberOfDots, false, false);
            iGA_Int[i].length=numberOfDots;
        }
        //---------------------------------
        // All set
        // First : hydrate tabs
        range(0,dotList.size()).parallel().forEach( index -> {
            DotN dn = dotList.get(index);
            //int index = dn.dotIndex;
            iGA_Float[GPUDots.PX].cpuArray[index] = dn.pos.x;  iGA_Float[GPUDots.PY].cpuArray[index] = dn.pos.y;  iGA_Float[GPUDots.PZ].cpuArray[index] = dn.pos.z;
            iGA_Float[GPUDots.NX].cpuArray[index] = dn.Norm.x; iGA_Float[GPUDots.NY].cpuArray[index] = dn.Norm.y; iGA_Float[GPUDots.NZ].cpuArray[index] = dn.Norm.z;
            
            iGA_Float[GPUDots.SUPER_DOT_RADIUS_SQUARED].cpuArray[index]=dn.superDotRadiusSquared;
            iGA_Float[GPUDots.RELAXED].cpuArray[index]=0;//dn.relaxed;
            
            if (!dn.isSuperDot) {
                iGA_Int[GPUDots.CELL_ID].cpuArray[index]=dn.ct.idInt;
            } else {
                iGA_Int[GPUDots.CELL_ID].cpuArray[index]=-1;
            }
            iGA_Int[GPUDots.HAS_CONVERGED].cpuArray[index]=dn.hasConverged?1:0;
            iGA_Int[GPUDots.ALL_NEIGHBORS_HAVE_CONVERGED].cpuArray[index]=dn.allNeighborsHaveConverged?1:0;
            iGA_Int[GPUDots.ALL_NEIGHBORS_HAVE_CONVERGED_PREVIOUSLY].cpuArray[index]=dn.allNeighborsHaveConvergedPreviously?1:0;
        });
        
        // Second : push into device hydrated tabs
        
        iGA_Float[GPUDots.PX].push();iGA_Float[GPUDots.PY].push();iGA_Float[GPUDots.PZ].push();
        iGA_Float[GPUDots.NX].push();iGA_Float[GPUDots.NY].push();iGA_Float[GPUDots.NZ].push();
        iGA_Float[GPUDots.SUPER_DOT_RADIUS_SQUARED].push();
        iGA_Float[GPUDots.RELAXED].push();
        
        iGA_Int[GPUDots.CELL_ID].push();
        iGA_Int[GPUDots.HAS_CONVERGED].push();
        iGA_Int[GPUDots.ALL_NEIGHBORS_HAVE_CONVERGED].push();
        iGA_Int[GPUDots.ALL_NEIGHBORS_HAVE_CONVERGED_PREVIOUSLY].push();
        
        // Last : reset (=put zero) in GPUMem when needed
        // FX, MX, RFX
        iGA_Float[GPUDots.FX].setGpuMemToZero();
        iGA_Float[GPUDots.FY].setGpuMemToZero();
        iGA_Float[GPUDots.FZ].setGpuMemToZero();
        
        iGA_Float[GPUDots.MX].setGpuMemToZero();
        iGA_Float[GPUDots.MY].setGpuMemToZero();
        iGA_Float[GPUDots.MZ].setGpuMemToZero();
        
        iGA_Float[GPUDots.RFX].setGpuMemToZero();
        iGA_Float[GPUDots.RFY].setGpuMemToZero();
        iGA_Float[GPUDots.RFZ].setGpuMemToZero();
        
        // N_Neigh
        iGA_Int[GPUDots.N_NEIGH].setGpuMemToZero();
        this.hasBeenPushed=true;
    }
    
    public void pop(ArrayList<DotN> dotList) {
        // Returns updated:
        // Float : Moment, Force, RepulsiveForce        
        // Int : N_Neigh, and all_neighbors_have_converged
        // Fetch Data from GPU
        // Copy the data from the device back to the host and clean up
       
        iGA_Float[GPUDots.FX].pop();iGA_Float[GPUDots.FY].pop();iGA_Float[GPUDots.FZ].pop();
        iGA_Float[GPUDots.MX].pop();iGA_Float[GPUDots.MY].pop();iGA_Float[GPUDots.MZ].pop();
        iGA_Float[GPUDots.RFX].pop();iGA_Float[GPUDots.RFY].pop();iGA_Float[GPUDots.RFZ].pop();
        
        iGA_Float[GPUDots.RELAXED].pop();
        iGA_Int[GPUDots.N_NEIGH].pop();
        iGA_Int[GPUDots.ALL_NEIGHBORS_HAVE_CONVERGED].pop();
        // Put data into dotList
        dotList.parallelStream().forEach(dn -> {
            int index = dn.dotIndex;
            
            dn.force.x += iGA_Float[GPUDots.FX].cpuArray[index];
            dn.force.y += iGA_Float[GPUDots.FY].cpuArray[index];
            dn.force.z += iGA_Float[GPUDots.FZ].cpuArray[index];
            
            dn.repForce.x += iGA_Float[GPUDots.RFX].cpuArray[index];
            dn.repForce.y += iGA_Float[GPUDots.RFY].cpuArray[index];
            dn.repForce.z += iGA_Float[GPUDots.RFZ].cpuArray[index];
            
            dn.moment.x += iGA_Float[GPUDots.MX].cpuArray[index];
            dn.moment.y += iGA_Float[GPUDots.MY].cpuArray[index];
            dn.moment.z += iGA_Float[GPUDots.MZ].cpuArray[index];
            
            dn.relaxed = java.lang.Float.max(iGA_Float[GPUDots.RELAXED].cpuArray[index],dn.relaxed);
            dn.N_Neighbor = iGA_Int[GPUDots.N_NEIGH].cpuArray[index];
                
            dn.allNeighborsHaveConverged = (iGA_Int[GPUDots.ALL_NEIGHBORS_HAVE_CONVERGED].cpuArray[index]==1);  
        });
    }
    
    public void freeMem() {
            hasBeenPushed=false;
            //Floats          
            for (int i=0;i<GPUDots.N_PARAMS_FLOAT;i++) {
                iGA_Float[i].freeMem();
            }            
            // Ints
            for (int i=0;i<GPUDots.N_PARAMS_INT;i++) {
                iGA_Int[i].freeMem();
            }
    }
}
