package eu.kiaru.limeseg.opt;

import jcuda.Pointer;
import jcuda.Sizeof;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import eu.kiaru.limeseg.cudaHelper.FloatGpuArray;
import eu.kiaru.limeseg.cudaHelper.IntGpuArray;

/**
 *
 * @author Nicolas Chiaruttini
 */
public class WGPUBlockList extends GPUBlockList{    
    // One data point per block
    FloatGpuArray avgX, avgY, avgZ; // average position per block
    FloatGpuArray dirX, dirY, dirZ; // direction of furthest point in block
    
    IntGpuArray idPtFar;    
    FloatGpuArray distPtFar;
    
    FloatGpuArray fGA_pScalVal;
    IntGpuArray iGA_rkBlPos, iGA_rkBlNeg, iGA_rkBlMid0, iGA_rkBlMid1;
    IntGpuArray iGA_nPtBlPos, iGA_nPtBlNeg, iGA_nPtBlMid0, iGA_nPtBlMid1;
    IntGpuArray iGA_newBlockCvg, iGA_newBlockLevel;
    IntGpuArray iGA_WhatToDoWithTheseBlocks;
    
    IntGpuArray nPtKeep, addrPt;
    IntGpuArray nBlocsKeep, addrBloc;
    
    int nBlocksMaxDuringProcessing=0;
    int nDotsInTree;
    
    public WGPUBlockList() {
        super();
        avgX = new FloatGpuArray();
        avgY = new FloatGpuArray();
        avgZ = new FloatGpuArray();
        
        dirX = new FloatGpuArray();
        dirY = new FloatGpuArray();
        dirZ = new FloatGpuArray();
        
        idPtFar = new IntGpuArray();
        
        fGA_pScalVal = new FloatGpuArray();
        distPtFar = new FloatGpuArray();
        iGA_rkBlPos = new IntGpuArray();
        iGA_rkBlNeg = new IntGpuArray();
        iGA_rkBlMid0 = new IntGpuArray();
        iGA_rkBlMid1 = new IntGpuArray();
        
        iGA_nPtBlPos = new IntGpuArray();
        iGA_nPtBlNeg = new IntGpuArray();
        iGA_nPtBlMid0 = new IntGpuArray();
        iGA_nPtBlMid1 = new IntGpuArray();
        
        iGA_newBlockCvg = new IntGpuArray();
        iGA_newBlockLevel = new IntGpuArray();
        iGA_WhatToDoWithTheseBlocks = new IntGpuArray();
        
        nPtKeep = new IntGpuArray(); 
        nPtKeep.ensureAllocatedSize(4, false, false);
        nPtKeep.length=4;
        nBlocsKeep = new IntGpuArray();
        nBlocsKeep.ensureAllocatedSize(4, false, false);
        nBlocsKeep.length=4;
        
        addrPt = new IntGpuArray(); 
        addrBloc = new IntGpuArray();
    }
    
    void computeCenters(GPUDots gDots, int gpuBlockSize) {
        //this.writeGPUBlockInfos();
        avgX.ensureAllocatedSize(nBlocks, false, false); avgX.length=nBlocks;
        avgY.ensureAllocatedSize(nBlocks, false, false); avgY.length=nBlocks;
        avgZ.ensureAllocatedSize(nBlocks, false, false); avgZ.length=nBlocks;  
        
        avgX.setGpuMemToZero();
        avgY.setGpuMemToZero();
        avgZ.setGpuMemToZero();
        
        Pointer kernelParameters = Pointer.to(
                    // Dots properties
                    Pointer.to(gDots.iGA_Float[GPUDots.PX].gpuArray), 
                    Pointer.to(gDots.iGA_Float[GPUDots.PY].gpuArray),
                    Pointer.to(gDots.iGA_Float[GPUDots.PZ].gpuArray),
                    // Blocks Properties
                    Pointer.to(iGA_arrayDotsIndexes.gpuArray),
                    Pointer.to(iGA_addrStartBlock0.gpuArray),Pointer.to(iGA_nPtBlock0.gpuArray),
                    Pointer.to(iGA_addrStartBlock1.gpuArray),Pointer.to(iGA_nPtBlock1.gpuArray),
                    Pointer.to(avgX.gpuArray), Pointer.to(avgY.gpuArray),Pointer.to(avgZ.gpuArray),
                    Pointer.to(iGA_idBlock.gpuArray),
                    Pointer.to(iGA_offsIntBlock.gpuArray)   
            );
        cuCtxSynchronize();
        cuLaunchKernel(Optimizer.fAVGBlocks, 
                           nGPUBlocks,  1, 1,             // Grid dimension 
                           gpuBlockSize, 1, 1,      // Block dimension
                           6*Sizeof.INT+gpuBlockSize*3*Sizeof.FLOAT, null,                         // Shared memory size and stream 
                           kernelParameters, null                      // Kernel- and extra parameters
        ); 
        cuCtxSynchronize();
    }
    
    void computeDirFurthestPoint(GPUDots gDots, int gpuBlockSize) {
        idPtFar.ensureAllocatedSize(nBlocks, false, false);idPtFar.length=nBlocks;
        distPtFar.ensureAllocatedSize(nBlocks, false, false);distPtFar.length=nBlocks;
        distPtFar.setGpuMemToZero();
        Pointer kernelParameters = Pointer.to(
                    // Dots properties
                    Pointer.to(gDots.iGA_Float[GPUDots.PX].gpuArray), 
                    Pointer.to(gDots.iGA_Float[GPUDots.PY].gpuArray),
                    Pointer.to(gDots.iGA_Float[GPUDots.PZ].gpuArray),
                    // Blocks Properties
                    Pointer.to(iGA_arrayDotsIndexes.gpuArray),
                    Pointer.to(iGA_addrStartBlock0.gpuArray),Pointer.to(iGA_nPtBlock0.gpuArray),
                    Pointer.to(iGA_addrStartBlock1.gpuArray),Pointer.to(iGA_nPtBlock1.gpuArray),
                    Pointer.to(avgX.gpuArray), Pointer.to(avgY.gpuArray),Pointer.to(avgZ.gpuArray),
                    Pointer.to(iGA_idBlock.gpuArray),
                    Pointer.to(iGA_offsIntBlock.gpuArray),                      
                    
                    // Output values
                    Pointer.to(idPtFar.gpuArray),
                    Pointer.to(distPtFar.gpuArray)
            );
        cuCtxSynchronize();
        cuLaunchKernel(Optimizer.fKDTree_FindFurthestBlocks, 
                           nGPUBlocks,  1, 1,             // Grid dimension 
                           gpuBlockSize, 1, 1,      // Block dimension
                           6*Sizeof.INT+3*Sizeof.FLOAT+gpuBlockSize*(Sizeof.FLOAT+Sizeof.INT), null,                         // Shared memory size and stream 
                           kernelParameters, null                      // Kernel- and extra parameters
        );
        cuCtxSynchronize();
        
        dirX.ensureAllocatedSize(nBlocks, false, false);dirX.length=nBlocks;
        dirY.ensureAllocatedSize(nBlocks, false, false);dirY.length=nBlocks;
        dirZ.ensureAllocatedSize(nBlocks, false, false);dirZ.length=nBlocks;
        
        kernelParameters = Pointer.to(
                    // Dots properties
                    Pointer.to(gDots.iGA_Float[GPUDots.PX].gpuArray), 
                    Pointer.to(gDots.iGA_Float[GPUDots.PY].gpuArray),
                    Pointer.to(gDots.iGA_Float[GPUDots.PZ].gpuArray),
                    // Blocks Properties
                    // Pointer.to(iGA_arrayDotsIndexes.gpuArray),
                    Pointer.to(avgX.gpuArray), Pointer.to(avgY.gpuArray),Pointer.to(avgZ.gpuArray),
                    Pointer.to(idPtFar.gpuArray),
                    // Output values                    
                    Pointer.to(dirX.gpuArray), Pointer.to(dirY.gpuArray),Pointer.to(dirZ.gpuArray),
                    Pointer.to(new float[]{this.nBlocks})
            );    
        gpuBlockSize=Optimizer.MAX_THREADS_PER_BLOCK;
        int nGPUBlocksRequired = ((int)(this.nBlocks)-1)/gpuBlockSize+1; 
        // Take care! very different call this time!
        cuCtxSynchronize();
        cuLaunchKernel(Optimizer.fCalcDir, 
                           nGPUBlocksRequired,  1, 1,             // Grid dimension 
                           gpuBlockSize, 1, 1,      // Block dimension
                           0, null,                         // Shared memory size and stream 
                           kernelParameters, null                      // Kernel- and extra parameters
        ); 
        cuCtxSynchronize();
    }
    
    public void mapDotsIntoNewBlocksAndGetRank(GPUDots gDots, int gpuBlockSize, float limitInteractAttract) {
        fGA_pScalVal.ensureAllocatedSize(nDots, false, false);fGA_pScalVal.length=nDots;
        // For Ranking : [0 NDots]= block0, [NDots 2*NDots] = block1
        iGA_rkBlPos.ensureAllocatedSize(2*nDots, false, false);iGA_rkBlPos.length=2*nDots;
        iGA_rkBlNeg.ensureAllocatedSize(2*nDots, false, false);iGA_rkBlNeg.length=2*nDots;
        iGA_rkBlMid0.ensureAllocatedSize(2*nDots, false, false);iGA_rkBlMid0.length=2*nDots;
        iGA_rkBlMid1.ensureAllocatedSize(2*nDots, false, false);iGA_rkBlMid1.length=2*nDots;
        // For the number of dots [O nBlocks]=block0, [nBlocks 2*nBlocks]= block1
        iGA_nPtBlPos.ensureAllocatedSize(2*nBlocks, false, false);iGA_nPtBlPos.length=2*nBlocks;
        iGA_nPtBlNeg.ensureAllocatedSize(2*nBlocks, false, false);iGA_nPtBlNeg.length=2*nBlocks;
        iGA_nPtBlMid0.ensureAllocatedSize(2*nBlocks, false, false);iGA_nPtBlMid0.length=2*nBlocks;
        iGA_nPtBlMid1.ensureAllocatedSize(2*nBlocks, false, false);iGA_nPtBlMid1.length=2*nBlocks;
        
        iGA_newBlockLevel.ensureAllocatedSize(4*nBlocks, false, false);iGA_newBlockLevel.length=4*nBlocks; // How could this bug have persisted ??
        iGA_newBlockLevel.setGpuMemToZero();
        
        iGA_newBlockCvg.ensureAllocatedSize(4*nBlocks, false, false);iGA_newBlockCvg.length=4*nBlocks;
        iGA_newBlockCvg.setGpuMemToZero();
        // if remains at 0 this means to everything has converged in the block
        
        iGA_nPtBlPos.setGpuMemToZero();
        iGA_nPtBlNeg.setGpuMemToZero();
        iGA_nPtBlMid0.setGpuMemToZero();
        iGA_nPtBlMid1.setGpuMemToZero();
        
        float sqLimitInteractAttract = (float) java.lang.Math.sqrt(limitInteractAttract); 
        Pointer kernelParameters = Pointer.to(
                    // Dots properties
                    Pointer.to(gDots.iGA_Float[GPUDots.PX].gpuArray), 
                    Pointer.to(gDots.iGA_Float[GPUDots.PY].gpuArray),
                    Pointer.to(gDots.iGA_Float[GPUDots.PZ].gpuArray),
                    Pointer.to(gDots.iGA_Float[GPUDots.SUPER_DOT_RADIUS_SQUARED].gpuArray), // necessary for the mapping
                    Pointer.to(gDots.iGA_Int[GPUDots.ALL_NEIGHBORS_HAVE_CONVERGED].gpuArray),
                    // Blocks Properties
                    Pointer.to(iGA_arrayDotsIndexes.gpuArray),
                    Pointer.to(iGA_addrStartBlock0.gpuArray),Pointer.to(iGA_nPtBlock0.gpuArray),
                    Pointer.to(iGA_addrStartBlock1.gpuArray),Pointer.to(iGA_nPtBlock1.gpuArray),
                    Pointer.to(avgX.gpuArray), Pointer.to(avgY.gpuArray),Pointer.to(avgZ.gpuArray),
                    Pointer.to(dirX.gpuArray), Pointer.to(dirY.gpuArray),Pointer.to(dirZ.gpuArray),
                    Pointer.to(iGA_blockLevel.gpuArray), // to know between level 0 and above
                    Pointer.to(iGA_idBlock.gpuArray),
                    Pointer.to(iGA_offsIntBlock.gpuArray),
                    // Output values                    
                    Pointer.to(fGA_pScalVal.gpuArray), 
                    Pointer.to(iGA_rkBlPos.gpuArray), 
                    Pointer.to(iGA_rkBlNeg.gpuArray),
                    Pointer.to(iGA_rkBlMid0.gpuArray),
                    Pointer.to(iGA_rkBlMid1.gpuArray),
                    // 
                    Pointer.to(iGA_nPtBlPos.gpuArray), 
                    Pointer.to(iGA_nPtBlNeg.gpuArray), 
                    Pointer.to(iGA_nPtBlMid0.gpuArray), 
                    Pointer.to(iGA_nPtBlMid1.gpuArray), 
                    //
                    Pointer.to(iGA_newBlockCvg.gpuArray),
                    //
                    Pointer.to(new int[]{nBlocks}),
                    Pointer.to(new int[]{nDots}),
                    Pointer.to(new float[]{sqLimitInteractAttract})// offset for blocks : 0 or 1
            );
        cuCtxSynchronize();
        cuLaunchKernel(Optimizer.fMapDots, 
                           nGPUBlocks,  1, 1,             // Grid dimension 
                           gpuBlockSize, 1, 1,      // Block dimension
                           7*Sizeof.INT+6*Sizeof.FLOAT, null,                         // Shared memory size and stream 
                           kernelParameters, null                      // Kernel- and extra parameters
        ); 
        cuCtxSynchronize();
        
    }
    
    public void dispatchBlocks(GPUDots gDots, 
                               WGPUBlockList gBlockListOut,
                               GPUBlockList gBlockListFinal,
                               GPUBlockList gBlockListIgnored, int gpuBlockSize, int minInteract, int minPointsToKeep) {
        // KEEP -> to Final
        // SPLIT -> to Out
        // DISCARD -> to Ignored
        // TRASH -> nowhere
        iGA_WhatToDoWithTheseBlocks.ensureAllocatedSize(4*nBlocks, false, false);
        iGA_WhatToDoWithTheseBlocks.length=4*nBlocks;
        
        addrPt.ensureAllocatedSize(4*nBlocks, false, false);
        addrPt.length=4*nBlocks;
        
        addrBloc.ensureAllocatedSize(4*nBlocks, false, false);
        addrBloc.length=4*nBlocks;
        
        nPtKeep.ensureAllocatedSize(4, false, false); nPtKeep.length=4;
        nPtKeep.cpuArray[BlockOfDots.BLOCK_KEEP]=gBlockListFinal.nDots;
        nPtKeep.cpuArray[BlockOfDots.BLOCK_SPLIT]=0;//yes
        nPtKeep.cpuArray[BlockOfDots.BLOCK_DISCARD]=gBlockListIgnored.nDots;
        nPtKeep.cpuArray[BlockOfDots.BLOCK_TRASH]=0;//yes
        
        nBlocsKeep.ensureAllocatedSize(4, false, false); nBlocsKeep.length=4;
        nBlocsKeep.cpuArray[BlockOfDots.BLOCK_KEEP]=gBlockListFinal.nBlocks;
        nBlocsKeep.cpuArray[BlockOfDots.BLOCK_SPLIT]=0;//yes
        nBlocsKeep.cpuArray[BlockOfDots.BLOCK_DISCARD]=gBlockListIgnored.nBlocks;
        nBlocsKeep.cpuArray[BlockOfDots.BLOCK_TRASH]=0;//yes
        
        nPtKeep.push();
        nBlocsKeep.push();
        
        // Initialize addr and bloc...
        
        // The tree has 4 times more blocks maximum than before.
        // Some of these blocks will be trashed, or kept for force computation, or kept to be splited, 
        // or ignored but kept to make superdots 
        Pointer kernelParameters = Pointer.to(// Dots properties
                    // Blocks Properties
                    Pointer.to(iGA_nPtBlock0.gpuArray),
                    Pointer.to(iGA_nPtBlock1.gpuArray),
                    Pointer.to(iGA_blockLevel.gpuArray),                 
                    Pointer.to(iGA_nPtBlPos.gpuArray), 
                    Pointer.to(iGA_nPtBlNeg.gpuArray), 
                    Pointer.to(iGA_nPtBlMid0.gpuArray), 
                    Pointer.to(iGA_nPtBlMid1.gpuArray),  
                    Pointer.to(iGA_newBlockCvg.gpuArray),                    
                    Pointer.to(iGA_newBlockLevel.gpuArray),
                    Pointer.to(new int[]{nDots}), // offset for blocks : 0 or 1
                    Pointer.to(new int[]{nBlocks}),
                    Pointer.to(new int[]{minInteract}),
                    Pointer.to(new int[]{minPointsToKeep}),
                    // Output values 
                    Pointer.to(iGA_WhatToDoWithTheseBlocks.gpuArray),
                    Pointer.to(nPtKeep.gpuArray),
                    Pointer.to(nBlocsKeep.gpuArray), 
                    Pointer.to(addrPt.gpuArray),
                    Pointer.to(addrBloc.gpuArray)
        );
        int nGPUBlocksRequired = ((int)(this.nBlocks)-1)/gpuBlockSize+1; 
        //System.out.println("nGPUBlocksRequired="+ nGPUBlocksRequired);
        cuCtxSynchronize();
        cuLaunchKernel(Optimizer.fSortBlocks, 
                           nGPUBlocksRequired,  1, 1,             // Grid dimension 
                           gpuBlockSize, 1, 1,      // Block dimension
                           0, null,                         // Shared memory size and stream 
                           kernelParameters, null                      // Kernel- and extra parameters
        );
        cuCtxSynchronize();
        
        nPtKeep.pop();
        nBlocsKeep.pop();
        
        // first pour les array dots
        gBlockListOut.nDots=nPtKeep.cpuArray[BlockOfDots.BLOCK_SPLIT];
        gBlockListOut.iGA_arrayDotsIndexes.ensureAllocatedSize(gBlockListOut.nDots, false, false);  // no need to keep gpu and cpu data
        gBlockListOut.iGA_arrayDotsIndexes.length=gBlockListOut.nDots;
        
        gBlockListFinal.nDots=nPtKeep.cpuArray[BlockOfDots.BLOCK_KEEP];
        gBlockListFinal.iGA_arrayDotsIndexes.ensureAllocatedSize(gBlockListFinal.nDots, true, false); // we keep gpu data
        gBlockListFinal.iGA_arrayDotsIndexes.length=gBlockListFinal.nDots;
        
        gBlockListIgnored.nDots=nPtKeep.cpuArray[BlockOfDots.BLOCK_DISCARD];
        gBlockListIgnored.iGA_arrayDotsIndexes.ensureAllocatedSize(gBlockListIgnored.nDots, true, false); // we keep gpu data
        gBlockListIgnored.iGA_arrayDotsIndexes.length=gBlockListIgnored.nDots;
        
        // second pour les blocs
        gBlockListOut.nBlocks=nBlocsKeep.cpuArray[BlockOfDots.BLOCK_SPLIT];
        gBlockListOut.iGA_addrStartBlock0.ensureAllocatedSize(gBlockListOut.nBlocks, false, false);
        gBlockListOut.iGA_addrStartBlock0.length=gBlockListOut.nBlocks;
        gBlockListOut.iGA_addrStartBlock1.ensureAllocatedSize(gBlockListOut.nBlocks, false, false);
        gBlockListOut.iGA_addrStartBlock1.length=gBlockListOut.nBlocks;
        gBlockListOut.iGA_nPtBlock0.ensureAllocatedSize(gBlockListOut.nBlocks, false, false);
        gBlockListOut.iGA_nPtBlock0.length=gBlockListOut.nBlocks;
        gBlockListOut.iGA_nPtBlock1.ensureAllocatedSize(gBlockListOut.nBlocks, false, false);
        gBlockListOut.iGA_nPtBlock1.length=gBlockListOut.nBlocks;
        gBlockListOut.iGA_blockLevel.ensureAllocatedSize(gBlockListOut.nBlocks, false, false);
        //gBlockListOut.iGA_blockLevel.setGpuMemToZero();
        gBlockListOut.iGA_blockLevel.length=gBlockListOut.nBlocks;
        
        gBlockListFinal.nBlocks=nBlocsKeep.cpuArray[BlockOfDots.BLOCK_KEEP];
        gBlockListFinal.iGA_addrStartBlock0.ensureAllocatedSize(gBlockListFinal.nBlocks, true, false);
        gBlockListFinal.iGA_addrStartBlock0.length=gBlockListFinal.nBlocks;
        gBlockListFinal.iGA_addrStartBlock1.ensureAllocatedSize(gBlockListFinal.nBlocks, true, false);
        gBlockListFinal.iGA_addrStartBlock1.length=gBlockListFinal.nBlocks;
        gBlockListFinal.iGA_nPtBlock0.ensureAllocatedSize(gBlockListFinal.nBlocks, true, false);
        gBlockListFinal.iGA_nPtBlock0.length=gBlockListFinal.nBlocks;
        gBlockListFinal.iGA_nPtBlock1.ensureAllocatedSize(gBlockListFinal.nBlocks, true, false);
        gBlockListFinal.iGA_nPtBlock1.length=gBlockListFinal.nBlocks;
        gBlockListFinal.iGA_blockLevel.ensureAllocatedSize(gBlockListFinal.nBlocks, true, false);
        //gBlockListFinal.iGA_blockLevel.setGpuMemToZero();
        gBlockListFinal.iGA_blockLevel.length=gBlockListFinal.nBlocks;
        gBlockListIgnored.nBlocks=nBlocsKeep.cpuArray[BlockOfDots.BLOCK_DISCARD];
        gBlockListIgnored.iGA_addrStartBlock0.ensureAllocatedSize(gBlockListIgnored.nBlocks, true, false);
        gBlockListIgnored.iGA_addrStartBlock0.length=gBlockListIgnored.nBlocks;
        gBlockListIgnored.iGA_addrStartBlock1.ensureAllocatedSize(gBlockListIgnored.nBlocks, true, false);
        gBlockListIgnored.iGA_addrStartBlock1.length=gBlockListIgnored.nBlocks;
        gBlockListIgnored.iGA_nPtBlock0.ensureAllocatedSize(gBlockListIgnored.nBlocks, true, false);
        gBlockListIgnored.iGA_nPtBlock0.length=gBlockListIgnored.nBlocks;
        gBlockListIgnored.iGA_nPtBlock1.ensureAllocatedSize(gBlockListIgnored.nBlocks, true, false);
        gBlockListIgnored.iGA_nPtBlock1.length=gBlockListIgnored.nBlocks;
        gBlockListIgnored.iGA_blockLevel.ensureAllocatedSize(gBlockListIgnored.nBlocks, true, false);
        gBlockListIgnored.iGA_blockLevel.length=gBlockListIgnored.nBlocks;
  
        kernelParameters = Pointer.to(
                    // Blocks Properties
                    Pointer.to(iGA_arrayDotsIndexes.gpuArray),
                    Pointer.to(iGA_addrStartBlock0.gpuArray),Pointer.to(iGA_nPtBlock0.gpuArray),
                    Pointer.to(iGA_addrStartBlock1.gpuArray),Pointer.to(iGA_nPtBlock1.gpuArray),
                    Pointer.to(iGA_blockLevel.gpuArray), // to know between level 0 and above
                    // per GPU block In
                    Pointer.to(iGA_idBlock.gpuArray),
                    Pointer.to(iGA_offsIntBlock.gpuArray),
                    // Output values                    
                    Pointer.to(iGA_rkBlPos.gpuArray), 
                    Pointer.to(iGA_rkBlNeg.gpuArray),
                    Pointer.to(iGA_rkBlMid0.gpuArray),
                    Pointer.to(iGA_rkBlMid1.gpuArray),
                    // 
                    Pointer.to(iGA_nPtBlPos.gpuArray), 
                    Pointer.to(iGA_nPtBlNeg.gpuArray), 
                    Pointer.to(iGA_nPtBlMid0.gpuArray), 
                    Pointer.to(iGA_nPtBlMid1.gpuArray), 
                    //
                    Pointer.to(new int[]{nBlocks}),
                    Pointer.to(new int[]{nDots}),
                    // How to deal with new blocks
                    Pointer.to(iGA_WhatToDoWithTheseBlocks.gpuArray),
                    Pointer.to(addrPt.gpuArray),
                    Pointer.to(addrBloc.gpuArray),
                    Pointer.to(iGA_newBlockLevel.gpuArray),
                    // Outputs... many of them!
                    // Bloc Keep,
                    Pointer.to(gBlockListOut.iGA_arrayDotsIndexes.gpuArray),
                    Pointer.to(gBlockListOut.iGA_addrStartBlock0.gpuArray),Pointer.to(gBlockListOut.iGA_nPtBlock0.gpuArray),
                    Pointer.to(gBlockListOut.iGA_addrStartBlock1.gpuArray),Pointer.to(gBlockListOut.iGA_nPtBlock1.gpuArray),
                    Pointer.to(gBlockListOut.iGA_blockLevel.gpuArray), // to know between level 0 and above
                    // Bloc Final
                    Pointer.to(gBlockListFinal.iGA_arrayDotsIndexes.gpuArray),
                    Pointer.to(gBlockListFinal.iGA_addrStartBlock0.gpuArray),Pointer.to(gBlockListFinal.iGA_nPtBlock0.gpuArray),
                    Pointer.to(gBlockListFinal.iGA_addrStartBlock1.gpuArray),Pointer.to(gBlockListFinal.iGA_nPtBlock1.gpuArray),
                    Pointer.to(gBlockListFinal.iGA_blockLevel.gpuArray), // to know between level 0 and above
                    // Bloc Discard
                    Pointer.to(gBlockListIgnored.iGA_arrayDotsIndexes.gpuArray),
                    Pointer.to(gBlockListIgnored.iGA_addrStartBlock0.gpuArray),Pointer.to(gBlockListIgnored.iGA_nPtBlock0.gpuArray),
                    Pointer.to(gBlockListIgnored.iGA_addrStartBlock1.gpuArray),Pointer.to(gBlockListIgnored.iGA_nPtBlock1.gpuArray),
                    Pointer.to(gBlockListIgnored.iGA_blockLevel.gpuArray)
            );
        cuCtxSynchronize();
        cuLaunchKernel(Optimizer.fDispatchDots, 
                           nGPUBlocks,  1, 1,             // Grid dimension 
                           gpuBlockSize, 1, 1,      // Block dimension
                           70*Sizeof.INT, null,                         // Shared memory size and stream 
                           kernelParameters, null                      // Kernel- and extra parameters
        ); 
        cuCtxSynchronize();    
        gBlockListOut.hasBeenPushed=true;
        gBlockListIgnored.hasBeenPushed=true;
        gBlockListFinal.hasBeenPushed=true;
    }    
    // Cudamemcpystride    
    @Override
    public void freeMem() {
        super.freeMem();
        avgX.freeMem();
        avgY.freeMem();
        avgZ.freeMem(); // average position per block
        dirX.freeMem();
        dirY.freeMem();
        dirZ.freeMem(); // direction of furthest point in block
    
        idPtFar.freeMem();
        distPtFar.freeMem();
    
        fGA_pScalVal.freeMem();
        iGA_rkBlPos.freeMem();
        iGA_rkBlNeg.freeMem();
        iGA_rkBlMid0.freeMem();
        iGA_rkBlMid1.freeMem();
        iGA_nPtBlPos.freeMem();
        iGA_nPtBlNeg.freeMem();
        iGA_nPtBlMid0.freeMem();
        iGA_nPtBlMid1.freeMem();
        iGA_newBlockCvg.freeMem(); 
        iGA_newBlockLevel.freeMem();
        iGA_WhatToDoWithTheseBlocks.freeMem();
        nPtKeep.freeMem(); 
        addrPt.freeMem();
        nBlocsKeep.freeMem(); 
        addrBloc.freeMem();
    }
}
