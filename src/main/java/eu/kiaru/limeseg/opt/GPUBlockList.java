package eu.kiaru.limeseg.opt;

import java.util.ArrayList;
import java.util.stream.IntStream;

import eu.kiaru.limeseg.cudaHelper.IntGpuArray;
import eu.kiaru.limeseg.struct.DotN;

/**
 * List of Block used for GPU space partitioning
 * @author Nicolas Chiaruttini
 */
public class GPUBlockList {
    boolean hasBeenPushed;
    
    
    // Big array of dots : most of the memory is here
    IntGpuArray iGA_arrayDotsIndexes;    
    
    // One point per starting block
    IntGpuArray iGA_addrStartBlock0;
    IntGpuArray iGA_addrStartBlock1;
    IntGpuArray iGA_nPtBlock0;
    IntGpuArray iGA_nPtBlock1;
    IntGpuArray iGA_blockLevel;
    
    
    // ----------- For corces computation
    // One point per GPU Block
    IntGpuArray iGA_idBlock;
    IntGpuArray iGA_offsIntBlock;
    
    int nDots, nBlocks, nGPUBlocks, nInteractions;
    
    
    public GPUBlockList() {
        hasBeenPushed=false;
        nDots=0;
        nBlocks=0;
        nGPUBlocks=0;
        
        iGA_arrayDotsIndexes = new IntGpuArray();
        
        iGA_addrStartBlock0 = new IntGpuArray();
        iGA_nPtBlock0= new IntGpuArray();
        iGA_addrStartBlock1 = new IntGpuArray();
        iGA_nPtBlock1= new IntGpuArray();
        iGA_blockLevel= new IntGpuArray();
        
        iGA_idBlock= new IntGpuArray();
        iGA_offsIntBlock= new IntGpuArray();   
    }
    
    
    public static int ONE_THREAD_PER_INTERACTION = 0;
    public static int ONE_THREAD_PER_DOT = 1;
    
    public int getAverageNumberOfRequiredThreadPerBlock(int mode) {
        int ans=0;
        iGA_nPtBlock0.pop();
        iGA_nPtBlock1.pop();
        iGA_blockLevel.pop();
        if (mode==ONE_THREAD_PER_INTERACTION) { 
            nInteractions=0;
            for (int i=0;i<nBlocks;i++)  {                
                int numberOfInteractions;
                if (iGA_blockLevel.cpuArray[i]==0) {
                    numberOfInteractions = (iGA_nPtBlock0.cpuArray[i])*(iGA_nPtBlock0.cpuArray[i]-1)/2;
                } else {
                    numberOfInteractions = iGA_nPtBlock0.cpuArray[i]*iGA_nPtBlock1.cpuArray[i];
                }
                nInteractions+=numberOfInteractions;
            }
            ans=nInteractions/nBlocks;
        }
        if (mode==ONE_THREAD_PER_DOT) {
            for (int i=0;i<nBlocks;i++)  {
               ans += iGA_nPtBlock0.cpuArray[i]+iGA_nPtBlock1.cpuArray[i];
            }
            ans=ans/nBlocks;
        }
        
        return ans;
    }
    
    public void mapBlocksToGPU(int mode, int gpuBlockSize){//, boolean needsPop) {
        int nGPUBlocksRequired = 0;
//        assert nBlocks>0;
//        assert hasBeenPushed;
        int[] iGPUBlockStart = new int[nBlocks];
        int[] nGPUBlockNeeded = new int[nBlocks];
               
            iGA_nPtBlock0.pop();
            iGA_nPtBlock1.pop();
            iGA_blockLevel.pop();
            
        if (mode==ONE_THREAD_PER_INTERACTION) { 
            nInteractions=0;
            for (int i=0;i<nBlocks;i++)  {
                // Count number of interactions
                int numberOfInteractions;
                if (iGA_blockLevel.cpuArray[i]==0) {
                    numberOfInteractions = (iGA_nPtBlock0.cpuArray[i])*(iGA_nPtBlock0.cpuArray[i]-1)/2;
                } else {
                    numberOfInteractions = iGA_nPtBlock0.cpuArray[i]*iGA_nPtBlock1.cpuArray[i];
                }
                int nGPUBLocksForThisBlock=1;  
                nInteractions+=numberOfInteractions;
                if (numberOfInteractions>gpuBlockSize) {
                    nGPUBLocksForThisBlock = ((int)(numberOfInteractions)-1)/gpuBlockSize+1; 
                }
                iGPUBlockStart[i]=nGPUBlocksRequired;
                nGPUBlockNeeded[i]=nGPUBLocksForThisBlock;
                nGPUBlocksRequired+=nGPUBLocksForThisBlock;
            }                      
        } 
        
        if (mode==ONE_THREAD_PER_DOT) {
            for (int i=0;i<nBlocks;i++)  {
                // Count number of interactions
                int numberOfDotsInBlock = iGA_nPtBlock0.cpuArray[i]+iGA_nPtBlock1.cpuArray[i];
                int nGPUBLocksForThisBlock=1;                
                if (numberOfDotsInBlock>gpuBlockSize) {
                    nGPUBLocksForThisBlock = ((int)(numberOfDotsInBlock)-1)/gpuBlockSize+1; 
                }
                iGPUBlockStart[i]=nGPUBlocksRequired;
                nGPUBlockNeeded[i]=nGPUBLocksForThisBlock;
                nGPUBlocksRequired+=nGPUBLocksForThisBlock;
            }  
        }
        
        iGA_idBlock.ensureAllocatedSize(nGPUBlocksRequired, false, false);iGA_idBlock.length=nGPUBlocksRequired;
        iGA_offsIntBlock.ensureAllocatedSize(nGPUBlocksRequired, false, false);iGA_offsIntBlock.length=nGPUBlocksRequired;
        IntStream.range(0, nBlocks).parallel().forEach(i -> {
            int indexGPUBlock= iGPUBlockStart[i];
            for (int j=0;j<nGPUBlockNeeded[i];j++) {
                iGA_idBlock.cpuArray[indexGPUBlock]=i;
                iGA_offsIntBlock.cpuArray[indexGPUBlock]=gpuBlockSize*j;
                indexGPUBlock++;
            }
        });
        
        iGA_idBlock.push();
        iGA_offsIntBlock.push();            
        nGPUBlocks = nGPUBlocksRequired;        
    }
    
    public void push(ArrayList<BlockOfDots> blockList) {
            // First step : map blocks into GPU Blocks
            // Memory allocation if necessary
            int nBlocksRequired=0;
            int nDotsRequired=0;
            
            nBlocksRequired=blockList.size();
            iGA_addrStartBlock0.ensureAllocatedSize(nBlocksRequired, false, false);iGA_addrStartBlock0.length=nBlocksRequired;
            iGA_addrStartBlock1.ensureAllocatedSize(nBlocksRequired, false, false);iGA_addrStartBlock1.length=nBlocksRequired;
            iGA_nPtBlock0.ensureAllocatedSize(nBlocksRequired, false, false);iGA_nPtBlock0.length=nBlocksRequired;
            iGA_nPtBlock1.ensureAllocatedSize(nBlocksRequired, false, false);iGA_nPtBlock1.length=nBlocksRequired;
            iGA_blockLevel.ensureAllocatedSize(nBlocksRequired, false, false);iGA_blockLevel.length=nBlocksRequired;
            
            int currentPosInArrayDotsIndex=0;
            
            for (int i=0;i<nBlocksRequired;i++)  {
                BlockOfDots block = blockList.get(i);  
                block.blockKeyIndex=i;
                
                int nPtBl0 = block.dotsInBlock0.size();
                iGA_nPtBlock0.cpuArray[i]=nPtBl0;
                iGA_addrStartBlock0.cpuArray[i]=currentPosInArrayDotsIndex;
                currentPosInArrayDotsIndex+=nPtBl0;
                
                int nPtBl1;
                if (block.blockLevel==0) {nPtBl1=0;} else {nPtBl1 = block.dotsInBlock1.size();}
                iGA_nPtBlock1.cpuArray[i]=nPtBl1;
                iGA_addrStartBlock1.cpuArray[i]=currentPosInArrayDotsIndex;
                currentPosInArrayDotsIndex+=nPtBl1;
            }                
            nDotsRequired=currentPosInArrayDotsIndex;            
  
            iGA_arrayDotsIndexes.ensureAllocatedSize(nDotsRequired, false, false);iGA_arrayDotsIndexes.length=nDotsRequired;
            
            blockList.parallelStream().forEach(block -> {              
                int iBlock=block.blockKeyIndex;
                iGA_blockLevel.cpuArray[iBlock]=block.blockLevel;                
                
                int iDot=iGA_addrStartBlock0.cpuArray[iBlock];//addrStartBlock0[iBlock];
                for (int j=0;j<iGA_nPtBlock0.cpuArray[iBlock];j++) {
                    iGA_arrayDotsIndexes.cpuArray[iDot]=block.dotsInBlock0.get(j).dotIndex;
                    iDot++;
                }
                for (int j=0;j<iGA_nPtBlock1.cpuArray[iBlock];j++) {
                    iGA_arrayDotsIndexes.cpuArray[iDot]=block.dotsInBlock1.get(j).dotIndex;
                    iDot++;
                }
            });
            
            // Hydrate GPU Mem
            iGA_nPtBlock0.push();
            iGA_nPtBlock1.push();
            iGA_addrStartBlock0.push();
            iGA_addrStartBlock1.push();
            iGA_blockLevel.push();
            
            iGA_arrayDotsIndexes.push();
            
            nDots = nDotsRequired;
            nBlocks = blockList.size();
            hasBeenPushed=true;
    }
    
    public ArrayList<BlockOfDots> popArrayOfBlockFromGPU() {
        return null;
    }
    
    public void resetBlocks() {
        nDots=0;
        nBlocks=0;
        iGA_nPtBlock0.setGpuMemToZero();
        iGA_nPtBlock1.setGpuMemToZero();
        iGA_blockLevel.setGpuMemToZero();
        iGA_addrStartBlock0.setGpuMemToZero();
        iGA_addrStartBlock1.setGpuMemToZero();
        iGA_arrayDotsIndexes.setGpuMemToZero();
        iGA_addrStartBlock0.setGpuMemToZero();
        iGA_nPtBlock0.setGpuMemToZero();
        iGA_addrStartBlock1.setGpuMemToZero();
        iGA_nPtBlock1.setGpuMemToZero();
        iGA_blockLevel.setGpuMemToZero();
        iGA_idBlock.setGpuMemToZero();
        iGA_offsIntBlock.setGpuMemToZero();
    }
    
    public ArrayList<BlockOfDots>  pop(ArrayList<DotN> dots_in) {
        // Fetch GPU Mem
        assert hasBeenPushed;
        iGA_nPtBlock0.pop();
        iGA_nPtBlock1.pop();
        iGA_addrStartBlock0.pop();
        iGA_addrStartBlock1.pop();
        iGA_blockLevel.pop();
        iGA_arrayDotsIndexes.pop();
        ArrayList<BlockOfDots> listOfBlocks = new ArrayList<>(nBlocks);       
        for (int idBlock=0;idBlock<nBlocks;idBlock++) {      
            int nPtBl0 = iGA_nPtBlock0.cpuArray[idBlock];
            int nPtBl1 = iGA_nPtBlock1.cpuArray[idBlock];
            BlockOfDots curBlock = new BlockOfDots(iGA_blockLevel.cpuArray[idBlock],nPtBl0,nPtBl1);
            int addr_Pt = iGA_addrStartBlock0.cpuArray[idBlock];
            for (int j = 0;j<nPtBl0;j++) {  
                DotN dn = dots_in.get(iGA_arrayDotsIndexes.cpuArray[addr_Pt+j]);
                curBlock.dotsInBlock0.add(dn);
            }
            addr_Pt = iGA_addrStartBlock1.cpuArray[idBlock];
            for (int j = 0;j<nPtBl1;j++) {                
                curBlock.dotsInBlock1.add(dots_in.get(iGA_arrayDotsIndexes.cpuArray[addr_Pt+j]));
            }            
            listOfBlocks.add(curBlock);
        }        
        return listOfBlocks;
    }
    
    public void freeMem() {
        hasBeenPushed=false;
        iGA_arrayDotsIndexes.freeMem(); 
        iGA_addrStartBlock0.freeMem(); 
        iGA_addrStartBlock1.freeMem(); 
        iGA_nPtBlock0.freeMem(); 
        iGA_nPtBlock1.freeMem(); 
        iGA_blockLevel.freeMem();
        iGA_idBlock.freeMem();
        iGA_offsIntBlock.freeMem(); 
    }    
}

