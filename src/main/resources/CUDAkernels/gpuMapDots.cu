extern "C"

/*
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
                    Pointer.to(new int[]{nDots}) // offset for blocks : 0 or 1
            );
*/

__global__ void mapDots(// Dots props
						   float* pX,
                           float* pY,
                           float* pZ,
						   float* sDotRadiusSquared,
						   int* allNeighConverged,
						   //Tree specs
						   // per Block In
						   int* dotIndexes, 
                           int* stBl0, int* nPtBl0,
						   int* stBl1, int* nPtBl1,
						   float* avgPX, 
                           float* avgPY,
                           float* avgPZ,
						   float* dirX,
						   float* dirY,
						   float* dirZ,
						   int* blLevel,
						   // per GPU Block In
						   int* idBl, 
						   int* offsBl,
						   // output values, per dot
						   float* pScalVal,
						   int* rkBlPos,
						   int* rkBlNeg,
						   int* rkBlMid0,
						   int* rkBlMid1,
						   // output value, per Blocks Out
							int* nPtBlPos,
							int* nPtBlNeg,
							int* nPtBlMid0,
							int* nPtBlMid1,
							int* newBlockCvg,
							int nBlocksIn,
							int nDotsIn,
							float sqInteract
						)
{
	extern __shared__ int array[]; 
	float* avg = (float*)&array[7];
	float* dir = (float*)&avg[3];
	// Fetch block data
    int iGPUBlock=blockIdx.x;
	int iThread=threadIdx.x;
	int idBloc;
	if (iThread==0) {
		 idBloc=idBl[iGPUBlock];		 
		 array[0]=offsBl[iGPUBlock];
		 array[1]=stBl0[idBloc];
		 array[2]=nPtBl0[idBloc];
		 array[3]=stBl1[idBloc];		 
		 array[4]=nPtBl1[idBloc];
		 array[5]=blLevel[idBloc];
		 array[6]=idBloc;		 
		 avg[0]=avgPX[idBloc];
		 avg[1]=avgPY[idBloc];
		 avg[2]=avgPZ[idBloc];
		 dir[0]=dirX[idBloc];
		 dir[1]=dirY[idBloc];
		 dir[2]=dirZ[idBloc];
	}	
	__syncthreads();		
	int offsPt = array[0];
	int startIndexBl0 = array[1];	 
	int nPtBlock0 = array[2];
	int startIndexBl1 = array[3]; // useless in fact
	int nPtBlock1 = array[4];
	int blockLevel = array[5];
	int nPts = nPtBlock0 + nPtBlock1;	
	int ptToBeComputed = iThread+offsPt;
	idBloc = array[6];

	if (ptToBeComputed<nPts) {

		// prevents overflow
		
		int addr_pt = startIndexBl0+ptToBeComputed;
		int id_pt=dotIndexes[addr_pt];
		// put all negative
		rkBlNeg[addr_pt]=-1;
		rkBlPos[addr_pt]=-1;
		rkBlMid0[addr_pt]=-1;
		rkBlMid1[addr_pt]=-1;
		rkBlNeg[addr_pt+nDotsIn]=-1;
		rkBlPos[addr_pt+nDotsIn]=-1;
		rkBlMid0[addr_pt+nDotsIn]=-1;
		rkBlMid1[addr_pt+nDotsIn]=-1;
		//
		int cvg = allNeighConverged[id_pt]; // 1 if all converged; 0 otherwise
		int dx = pX[id_pt]-avg[0];
		int dy = pY[id_pt]-avg[1];
		int dz = pZ[id_pt]-avg[2];
		float pScal = (dx*dir[0]+dy*dir[1]+dz*dir[2])/sqInteract;
		pScalVal[id_pt] = pScal;
		int inBloc1 = (ptToBeComputed>=nPtBlock0);
		float sDRadius = sqrtf(sDotRadiusSquared[id_pt]);
		if (pScal<0) {
			rkBlNeg[addr_pt+inBloc1*nDotsIn] = atomicAdd(& nPtBlNeg[idBloc+inBloc1*nBlocksIn], 1);
			if (cvg==0) {newBlockCvg[4*idBloc]=1;} // quick convergence block test*/
			if ((sDRadius>0)&&(pScal+sDRadius/sqInteract>=0)) {				
				rkBlPos[addr_pt] = atomicAdd(& nPtBlPos[idBloc], 1);
			}
		} else {
			rkBlPos[addr_pt+inBloc1*nDotsIn] = atomicAdd(& nPtBlPos[idBloc+inBloc1*nBlocksIn], 1);
			if (cvg==0) {newBlockCvg[4*idBloc+1]=1;} // quick convergence block test*/
			if ((sDRadius>0)&&(pScal-sDRadius/sqInteract<0)) {
				rkBlNeg[addr_pt] = atomicAdd(& nPtBlNeg[idBloc], 1);
			}
		}
		if (sDRadius==0) {
			// not a superdot
			if (blockLevel==0) {
				if ((pScal>-1)&&(pScal<0)) {
					rkBlMid0[addr_pt] = atomicAdd(& nPtBlMid0[idBloc], 1);
					if (cvg==0) {newBlockCvg[4*idBloc+2]=1;} // quick convergence block test
				}
				if ((pScal<1)&&(pScal>=0)) {
					rkBlMid0[addr_pt+nDotsIn] = atomicAdd(& nPtBlMid0[idBloc+nBlocksIn], 1);
					if (cvg==0) {newBlockCvg[4*idBloc+2]=1;} // quick convergence block test
				}
			} else {
				if ((pScal>-1)&&(pScal<0)) {
					if (inBloc1) {
						//Mid1Block0
						rkBlMid1[addr_pt] = atomicAdd(& nPtBlMid1[idBloc], 1);
						if (cvg==0) {newBlockCvg[4*idBloc+3]=1;}
					} else {
						//Mid0Block0
						rkBlMid0[addr_pt] = atomicAdd(& nPtBlMid0[idBloc], 1);
						if (cvg==0) {newBlockCvg[4*idBloc+2]=1;}
					}					
				}
				if ((pScal<1)&&(pScal>=0)) {
					if (inBloc1) {
						//Mid0Block1
						rkBlMid0[addr_pt+nDotsIn] = atomicAdd(& nPtBlMid0[idBloc+nBlocksIn], 1);
						if (cvg==0) {newBlockCvg[4*idBloc+2]=1;}
					} else {
						//Mid1Block1
						rkBlMid1[addr_pt+nDotsIn] = atomicAdd(& nPtBlMid1[idBloc+nBlocksIn], 1);
						if (cvg==0) {newBlockCvg[4*idBloc+3]=1;}
					}
				}
			}
		}	
	}
}



































