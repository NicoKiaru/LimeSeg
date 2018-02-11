/*

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
                    // Output values 
                    Pointer.to(iGA_WhatToDoWithTheseBlocks.gpuArray)
*/

extern "C"
__global__ void sortBlocks(// Old tree specs
						   int* nPtBl0,
                           int* nPtBl1,
                           int* blLev,
							// New tree specs
							int* nPtBlPos,
							int* nPtBlNeg,
							int* nPtBlMid0,
							int* nPtBlMid1,
							int* newBlockCvg,
							int* newBlockLvl,
							int nDots,
							int nBlocks, 
							int minInteract,
							int minPointsToKeep,
							// Output
							int* whatToDo, 
							int* nPtKeep, 
							int* nBlocsKeep,
							int* addrPt,
							int* addrBloc
						)
{   
	int idBloc = blockIdx.x*blockDim.x+threadIdx.x;
	if (idBloc<nBlocks) {
		// do stuff
		// first compute the number of interactions before split
		float nInteractBefore=0;
		float nPtB0 = (float)(nPtBl0[idBloc]);
		float nPtB1 = (float)(nPtBl1[idBloc]);
		int level = blLev[idBloc];
		
		// now compute the number of interaction after spliting blocks
		//__int2float_rd
		float nInteractSubBlockNeg;
		float nInteractSubBlockPos;
		int nPtBlTotPos = nPtBlPos[idBloc]+nPtBlPos[idBloc+nBlocks];
		int nPtBlTotNeg = nPtBlNeg[idBloc]+nPtBlNeg[idBloc+nBlocks];
		int nPtBlTotMid0 = nPtBlMid0[idBloc]+nPtBlMid0[idBloc+nBlocks];
		int nPtBlTotMid1 = nPtBlMid1[idBloc]+nPtBlMid1[idBloc+nBlocks];
		
		if (level>0) {
			nInteractBefore = nPtB0*nPtB1;
		} else {
			nInteractBefore=0.5*nPtB0*(nPtB0+1);
			nPtBlTotPos = nPtBlPos[idBloc];
			nPtBlTotNeg = nPtBlNeg[idBloc];
		}
		
		float nInteractSubBlockMid0=(float)(nPtBlMid0[idBloc])*(float)(nPtBlMid0[idBloc+nBlocks]);//*(newBlockCvg[4*idBloc+2]);
		float nInteractSubBlockMid1=(float)(nPtBlMid1[idBloc])*(float)(nPtBlMid1[idBloc+nBlocks]);//*(newBlockCvg[4*idBloc+3]);
		//printf("nInteractSubBlockMid0 = %f, car nPtBlMid0Bl0 = %i, et nPtBlMid0Bl1 = %i, gpubloc = %i, bloc = %i \n",nInteractSubBlockMid0,nPtBlMid0[idBloc],nPtBlMid0[idBloc+nBlocks],blockIdx.x,idBloc);
		
		//printf("nInteractSubBlockMid1 = %f, car nPtBlMid1Bl0 = %i, et nPtBlMid1Bl1 = %i, gpubloc = %i, bloc = %i \n",nInteractSubBlockMid1,nPtBlMid1[idBloc],nPtBlMid1[idBloc+nBlocks],blockIdx.x,idBloc);
		if (level==0) {
			nInteractSubBlockNeg=(float)(nPtBlNeg[idBloc])*(float)(nPtBlNeg[idBloc]+1)*0.5*(newBlockCvg[4*idBloc+0]);
			nInteractSubBlockPos=(float)(nPtBlPos[idBloc])*(float)(nPtBlPos[idBloc]+1)*0.5*(newBlockCvg[4*idBloc+1]);
			//nPtB0=
		} else {
			nInteractSubBlockNeg=(float)(nPtBlNeg[idBloc])*(float)(nPtBlNeg[idBloc+nBlocks])*(newBlockCvg[4*idBloc+0]);
			nInteractSubBlockPos=(float)(nPtBlPos[idBloc])*(float)(nPtBlPos[idBloc+nBlocks])*(newBlockCvg[4*idBloc+1]);
		}

		newBlockLvl[4*idBloc+0]=level; // bloc Neg
		newBlockLvl[4*idBloc+1]=level; // bloc Pos
		newBlockLvl[4*idBloc+2]=level+1; // bloc Mid0
		newBlockLvl[4*idBloc+3]=level+1; // bloc Mid1
		
		float nInteractAfter = nInteractSubBlockNeg+nInteractSubBlockPos+nInteractSubBlockMid0+nInteractSubBlockMid1;
		// s'il y a plus d'interaction apres la coupure qu'avant ou s'il y a moins d'une certaine quantité d'interaction : option 0 : copy directement les blocs enfants vers le final
		// default = trash
		whatToDo[4*idBloc+0]=3;
		whatToDo[4*idBloc+1]=3;
		whatToDo[4*idBloc+2]=3;
		whatToDo[4*idBloc+3]=3;
		// KEEP = 1
		// SPLIT = 0
		// DISCARD = 2
		// TRASH = 3
		if (nInteractAfter>nInteractBefore) {
			// KEEP ALL = 1
			whatToDo[4*idBloc+0]=1;
			whatToDo[4*idBloc+1]=1;
			whatToDo[4*idBloc+2]=1;
			whatToDo[4*idBloc+3]=1;
		} else {
			if (nInteractSubBlockPos>0) {
				if ((nInteractSubBlockPos>=minInteract)&&((nPtBlPos[idBloc]+nPtBlPos[idBloc+nBlocks])<(nPtB0+nPtB1))) {
					// SPLIT = 0
					whatToDo[4*idBloc+1]=0;
				} else {
					// KEEP = 1
					whatToDo[4*idBloc+1]=1;	
				}
			} else {
				if ((nPtBlPos[idBloc]+nPtBlPos[idBloc+nBlocks])>minPointsToKeep) {
					// DISCARD = 2
					whatToDo[4*idBloc+1]=2;
				} else {
					// TRASH = 3
					whatToDo[4*idBloc+1]=3;
				}
			}
			if (nInteractSubBlockNeg>0) {
				if ((nInteractSubBlockNeg>=minInteract)&&((nPtBlNeg[idBloc]+nPtBlNeg[idBloc+nBlocks])<(nPtB0+nPtB1))) {
					// SPLIT = 0
					whatToDo[4*idBloc+0]=0;
				} else {
					// KEEP = 1
					whatToDo[4*idBloc+0]=1;
				}
			} else {
				if ((nPtBlNeg[idBloc]+nPtBlNeg[idBloc+nBlocks])>minPointsToKeep) {
					// DISCARD = 2
					whatToDo[4*idBloc+0]=2;
				} else {
					// TRASH = 3
					whatToDo[4*idBloc+0]=3;					
				}
			}
			if (nInteractSubBlockMid0>0) {
				if ((nInteractSubBlockMid0>=minInteract)&&((nPtBlMid0[idBloc]+nPtBlMid0[idBloc+nBlocks])<(nPtB0+nPtB1))) {
					// SPLIT = 0
					whatToDo[4*idBloc+2]=0;
				} else {
					// KEEP = 1
					whatToDo[4*idBloc+2]=1;
				}
			} else {
				// trash
				whatToDo[4*idBloc+2]=3;		
			}
			if (nInteractSubBlockMid1>0) {
				if ((nInteractSubBlockMid1>=minInteract)&&((nPtBlMid1[idBloc]+nPtBlMid1[idBloc+nBlocks])<(nPtB0+nPtB1))) {
					// SPLIT = 0
					whatToDo[4*idBloc+3]=0;
				} else {
					// KEEP = 1
					whatToDo[4*idBloc+3]=1;
				}
			} else {
				// TRASH = 3
				//printf("Trash Mid1 \n");
				whatToDo[4*idBloc+3]=3;		
			}
		}
		//printf("Je suis le Pt %i et j'appart au bloc %i, sachant qu'on est dans le blocGPU %i \n",id_pt,id_bloc, blockIdx.x);
		int indexNewBloc = 4*idBloc+0;
		//printf("Je suis le Bloc %i et ce qu'on doit faire c'est %i pour le nouveau bloc %i \n",idBloc,whatToDo[indexNewBloc], indexNewBloc);
		addrBloc[indexNewBloc]=atomicAdd(& nBlocsKeep[whatToDo[indexNewBloc]], 1          );
		addrPt [indexNewBloc]=atomicAdd(& nPtKeep[whatToDo[indexNewBloc]]   , nPtBlTotNeg);
		//printf("addrBloc = %i \n", addrBloc[indexNewBloc]);
		indexNewBloc = 4*idBloc+1;
		//printf("Je suis le Bloc %i et ce qu'on doit faire c'est %i pour le nouveau bloc %i \n",idBloc,whatToDo[indexNewBloc], indexNewBloc);
		addrBloc[indexNewBloc]=atomicAdd(& nBlocsKeep[whatToDo[indexNewBloc]], 1          );
		addrPt [indexNewBloc]=atomicAdd(& nPtKeep[whatToDo[indexNewBloc]]   , nPtBlTotPos);
		indexNewBloc = 4*idBloc+2;
		//printf("Je suis le Bloc %i et ce qu'on doit faire c'est %i pour le nouveau bloc %i \n",idBloc,whatToDo[indexNewBloc], indexNewBloc);
		addrBloc[indexNewBloc]=atomicAdd(& nBlocsKeep[whatToDo[indexNewBloc]], 1          );
		addrPt [indexNewBloc]=atomicAdd(& nPtKeep[whatToDo[indexNewBloc]]   , nPtBlTotMid0);
		indexNewBloc = 4*idBloc+3;
		//printf("Je suis le Bloc %i et ce qu'on doit faire c'est %i pour le nouveau bloc %i \n",idBloc,whatToDo[indexNewBloc], indexNewBloc);
		addrBloc[indexNewBloc]=atomicAdd(& nBlocsKeep[whatToDo[indexNewBloc]], 1          );
		addrPt [indexNewBloc]=atomicAdd(& nPtKeep[whatToDo[indexNewBloc]]   , nPtBlTotMid1);		
		// s'il y a zero interaction et plus d'une certaine quantité de points -> va vers les ignored : option 1
		// s'il y a zero interaction et moins d'une certaine quantité de pointe -> trash : option 2			
	}
}
