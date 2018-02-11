extern "C"
__global__ void computeAVG(// Dots props
						   float* pX,
                           float* pY,
                           float* pZ,
						   //Tree specs
						   // per Block
						   int* dotIndexes, 
                           int* stBl0, int* nPtBl0,
						   int* stBl1, int* nPtBl1,
						   float* avgPX, 
                           float* avgPY,
                           float* avgPZ,
						   // per GPU Block
						   int* idBl, int* offsBl)
{
    extern __shared__ int array[]; 
	float* posdata_add = (float*)&array[5];
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
		 //printf("Bah je suis le bloc %i, je process le bloc %i, et je démarre à %i \n", iGPUBlock, idBloc, array[1]);
	}
	
	__syncthreads();	
	
	int offsPt = array[0];
	int startIndexBl0 = array[1];	 
	int nPtBlock0 = array[2];
	int startIndexBl1 = array[3]; // useless in fact
	int nPtBlock1 = array[4];
	int nPts = nPtBlock0 + nPtBlock1;
	
	int ptToBeComputed = iThread+offsPt;
	int cPos=3*iThread;
	
	idBloc=idBl[iGPUBlock]; // to be removed!!!
	
	if (ptToBeComputed<nPts) {
		int id_pt=dotIndexes[startIndexBl0+ptToBeComputed];
		//if (id_pt<0) {printf("Probleme!!! id_pt = %i; bloc %i; thread %i; gpubloc %i \n", id_pt, idBloc, threadIdx.x, blockIdx.x);}
		posdata_add[cPos+0]=pX[id_pt];
        posdata_add[cPos+1]=pY[id_pt];
        posdata_add[cPos+2]=pZ[id_pt];
	} else {
	    posdata_add[cPos+0]=0;
        posdata_add[cPos+1]=0;
        posdata_add[cPos+2]=0;
	}
	
	__syncthreads();
    // All data copied to shared Mem
    for (unsigned int s=blockDim.x/2;s>0;s>>=1) 
    {
        if (iThread<s) {       
            int tShift=3*s; // not sure of that...
            posdata_add[cPos+0]+=posdata_add[cPos+tShift];
			posdata_add[cPos+1]+=posdata_add[cPos+tShift+1];
            posdata_add[cPos+2]+=posdata_add[cPos+tShift+2];
        }
        __syncthreads();
    }          
    if (iThread==0) {
		float nPtInBlock = nPts;
        atomicAdd(& avgPX[idBloc], posdata_add[0]/nPtInBlock);
        atomicAdd(& avgPY[idBloc], posdata_add[1]/nPtInBlock);
        atomicAdd(& avgPZ[idBloc], posdata_add[2]/nPtInBlock);
    }     
}
