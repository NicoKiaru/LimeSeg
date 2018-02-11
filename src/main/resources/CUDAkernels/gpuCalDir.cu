extern "C"

__global__ void calcDir(// Dots props
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
						   int* idBl, int* offsBl,
						   
						   // output values, per block
						   int* idFurthest, float* dMax
								/*float* pX,float* pY,float* pZ, 
                                 float* avgPX, float* avgPY, float* avgPZ,
                                 int* lockBlock, float* dMax, 
                                 int* idFurthest,
                                 int* id_in, int* id_bl_in*/
								 )
{   
    extern __shared__ int array[];    
    float* posAVGBlock = (float*)&array[5]; 
    float* dMaxPt = (float*)&posAVGBlock[3];
    int*   iMaxPt =   (int*)&dMaxPt[blockDim.x];    
    
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
		 posAVGBlock[0]=avgPX[idBloc];
		 posAVGBlock[1]=avgPY[idBloc];
		 posAVGBlock[2]=avgPZ[idBloc];
	}
	__syncthreads();	
	
	int offsPt = array[0];
	int startIndexBl0 = array[1];	 
	int nPtBlock0 = array[2];
	int startIndexBl1 = array[3]; // useless in fact
	int nPtBlock1 = array[4];
	int nPts = nPtBlock0 + nPtBlock1;
	int ptToBeComputed = iThread+offsPt;
	int mx=posAVGBlock[0];
	int my=posAVGBlock[1];
	int mz=posAVGBlock[2];

	if (ptToBeComputed<nPts) {
		int id_pt=dotIndexes[startIndexBl0+ptToBeComputed];
		float xval=(pX[id_pt]-mx);
        float yval=(pY[id_pt]-my);
        float zval=(pZ[id_pt]-mz);                
        dMaxPt[iThread]=xval*xval+yval*yval+zval*zval;
        iMaxPt[iThread]=id_pt;
	} else {
		dMaxPt[iThread]=-1;
        iMaxPt[iThread]=-1;
	}       
    __syncthreads();
    // All data copied to shared Mem
         
    
}
