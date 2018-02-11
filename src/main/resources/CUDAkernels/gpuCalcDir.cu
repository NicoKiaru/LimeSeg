extern "C"
__global__ void calcDir(// Dots props
						   float* pX,
                           float* pY,
                           float* pZ,
						   //Tree specs
						   // per Block
						   //int* dotIndexes,
						   float* avgPX, 
                           float* avgPY,
                           float* avgPZ,
						   int* idFurthest,
						   // per GPU Block
						   // output values, per block
						   float* dirX, 
                           float* dirY,
                           float* dirZ,
						   float nBlocs
						)
{   
	int idBloc = blockIdx.x*blockDim.x+threadIdx.x;
	if (idBloc<nBlocs) {
		int mx=avgPX[idBloc];
		int my=avgPY[idBloc];
		int mz=avgPZ[idBloc];
		int idPtFurthest = idFurthest[idBloc];
		float dx=pX[idPtFurthest]-mx;
		float dy=pY[idPtFurthest]-my;
		float dz=pZ[idPtFurthest]-mz;
		float dist = sqrtf(dx*dx+dy*dy+dz*dz);
		dirX[idBloc]=dx/dist;
		dirY[idBloc]=dy/dist;
		dirZ[idBloc]=dz/dist;		
	}
}
