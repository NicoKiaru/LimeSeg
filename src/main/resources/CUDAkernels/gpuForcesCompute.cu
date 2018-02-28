extern "C"

__global__ void forceCompute(float* pX,float* pY,float* pZ, 
                             float* nX,float* nY,float* nZ,
                             float* FX,float* FY,float* FZ,
                             float* RFX,float* RFY,float* RFZ,
                             float* MX,float* MY,float* MZ,
			     float* sDRadius_squared, float* relaxed,
                             int* NN,
                             int* CID,							 
                             int* hasConverged,							 
                             int* allNeighborsHaveConverged,
							 int* allNeighborsHaveConvergedPreviously,
                             // KD Tree specs
							 // per Block
                             int* stBl0, int* nPtBl0,
							 int* stBl1, int* nPtBl1,
                             int* blLevel,  
							 // per GPU Block
							 int* idBl, int* offsBl,
							 // for all
							 int* dotIndexes,
                             // Integration specs
                             float r_0,
                             float k_align,
                             float k_bend,
                             float radiusTresholdInteract,
                             float ka,
                             float pa,
                             float pr,
                             float maxDisplacementPerStep
                             )                           
                             
{    
     extern __shared__ int array[];
	 int iGPUBlock=blockIdx.x;
	 int iThread=threadIdx.x;
	 
	 if (iThread==0) {
		 int idBloc=idBl[iGPUBlock];		 
		 array[0]=offsBl[iGPUBlock];
		 array[1]=stBl0[idBloc];
		 array[2]=nPtBl0[idBloc];
		 array[3]=stBl1[idBloc];		 
		 array[4]=nPtBl1[idBloc];		 
		 array[5]=blLevel[idBloc];
	 }
	 __syncthreads();
	  
	 int offsInteraction = array[0];
	 int startIndexBl0 = array[1];	 
	 int nPtBlock0 = array[2];
	 int startIndexBl1 = array[3];
	 int nPtBlock1 = array[4];
	 int blockLevel = array[5];
	 
	 int interactionToBeComputed = iThread+offsInteraction;
	 int iPt1=-1;
	 int iPt2=-1;
	 int totalNumberOfInteractions = (blockLevel==0)*(nPtBlock0*(nPtBlock0-1)/2)+
								     (blockLevel>0)*nPtBlock0*nPtBlock1;
	 if (interactionToBeComputed<totalNumberOfInteractions) {
		// It's not an overflow
		int ind0, ind1;
		if (blockLevel==0) {
			ind0 = nPtBlock0 - 2 - floor(sqrtf(-8*interactionToBeComputed + 4*nPtBlock0*(nPtBlock0-1)-7)/2.0 - 0.5);
			ind1 = interactionToBeComputed + ind0 + 1 - nPtBlock0*(nPtBlock0-1)/2 + (nPtBlock0-ind0)*((nPtBlock0-ind0)-1)/2;
			startIndexBl1=startIndexBl0;
		} else {
			ind1=interactionToBeComputed/nPtBlock0;
			ind0=interactionToBeComputed-ind1*nPtBlock0;
		}	 
		iPt1 = dotIndexes[startIndexBl0+ind0];
		iPt2 = dotIndexes[startIndexBl1+ind1];
		// that's a correct interaction
		float dx=pX[iPt2]-pX[iPt1];
        float dy=pY[iPt2]-pY[iPt1];
        float dz=pZ[iPt2]-pZ[iPt1];		
		float dist_Squared=(dx*dx+dy*dy+dz*dz);        
		int idC1=CID[iPt1];
		int idC2=CID[iPt2];		
		if ((idC1!=-1)&&(idC2!=-1)) {			
			if (dist_Squared<radiusTresholdInteract*radiusTresholdInteract*r_0*r_0) { 
				if (hasConverged[iPt1]==0) {allNeighborsHaveConverged[iPt2]=0;}
				if (hasConverged[iPt2]==0) {allNeighborsHaveConverged[iPt1]=0;}
                //float rfx, rfy, rfz;
				float kr=pa*ka/pr*powf(1,pr-pa);
                float dist = sqrtf(dist_Squared);
                float r=dist/r_0;
                float f_rep=(-pr*kr/powf(r, pr+1))*(r>1.0)-(r<=1.0)*((1.0-r)*maxDisplacementPerStep+pr*kr/powf(1.0, pr+1));				
				dx=dx/dist;dy=dy/dist;dz=dz/dist;
                float fx1=dx*f_rep;float fx2=-dx*f_rep;     
                float fy1=dy*f_rep;float fy2=-dy*f_rep;          
                float fz1=dz*f_rep;float fz2=-dz*f_rep; 
				if (idC1==idC2){             
					// iPt1 et 2 sont voisins
					atomicAdd(& NN[iPt1], 1);
					atomicAdd(& NN[iPt2], 1);
					// Neighbor              
                    // iPt1 et 2 sont voisins
                    atomicAdd(& RFX[iPt1], fx1);atomicAdd(& RFX[iPt2], fx2);
                    atomicAdd(& RFY[iPt1], fy1);atomicAdd(& RFY[iPt2], fy2);
                    atomicAdd(& RFZ[iPt1], fz1);atomicAdd(& RFZ[iPt2], fz2);
                              
                    float f_attract=(pa*ka/powf(r, pa+1))*(r>1.0)+(r<=1.0)*(pr*kr/powf(1.0, pr+1));
                    fx1+=dx*f_attract;fx2-=dx*f_attract;      
                    fy1+=dy*f_attract;fy2-=dy*f_attract;        
                    fz1+=dz*f_attract;fz2-=dz*f_attract;
					// Get data
					float nX1=nX[iPt1];float nX2=nX[iPt2];               
					float nY1=nY[iPt1];float nY2=nY[iPt2];               
					float nZ1=nZ[iPt1];float nZ2=nZ[iPt2];

                    float iFlatten = k_align*(dx*(nX1+nX2)+dy*(nY1+nY2)+dz*(nZ1+nZ2));     
                    fx1+=iFlatten*nX1;
                    fy1+=iFlatten*nY1;
                    fz1+=iFlatten*nZ1;
					fx2-=iFlatten*nX2;
                    fy2-=iFlatten*nY2;
                    fz2-=iFlatten*nZ2;
                              
                    float iPerpend1=-k_bend*(dx*nX1+dy*nY1+dz*nZ1);
					float iPerpend2=-k_bend*(dx*nX2+dy*nY2+dz*nZ2);
                    atomicAdd(& MX[iPt1], iPerpend1*dx);atomicAdd(& MX[iPt2], iPerpend2*dx);
                    atomicAdd(& MY[iPt1], iPerpend1*dy);atomicAdd(& MY[iPt2], iPerpend2*dy);
                    atomicAdd(& MZ[iPt1], iPerpend1*dz);atomicAdd(& MZ[iPt2], iPerpend2*dz);
				} else {
                    relaxed[iPt1]=1.0;
					relaxed[iPt2]=1.0;
                }
				atomicAdd(& FX[iPt1], fx1);atomicAdd(& FX[iPt2], fx2);
                atomicAdd(& FY[iPt1], fy1);atomicAdd(& FY[iPt2], fy2);
                atomicAdd(& FZ[iPt1], fz1);atomicAdd(& FZ[iPt2], fz2);
			}
		} else if (idC1*idC2<=0){
			int idSD, idND;
			if (idC1<0) {idSD=iPt1;idND=iPt2;} else {idSD=iPt2;idND=iPt1;}
			if ((dist_Squared-sDRadius_squared[idSD]<0)&&(allNeighborsHaveConvergedPreviously[idND]==0)) {
					// superdot is touched! atomic is unnecessary here
					allNeighborsHaveConverged[idSD]=0;
			}
		}
	}
}
    /*
     if ((iTx<LX)&&(iTy<LY)) {
          
          int iPtx=pStX[blockIdx.x]+iTx;
          int iPty=pStY[blockIdx.x]+iTy;
          
          int compute=0;
          
          if (level==0) {
               compute=1;
          } else {
             int kS1=kSo[iPtx];
             int kS2=kSo[iPty];
             if ((kS1+kS2)==((1<<(level))-1)) {
                  compute=1;
             }
          }
         
          if (compute==1) {              
              int iPt1 = io[iPtx];//threadIdx.x+blockIdx.x*blockDim.x;
              int iPt2 = io[iPty];//threadIdx.y+blockIdx.y*blockDim.y;
              //atomicAdd(& NN[iPt1], 1);
              //float r_0=#r_0;

              float kr=pa*ka/pr*powf(1,pr-pa);
              
              if (iPt1!=iPt2)
              {               
                 float dx=pX[iPt2]-pX[iPt1];
                 float dy=pY[iPt2]-pY[iPt1];
                 float dz=pZ[iPt2]-pZ[iPt1];
                 
                 float dist_Squared=(dx*dx+dy*dy+dz*dz);
                 if (dist_Squared<radiusTresholdInteract*radiusTresholdInteract*r_0*r_0) { 
                    // Get data
                    float nX1=nX[iPt1];float nX2=nX[iPt2];               
                    float nY1=nY[iPt1];float nY2=nY[iPt2];               
                    float nZ1=nZ[iPt1];float nZ2=nZ[iPt2];
                    int idC1=CID[iPt1];
                    int idC2=CID[iPt2];
                    // 
                    float f_rep, f_attract=0;
                    float fx, fy, fz;
                    //float rfx, rfy, rfz;
                    float dist = sqrt(dist_Squared);
                    float r=dist/r_0; 
                    f_rep=(-pr*kr/powf(r, pr+1))*(r>1.0)-(r<=1.0)*((1.0-r)*maxDisplacementPerStep+pr*kr/powf(1.0, pr+1));
                    // for convergence tests
                    
                    if (hasConverged[iPt2]==0) {allNeighborsHaveConverged[iPt1]=0;}
                    //atomicMin(& allNeighborsHaveConverged[iPt1], hasConverged[iPt2]);

                    dx=dx/dist;dy=dy/dist;dz=dz/dist;

                    fx=dx*f_rep;         
                    fy=dy*f_rep;         
                    fz=dz*f_rep;

                    if (idC1==idC2){ //&&(dist_Squared<radiusTresholdNeighbor*r_0*radiusTresholdNeighbor*r_0)) {
                              // Neighbor              
                              // iPt1 et 2 sont voisins
                              atomicAdd(& NN[iPt1], 1);
                              atomicAdd(& RFX[iPt1], fx);
                              atomicAdd(& RFY[iPt1], fy);
                              atomicAdd(& RFZ[iPt1], fz); 
                              
                              f_attract=(pa*ka/powf(r, pa+1))*(r>1.0)+(r<=1.0)*(pr*kr/powf(1.0, pr+1));
                              fx=dx*(f_rep+f_attract);         
                              fy=dy*(f_rep+f_attract);         
                              fz=dz*(f_rep+f_attract);

                              float iFlatten = k_align*(dx*(nX1+nX2)+dy*(nY1+nY2)+dz*(nZ1+nZ2));     
                              fx+=iFlatten*nX1;
                              fy+=iFlatten*nY1;
                              fz+=iFlatten*nZ1;
                              
                              float iPerpend1=-k_bend*(dx*nX1+dy*nY1+dz*nZ1);
                              atomicAdd(& MX[iPt1], iPerpend1*dx);
                              atomicAdd(& MY[iPt1], iPerpend1*dy);
                              atomicAdd(& MZ[iPt1], iPerpend1*dz);
                    } else {
                        relaxed[iPt1]=1.0;
                    }
                    atomicAdd(& FX[iPt1], fx);
                    atomicAdd(& FY[iPt1], fy);
                    atomicAdd(& FZ[iPt1], fz);          
                 }       
              }
         }         
    }*/

                   //rfx=dx*(f_rep);         
                   //rfy=dy*(f_rep);         
                   //rfz=dz*(f_rep);
                   
                   // Attractif
                   // pX[iPt1]=0;
                   // pX[iPt2]=0;   

                   //if (idC1==idC2) {
                   //     f_attract=(pa*ka/powf(r, pa+1))*(r>1.0);
                        //f_attract=0;
                   //}//*prodScalNorm;