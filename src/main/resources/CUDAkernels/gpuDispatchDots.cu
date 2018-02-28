extern "C"
__global__ void dispatchDots(
						   //Tree specs
						   // per Block In
						   int* dotIndexes, 
                           int* stBl0, int* nPtBl0,
						   int* stBl1, int* nPtBl1,
						   int* blLevel,
						   // per GPU Block In
						   int* idBl, 
						   int* offsBl,
						   // input values, per dot
						   int* rkBlPos,
						   int* rkBlNeg,
						   int* rkBlMid0,
						   int* rkBlMid1,
						   // input value, per Blocks Out
						   int* nPtBlPos,
						   int* nPtBlNeg,
						   int* nPtBlMid0,
						   int* nPtBlMid1,
						   int nBlocks,
						   int nDots, 
						   int* whatToDo, 
						   int* addrPt,
						   int* addrBloc,
						   int* newBlockLvl,
						   
						   // bloc split				   	
						   int* blKeep_dotIndexes, 
                           int* blKeep_stBl0, int* blKeep_nPtBl0,
						   int* blKeep_stBl1, int* blKeep_nPtBl1,
						   int* blKeep_blLevel,
						   
						   //bloc keep						   
						   int* blFinal_dotIndexes, 
                           int* blFinal_stBl0, int* blFinal_nPtBl0,
						   int* blFinal_stBl1, int* blFinal_nPtBl1,
						   int* blFinal_blLevel,
						   
						   // bloc discard
						   int* blDiscard_dotIndexes, 
                           int* blDiscard_stBl0, int* blDiscard_nPtBl0,
						   int* blDiscard_stBl1, int* blDiscard_nPtBl1,
						   int* blDiscard_blLevel
						   
						)
{
	extern __shared__ int array[]; 
	int* whatTD = (int*)&array[7];
	int* addrPtSh = (int*)&whatTD[4];
	int* addrBlSh = (int*)&addrPtSh[4];
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
		 
		 for (int i=0;i<4;i++) {
			 whatTD[i]=whatToDo[4*idBloc+i];
			 addrPtSh[i]=addrPt[4*idBloc+i];
			 addrBlSh[i]=addrBloc[4*idBloc+i];
			 /*if (array[0]==0) {
				printf("BlocIni= %i; NBloc= %i; AddrPt= %i; AddrBl= %i; WTd= %i \n",idBloc, i, addrPtSh[i], addrBlSh[i], whatTD[i] );
			 }*/
		 }	 
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
		// Oki, copy dots at the proper location
		int addr_pt = startIndexBl0+ptToBeComputed;
		int id_pt=dotIndexes[addr_pt];
		//if (id_pt<0) {printf("Ca joue pas \n");}
		int inBloc1 = (ptToBeComputed>=nPtBlock0);
		int rK, wTD;
		//int* tabDest;
		//int isSet=0;
		//int shouldBeSomeWhere = 0;
		// Let's handle bloc Neg = 0
		wTD=whatTD[0];
		if (wTD!=3) {
			//shouldBeSomeWhere=1;
			rK=rkBlNeg[addr_pt];//+inBloc1*nDots];
			if (rK!=-1) {
				//printf("on a mis %i qui vaut \n", (addrPtSh[0]+rK+inBloc1*nPtBlNeg[idBloc]), id_pt);
				//isSet=1;
				if (wTD==0) blKeep_dotIndexes[addrPtSh[0]+rK]=id_pt;//+inBloc1*nPtBlNeg[idBloc]]=id_pt;
				if (wTD==1) blFinal_dotIndexes[addrPtSh[0]+rK]=id_pt;//+inBloc1*nPtBlNeg[idBloc]]=id_pt;
				if (wTD==2) blDiscard_dotIndexes[addrPtSh[0]+rK]=id_pt;//+inBloc1*nPtBlNeg[idBloc]]=id_pt;
			}
			rK=rkBlNeg[addr_pt+nDots];
			if (rK!=-1) {
				//printf("on a mis %i qui vaut \n", (addrPtSh[0]+rK+inBloc1*nPtBlNeg[idBloc]), id_pt);
				//isSet=1;
				if (wTD==0) blKeep_dotIndexes[addrPtSh[0]+rK+nPtBlNeg[idBloc]]=id_pt;//+inBloc1*nPtBlNeg[idBloc]]=id_pt;
				if (wTD==1) blFinal_dotIndexes[addrPtSh[0]+rK+nPtBlNeg[idBloc]]=id_pt;//+inBloc1*nPtBlNeg[idBloc]]=id_pt;
				if (wTD==2) blDiscard_dotIndexes[addrPtSh[0]+rK+nPtBlNeg[idBloc]]=id_pt;//+inBloc1*nPtBlNeg[idBloc]]=id_pt;
			}
		}
		wTD=whatTD[1];
		if (wTD!=3) {
			//shouldBeSomeWhere=1;
			rK=rkBlPos[addr_pt];//+inBloc1*nDots];
			if (rK!=-1) {
				//printf("on a mis %i qui vaut %i\n", (addrPtSh[1]+rK+inBloc1*nPtBlPos[idBloc]), id_pt);
				//isSet=1;
				if (wTD==0) blKeep_dotIndexes[addrPtSh[1]+rK]=id_pt;
				if (wTD==1) blFinal_dotIndexes[addrPtSh[1]+rK]=id_pt;
				if (wTD==2) blDiscard_dotIndexes[addrPtSh[1]+rK]=id_pt;
			}
			rK=rkBlPos[addr_pt+nDots];//+inBloc1*nDots];
			if (rK!=-1) {
				//printf("on a mis %i qui vaut %i\n", (addrPtSh[1]+rK+inBloc1*nPtBlPos[idBloc]), id_pt);
				//isSet=1;
				if (wTD==0) blKeep_dotIndexes[addrPtSh[1]+rK+nPtBlPos[idBloc]]=id_pt;
				if (wTD==1) blFinal_dotIndexes[addrPtSh[1]+rK+nPtBlPos[idBloc]]=id_pt;
				if (wTD==2) blDiscard_dotIndexes[addrPtSh[1]+rK+nPtBlPos[idBloc]]=id_pt;
			}
		}
		wTD=whatTD[2];
		if (wTD!=3) {
			//shouldBeSomeWhere=1;
			rK=rkBlMid0[addr_pt];
			if (rK!=-1) {
				//printf("on a mis %i qui vaut %i\n",(addrPtSh[2]+rK), id_pt);
				//isSet=1;
				if (wTD==0) blKeep_dotIndexes[addrPtSh[2]+rK]=id_pt;
				if (wTD==1) blFinal_dotIndexes[addrPtSh[2]+rK]=id_pt;
				if (wTD==2) blDiscard_dotIndexes[addrPtSh[2]+rK]=id_pt;
			}
			rK=rkBlMid0[addr_pt+nDots];
			if (rK!=-1) {
				//printf("on a mis %i qui vaut %i\n", (addrPtSh[2]+rK+nPtBlMid0[idBloc]), id_pt);
				//isSet=1;
				if (wTD==0) blKeep_dotIndexes[addrPtSh[2]+rK+nPtBlMid0[idBloc]]=id_pt;
				if (wTD==1) blFinal_dotIndexes[addrPtSh[2]+rK+nPtBlMid0[idBloc]]=id_pt;
				if (wTD==2) blDiscard_dotIndexes[addrPtSh[2]+rK+nPtBlMid0[idBloc]]=id_pt;
			}
		}
		wTD=whatTD[3];
		if (wTD!=3) {
			//shouldBeSomeWhere=1;
			rK=rkBlMid1[addr_pt];
			if (rK!=-1) {
				//printf("on a mis %i qui vaut %i\n", (addrPtSh[3]+rK), id_pt);
				//isSet=1;
				if (wTD==0) blKeep_dotIndexes[addrPtSh[3]+rK]=id_pt;
				if (wTD==1) blFinal_dotIndexes[addrPtSh[3]+rK]=id_pt;
				if (wTD==2) blDiscard_dotIndexes[addrPtSh[3]+rK]=id_pt;
			}
			rK=rkBlMid1[addr_pt+nDots];
			if (rK!=-1) {
				//printf("on a mis %i qui vaut %i\n", (addrPtSh[3]+rK+nPtBlMid1[idBloc]), id_pt);
				//isSet=1;
				if (wTD==0) blKeep_dotIndexes[addrPtSh[3]+rK+nPtBlMid1[idBloc]]=id_pt;
				if (wTD==1) blFinal_dotIndexes[addrPtSh[3]+rK+nPtBlMid1[idBloc]]=id_pt;
				if (wTD==2) blDiscard_dotIndexes[addrPtSh[3]+rK+nPtBlMid1[idBloc]]=id_pt;
			}
		}
		//if ((isSet==0)&&(shouldBeSomeWhere==1)) {printf("De bleu! Le point %i n'a été mis nulle part! [%i, %i, %i, %i]\n", id_pt, whatTD[0], whatTD[1], whatTD[2], whatTD[3]);}
	}
	if ((iThread==0)&&(offsPt==0)) {
		// needs to fill bloc properties
		for (int i=0;i<4;i++) {
			int wTD = wTD=whatTD[i];
			int idNewBloc = addrBlSh[i];
			
			int nPtInBloc0, nPtInBloc1;
			if (i==0) {
				nPtInBloc0 = nPtBlNeg[idBloc];
				nPtInBloc1 = nPtBlNeg[idBloc+nBlocks];
			}
			if (i==1) {
				nPtInBloc0 = nPtBlPos[idBloc];
				nPtInBloc1 = nPtBlPos[idBloc+nBlocks];
			}
			if (i==2) {
				nPtInBloc0 = nPtBlMid0[idBloc];
				nPtInBloc1 = nPtBlMid0[idBloc+nBlocks];
			}
			if (i==3) {
				nPtInBloc0 = nPtBlMid1[idBloc];
				nPtInBloc1 = nPtBlMid1[idBloc+nBlocks];
			}
			//printf("\n idNewBloc = %i, on en fait %i \n nPtInBloc0 = %i, nPtInBloc1 = %i , addrPtSh = %i \n",idNewBloc, wTD,nPtInBloc0,nPtInBloc1, addrPtSh[i]);
			if (wTD==0) {
				//SPLIT
				//printf("SPLIT!!\n");
				blKeep_stBl0[idNewBloc]=addrPtSh[i];
				blKeep_nPtBl0[idNewBloc]=nPtInBloc0;
				blKeep_stBl1[idNewBloc]=addrPtSh[i]+nPtInBloc0;
				blKeep_nPtBl1[idNewBloc]=nPtInBloc1;
				blKeep_blLevel[idNewBloc]=newBlockLvl[4*idBloc+i];
			}
			if (wTD==1) {
				//KEEP
				blFinal_stBl0[idNewBloc]=addrPtSh[i];
				blFinal_nPtBl0[idNewBloc]=nPtInBloc0;
				blFinal_stBl1[idNewBloc]=addrPtSh[i]+nPtInBloc0;
				blFinal_nPtBl1[idNewBloc]=nPtInBloc1;
				blFinal_blLevel[idNewBloc]=newBlockLvl[4*idBloc+i];
			}
			if (wTD==2) {
				//DISCARD
				blDiscard_stBl0[idNewBloc]=addrPtSh[i];
				blDiscard_nPtBl0[idNewBloc]=nPtInBloc0;
				blDiscard_stBl1[idNewBloc]=addrPtSh[i]+nPtInBloc0;
				blDiscard_nPtBl1[idNewBloc]=nPtInBloc1;
				blDiscard_blLevel[idNewBloc]=newBlockLvl[4*idBloc+i];
			}
			
		}
	}
}