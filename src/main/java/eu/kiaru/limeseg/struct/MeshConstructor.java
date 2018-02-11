package eu.kiaru.limeseg.struct;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;

import eu.kiaru.limeseg.opt.BlockOfDots;
import eu.kiaru.limeseg.opt.Optimizer;
/**
 * @author Nicolas Chiaruttini
 *	Method to reconstruct mesh from point cloud
 */
public class MeshConstructor {
	
	ArrayList<TriangleN> triangles;
	final ArrayList<DotN> completeListOfDots;
	public static int MAX_CONNECTED_COMPONENT_SIZE=60;
	public static int MAX_SIMPLYCONNECTED_COMPONENT_SIZE=6;
	int finalNumberOfEdges;
	
	public MeshConstructor(ArrayList<DotN> dots_in) {	
		completeListOfDots = new ArrayList<>();
		for (int i=0;i<dots_in.size();i++) {
	            DotN nd = dots_in.get(i);
	            DotN nd_copy = new DotN(new Vector3D(nd.pos.x ,nd.pos.y ,nd.pos.z ),
	            		                new Vector3D(nd.Norm.x,nd.Norm.y,nd.Norm.z));
	            completeListOfDots.add(nd_copy);                
	        }        
		
		for (int i=0;i<completeListOfDots.size();i++) {
            DotN dn;
            dn = completeListOfDots.get(i);
            dn.dotIndex=i;
        }
	}		
	
	void storeEdge(Map<Integer,Map<Integer,TriangleN>> edges, DotN d1, DotN d2) {
    	// this is a potential edge		
		int highIndex,lowIndex;
    	
			if (d1.dotIndex>d2.dotIndex) {
	    		highIndex=d1.dotIndex;
	    		lowIndex=d2.dotIndex;    		
	    	} else {
	    		highIndex=d2.dotIndex;
	    		lowIndex=d1.dotIndex;
	    	}

    	Map<Integer,TriangleN> lowIndexHM;
    	if (edges.containsKey(highIndex)) {
    		lowIndexHM = edges.get(highIndex);
    	} else {
    		lowIndexHM = new HashMap<Integer,TriangleN>();
    		edges.put(highIndex, lowIndexHM);
    	}
    	if (lowIndexHM.get(lowIndex)!=null) {
    	} else {
    		lowIndexHM.put(lowIndex,null);
    	}
    	// can be null, or contains a triangle, or is removed if it contains 2
    }	
	
	int nBlockDone;
	
	public void registerEdges(Map<Integer,Map<Integer,TriangleN>> edges,ArrayList<DotN> dots, float rMax, boolean constructTrianglesByBlock) {
		Optimizer optForMeshConstruct = new Optimizer();
    	dots.forEach(d->{
    				d.reInit();
    				d.allNeighborsHaveConverged=false;
    				d.allNeighborsHaveConvergedPreviously=false;
    			}); // avoids moving dots when optimizing
    	// push the only relevant parameter for mesh construction
    	optForMeshConstruct.dots=dots;
    	optForMeshConstruct.setOptParam("d_0", rMax);
    	optForMeshConstruct.setOptParam("radiusTresholdInteract", 1f);
    	ArrayList<BlockOfDots> iniBlocks = new ArrayList<>();
        BlockOfDots firstBlock=new BlockOfDots(0,dots.size());
        firstBlock.dotsInBlock0=dots;
        // Gives index for GPU purposes
        iniBlocks.add(firstBlock);
        if (optForMeshConstruct.CUDAEnabled) {
        	optForMeshConstruct.CUDAEnabled=false;
        	optForMeshConstruct.buildTreeForNeighborsSearch(iniBlocks,1500);
        	optForMeshConstruct.CUDAEnabled=true;
        } else {
        	optForMeshConstruct.buildTreeForNeighborsSearch(iniBlocks,1500);
        }
    	float rMax2 = (rMax*rMax);
    	optForMeshConstruct.limitInteractAttract = rMax2;
    	// Now hydrates the list of neighbors
    	int nBlocks = optForMeshConstruct.allBlocks.size();
    	// let's do this block by block
    	// We need a two level hashmap to store information about the edges
    	// edges indexes will be A and B, with A>B
    	//System.out.print("Storing edges...");
    	int nBlockTotal =  optForMeshConstruct.allBlocks.size();
    	nBlockDone = 0; 
    	
    	optForMeshConstruct.allBlocks.forEach(block -> {
	    	if (constructTrianglesByBlock) {
	    		Map<Integer,Map<Integer,TriangleN>> partialEdgesSup = buildTrianglesOfBlock(block,rMax2);
	    		partialEdgesSup.values().removeIf(m -> (m.isEmpty()));
	    		partialEdgesSup.forEach((k,v) -> 
	    			edges.merge(k, v, (v1,v2) -> {
	    				v1.putAll(v2);
	    				return v1;
	    			}));
    		} else {
    			if (block.blockLevel!=0) {
    				 for (DotN dn1:block.dotsInBlock0) {
    	                 for (DotN dn2:block.dotsInBlock1) {
    	                 	if (Vector3D.dist2(dn1.pos,dn2.pos)<rMax2) {
    	                 		storeEdge(edges, dn1,dn2);
    	                     }
    	                 }
    	             }
    	         } else {
    				 int blSize = block.dotsInBlock0.size();
    	             for (int i=0;i<blSize-1;i++) {
    	                 DotN dn1 = block.dotsInBlock0.get(i);
    	                 for (int j=i+1;j<blSize;j++) {
    	                     DotN dn2 = block.dotsInBlock0.get(j);
    	                     if (Vector3D.dist2(dn1.pos,dn2.pos)<rMax2) {
    	                     	storeEdge(edges,dn1,dn2);
    	                     }
    	                 }
    	             }             
    			 }
    		}
    	});
	}
	
	public Map<Integer,Map<Integer,TriangleN>> buildTrianglesOfBlock(BlockOfDots block, float rMax2) {
		 Map<Integer,Map<Integer,TriangleN>> edgesSupOfBlock = new HashMap<Integer,Map<Integer,TriangleN>>();
		 if (block.blockLevel!=0) {
			 for (DotN dn1:block.dotsInBlock0) {
                 for (DotN dn2:block.dotsInBlock1) {
                 	if (Vector3D.dist2(dn1.pos,dn2.pos)<rMax2) {
                 		storeEdge(edgesSupOfBlock, dn1,dn2);
                     }
                 }
             }
         } else {
			 int blSize = block.dotsInBlock0.size();
             for (int i=0;i<blSize-1;i++) {
                 DotN dn1 = block.dotsInBlock0.get(i);
                 for (int j=i+1;j<blSize;j++) {
                     DotN dn2 = block.dotsInBlock0.get(j);
                     if (Vector3D.dist2(dn1.pos,dn2.pos)<rMax2) {
                     	storeEdge(edgesSupOfBlock,dn1,dn2);
                     }
                 }
             }             
		 }
		 
		 nBlockDone++;
		 buildTriangles(edgesSupOfBlock);
		 return edgesSupOfBlock;
	}


	Map<Integer,Map<Integer,TriangleN>> getComplementaryEdges(Map<Integer,Map<Integer,TriangleN>> edgesSup) {
		Map<Integer,Map<Integer,TriangleN>> edgesInf = new HashMap<Integer,Map<Integer,TriangleN>>();
        
        edgesSup.keySet().forEach(indexSup -> {
        	Map<Integer,TriangleN> hmap = edgesSup.get(indexSup);
        	hmap.keySet().forEach(indexInf -> {
        		Map<Integer,TriangleN> highIndexHM;
        		if (edgesInf.containsKey(indexInf)) {
        			highIndexHM = edgesInf.get(indexInf);
            	} else {
            		highIndexHM = new HashMap<Integer,TriangleN>();
            		edgesInf.put(indexInf, highIndexHM);
            	}
        		highIndexHM.put(indexSup,edgesSup.get(indexSup).get(indexInf));
        	});
        });
        return edgesInf;
	}
	
	void registerTri(Map<Integer,Map<Integer,TriangleN>> edgesS,
					  Map<Integer,Map<Integer,TriangleN>> edgesI,TriangleN tri) {

		registerEdge(edgesS,edgesI,tri.id1,tri.id2,tri);
		registerEdge(edgesS,edgesI,tri.id2,tri.id3,tri);
		registerEdge(edgesS,edgesI,tri.id3,tri.id1,tri);

	}
	
	
	void registerEdge(Map<Integer,Map<Integer,TriangleN>> edgesS,
					  Map<Integer,Map<Integer,TriangleN>> edgesI, int id1,int id2, TriangleN tri) {
		int indexHigh = id1;
		int indexLow = id2;
		if (indexHigh<indexLow) {
			int temp=indexHigh;
			indexHigh=indexLow;
			indexLow=temp;
		}
		
		if (edgesS.containsKey(indexHigh)) {
			if (edgesS.get(indexHigh).containsKey(indexLow)) {
				if (edgesS.get(indexHigh).get(indexLow)==null) {
					edgesS.get(indexHigh).put(indexLow, tri);
				} else {
					edgesS.get(indexHigh).remove(indexLow);
				}
			} else {
				edgesS.get(indexHigh).put(indexLow, tri);
			}
		} else {
			Map<Integer,TriangleN> lowIndexHM;
    		lowIndexHM = new HashMap<Integer,TriangleN>();
    		lowIndexHM.put(indexLow, tri);
    		edgesS.put(indexHigh, lowIndexHM);
		}
		
		if (edgesI.containsKey(indexLow)) {
			if (edgesI.get(indexLow).containsKey(indexHigh)) {
				if (edgesI.get(indexLow).get(indexHigh)==null) {
					edgesI.get(indexLow).put(indexHigh, tri);
				} else {
					edgesI.get(indexLow).remove(indexHigh);
				}
			} else {
				edgesI.get(indexLow).put(indexHigh, tri);
			}
		} else {
			Map<Integer,TriangleN> highIndexHM;
    		highIndexHM = new HashMap<Integer,TriangleN>();
    		highIndexHM.put(indexHigh, tri);
    		edgesI.put(indexLow, highIndexHM);
		}
	}
	
	
	void buildMeshFromConnectedComponent(
			Map<Integer,Map<Integer,TriangleN>> edgesS,
			Map<Integer,Map<Integer,TriangleN>> edgesI,
			LinkedList<DotN> component) {
		if (component.size()==2) {return;}
		ArrayList<TriangleN> comp_triangles = new ArrayList<TriangleN>();
		boolean goUp;
		boolean triangleValidated;
		boolean somethingDoneInTheLoop;
			
		do {
			Iterator<DotN> it = component.iterator();
			goUp=false;
			somethingDoneInTheLoop=false;
			while ((it.hasNext())&&(!goUp)) {
				// use break statement
				// 1 - let's find all neighbors
				DotN dn = it.next();
				Integer in = dn.dotIndex;
				ArrayList<Integer> neighborsIndex = new ArrayList<>();
				if (edgesS.containsKey(in)) {
					neighborsIndex.addAll(edgesS.get(in).keySet());
				}
				if (edgesI.containsKey(in)) {
					neighborsIndex.addAll(edgesI.get(in).keySet());
				}
				if (neighborsIndex.size()==0) {
					component.remove(dn);
					goUp=true;
					somethingDoneInTheLoop=true;
				}
				Integer i1=0;
				int it1=0;
				Integer i2=0;
				int it2=0;
				DotN d1 = null;
				DotN d2 = null;
				while((it1<neighborsIndex.size()-1)&&(!goUp)) {
					it2=it1+1;
					i1=neighborsIndex.get(it1);
					d1 = completeListOfDots.get(i1);
					while((it2<neighborsIndex.size())&&(!goUp)) {
						i2=neighborsIndex.get(it2);
						d2 = completeListOfDots.get(i2);
						// Is d1 / d2 / dn a valid triangle ?
						TriangleN tri = new TriangleN(d1.dotIndex,d2.dotIndex,dn.dotIndex);
						if (i1>in) {
							triangleValidated = testEdges(d1,dn,edgesS.get(i1).get(in),d2);
						} else {
							triangleValidated = testEdges(d1,dn,edgesS.get(in).get(i1),d2);
						}
						if (i2>in) {
							triangleValidated = triangleValidated && testEdges(d2,dn,edgesS.get(i2).get(in),d1);
						} else {
							triangleValidated = triangleValidated && testEdges(d2,dn,edgesS.get(in).get(i2),d1);
						}
						if (triangleValidated) {
							somethingDoneInTheLoop=true;
							comp_triangles.add(tri);
							registerTri(edgesS,edgesI,tri);
						}
						it2++;
						if (triangleValidated) {
							goUp=true;
							neighborsIndex.clear();
							if (edgesS.containsKey(in)) {
								neighborsIndex.addAll(edgesS.get(in).keySet());
							}
							if (edgesI.containsKey(in)) {
								neighborsIndex.addAll(edgesI.get(in).keySet());
							}
							if (neighborsIndex.size()==0) {
								component.remove(dn);
								goUp=true;
								somethingDoneInTheLoop=true;
							}					
						}
					}
					it1++;
				}
								
			}					
		} while (somethingDoneInTheLoop);
		if (component.size()>0) {			
			if (component.size()==3) {
				//  just do it...
				TriangleN tri = new TriangleN(component.get(0).dotIndex,component.get(1).dotIndex,component.get(2).dotIndex);
				comp_triangles.add(tri);
				registerTri(edgesS,edgesI,tri);
				
			}
		}
		triangles.addAll(comp_triangles);
	}
	
	public ArrayList<TriangleN> constructMesh(float r0) {
		// Labeling dots
		triangles = new ArrayList<>();
		
        // Initializes edges list
        Map<Integer,Map<Integer,TriangleN>> edgesSup = new HashMap<Integer,Map<Integer,TriangleN>>();
        
        // Building triangles
        //buildTriangles(edgesSup, dots_in,r0*1.5f);
        registerEdges(edgesSup, completeListOfDots,r0*1.5f,true);        
        buildTriangles(edgesSup);
        
        // Removes all unconnected edges
        edgesSup.values().forEach(m -> {
    		m.values().removeIf(tri -> (tri==null));
    	});   	     	
		edgesSup.values().removeIf(m -> (m.isEmpty()));
		
        // Building reverse edge list
        Map<Integer,Map<Integer,TriangleN>> edgesInf = getComplementaryEdges(edgesSup);
        
        // Separating dots based on their connectivity
        LinkedList<LinkedList<DotN>> connectedComponents = new LinkedList<LinkedList<DotN>>();
        HashSet<Integer> isConnected = new HashSet<>();
        Iterator<Integer> itSup = edgesSup.keySet().iterator();
        
        
        int nDiscarded = 0;
        while(itSup.hasNext()) {
        	Integer probedIndex = itSup.next();
        	if (!isConnected.contains(probedIndex)) {
        		LinkedList<DotN> currentConnectedComponent = new LinkedList<>();
        		currentConnectedComponent.add(completeListOfDots.get(probedIndex));
        		connectedComponents.add(findConnectedComponents(edgesSup,edgesInf,currentConnectedComponent,MAX_CONNECTED_COMPONENT_SIZE));
        		connectedComponents.getLast().forEach(dn -> isConnected.add(dn.dotIndex));
        		if (connectedComponents.getLast().size()>=MAX_CONNECTED_COMPONENT_SIZE) nDiscarded++;
        	}        	
        }
        
        // Removing too large components
        // Their edges
        int nEdges=0;
        finalNumberOfEdges=0;
        connectedComponents.stream().filter(listOfDot->(listOfDot.size()>=MAX_CONNECTED_COMPONENT_SIZE)).forEach(listDot -> {	
        		listDot.forEach(dn -> {
        			edgesSup.remove(dn.dotIndex);
        			edgesInf.remove(dn.dotIndex);
        			if (edgesSup.containsKey(dn.dotIndex)||edgesInf.containsKey(dn.dotIndex)) {
        				finalNumberOfEdges++;
        			}
        		});   	
        });
        
        // The components themselves
        connectedComponents.removeIf(listOfDot->(listOfDot.size()>=MAX_CONNECTED_COMPONENT_SIZE));       
        connectedComponents.stream().forEach(listOfDots -> {
        	buildMeshFromConnectedComponent(edgesSup,edgesInf,listOfDots);
        	
        });
        
        
        
        // Computing the number of remaining edges
        numberOfEdges=finalNumberOfEdges+edgesSup.values().stream().map(m->m.size()).reduce(0,(a,b) -> a+b);        
        
        triangles.parallelStream().forEach(triangle -> {
        	// it should be oriented correctly... 
        	// otherwise we reverse it by switching two indexes
        	DotN A = completeListOfDots.get(triangle.id1);
        	DotN B = completeListOfDots.get(triangle.id2);
        	DotN C = completeListOfDots.get(triangle.id3);
        	
        	Vector3D sumN = new Vector3D(A.Norm.x+B.Norm.x+C.Norm.x,
        								   A.Norm.y+B.Norm.y+C.Norm.y,
        								   A.Norm.z+B.Norm.z+C.Norm.z
        								  );
        	
        	Vector3D AB = new Vector3D(B.pos.x-A.pos.x,
        								 B.pos.y-A.pos.y,
        								 B.pos.z-A.pos.z);
        	
        	Vector3D AC = new Vector3D(C.pos.x-A.pos.x,
					 				     C.pos.y-A.pos.y,
					 				     C.pos.z-A.pos.z);
        	
        	if (Vector3D.prodScal(Vector3D.prodVect(AB,AC),sumN)<0) {
        		int temp = triangle.id2;
        		triangle.id2 = triangle.id1;
        		triangle.id1 = temp;
        	}        	
        	
        });
        return triangles;
	}
	
	public int numberOfEdges=-1;
	
	LinkedList<DotN> findConnectedComponents(
			Map<Integer,Map<Integer,TriangleN>> edgesS,
			Map<Integer,Map<Integer,TriangleN>> edgesI,
			LinkedList<DotN> connectedComponent, int maxSize) {
		if (connectedComponent.size()==maxSize) {
			return connectedComponent;
		}
		Integer indexLastElement = connectedComponent.getLast().dotIndex;
		Iterator<Integer> it;
		if (edgesS.get(indexLastElement)!=null) {
			it = edgesS.get(indexLastElement).keySet().iterator();
			while ((it.hasNext())&&(connectedComponent.size()<maxSize)) {				
				DotN dnTest = completeListOfDots.get(it.next());
				if (!connectedComponent.contains(dnTest)) {
					connectedComponent.add(dnTest);
					connectedComponent=findConnectedComponents(edgesS,edgesI,connectedComponent, maxSize);	
				}
			}
		}
		if (edgesI.get(indexLastElement)!=null) {
			it = edgesI.get(indexLastElement).keySet().iterator();
			while ((it.hasNext())&&(connectedComponent.size()<maxSize)) {				
				DotN dnTest = completeListOfDots.get(it.next());
				if (!connectedComponent.contains(dnTest)) {
					connectedComponent.add(dnTest);
					connectedComponent=findConnectedComponents(edgesS,edgesI,connectedComponent, maxSize);
				}
			}
		}
		return connectedComponent;		
	}
	
	
	public void buildTriangles(Map<Integer,Map<Integer,TriangleN>> edges) {
		int triSize;
    	do {
    		triSize = triangles.size();	
			boolean oneStepUp=false;
			boolean twoStepsUp=false;
			boolean goUp=false;
			// Go through all edges and try to find triangles. When an edge has two triangles, it has to be removed from edges
			Iterator<Integer> indexHighIt = edges.keySet().iterator();
			while ((indexHighIt.hasNext())&&(goUp==false)) {
				Integer indexHigh = indexHighIt.next();			
				oneStepUp=false;
				twoStepsUp=false;
				// ---------------- indexHigh
				// listing all triangles that may be constructed from this index
				// retrieve HashMap			
				Map<Integer,TriangleN> midIndexHM = edges.get(indexHigh);
				
				Set<Integer> neighBorsOfIndexHigh = midIndexHM.keySet();
				//System.out.println("indexH "+indexHigh+" has "+neighBorsOfIndexHigh.size()+" neighbors.");
				Iterator<Integer> indexMidIt = neighBorsOfIndexHigh.iterator();
				while ((twoStepsUp==false)&&(indexMidIt.hasNext())) {
					Integer indexMid = indexMidIt.next();	
					oneStepUp=false;
					twoStepsUp=false;
					// ------------------- indexMid
					//System.out.println("We have the edge H=\t "+indexHigh+"\t ; M= \t"+indexMid);
					// We have the edge indexHigh>indexMid, 
					Map<Integer,TriangleN> lowIndexHM = edges.get(indexMid);//.get(indexMid);
					if (lowIndexHM!=null) {
						Set<Integer> neighBorsOfIndexMid = lowIndexHM.keySet();
						Iterator<Integer> indexLowIt = neighBorsOfIndexMid.iterator();
						while ((oneStepUp==false)&&(indexLowIt.hasNext())) {
							Integer indexLow = indexLowIt.next();
							oneStepUp=false;
							twoStepsUp=false;
							// ------------------- indexLow
							// System.out.println("H="+indexHigh+"; M="+indexMid+"; L="+indexLow);
							if (neighBorsOfIndexHigh.contains(indexLow)) {
								//System.out.println("H="+indexHigh+"; M="+indexMid+"; L="+indexLow);
								// We may have a match
								// Let's find all the edges properties
								TriangleN HighMidTri = midIndexHM.get(indexMid);
								TriangleN MidLowTri = lowIndexHM.get(indexLow);
								TriangleN HighLowTri = midIndexHM.get(indexLow);
								DotN dHigh=completeListOfDots.get(indexHigh);
								DotN dMid=completeListOfDots.get(indexMid);
								DotN dLow=completeListOfDots.get(indexLow);
								TriangleN tri = new TriangleN(dHigh.dotIndex,dMid.dotIndex,dLow.dotIndex);
								boolean validated=true;
								
								if (HighMidTri!=null) {
									validated = testEdges(dHigh,dMid,HighMidTri,dLow);
								}
								if ((validated)&&(MidLowTri!=null))  {
									validated=validated&&testEdges(dMid,dLow,MidLowTri,dHigh);
								}
								if ((validated)&&(HighLowTri!=null))  {
									validated=validated&&testEdges(dHigh,dLow,HighLowTri,dMid);
								}
								oneStepUp=false;
								twoStepsUp=false;
								if (validated) {
									triangles.add(tri);
									if (HighMidTri==null) {
										midIndexHM.put(indexMid, tri);
									} else {
										midIndexHM.remove(indexMid);
										// If I remove this then I need to go two steps up in these nested loops
										twoStepsUp=true;
										oneStepUp=true;
									}
									if (MidLowTri==null) {
										lowIndexHM.put(indexLow, tri);
									} else {
										lowIndexHM.remove(indexLow);
										// if I remove this then I need to go one step up in these nested loops
										oneStepUp=true;
									}
									if (HighLowTri==null) {
										midIndexHM.put(indexLow, tri);
									} else {
										midIndexHM.remove(indexLow);
										// if I remove this then I need to go two steps up in these nested loops
										twoStepsUp=true;
										oneStepUp=true;
									}
									if (triangles.size()%100000==0) {									
										System.out.println(triangles.size()+" triangles.");
									}
								}							
							}	
						}
					}
				}
			}
			edges.values().removeIf(m -> m.isEmpty());
    	} while(triangles.size()!=triSize);
	}	
	
	public boolean testEdges(DotN dEdge1, DotN dEdge2, TriangleN tri, DotN dToTest) {
		DotN dTri=null; // this needs to be the third dotn of the triangle
		// brute force
		if (tri.id1==dEdge1.dotIndex) {
			//dEdge1 is d1 -> dtri is not dEdge1
			if (tri.id2==dEdge2.dotIndex) {
				//dEdge2 is d2 -> dtri is d3
				dTri = completeListOfDots.get(tri.id3);
			} else {
				//dEdge2 is necesserily d3 thus dTri is d2 
				dTri = completeListOfDots.get(tri.id2);//tri.d2;
			}
		} else {
			//d1 is not dEdge1
			if (tri.id1==dEdge2.dotIndex) {
				//but it is dEdge2 thus dTri is not d1
				//d1  = dEdge2
				if (tri.id2==dEdge1.dotIndex) {
					//and d2 is dEdge1 thus
					dTri=completeListOfDots.get(tri.id3);
				} else {
					// d1 is dEdge2 and d2 is not dEdge1 thus
					dTri=completeListOfDots.get(tri.id2);
				}
			} else {
				// d1 is not dEdge1 and not dEdge2 thus
				dTri=completeListOfDots.get(tri.id1);				
			}
		}
		Vector3D vEdge = Vector3D.VecteurDir(dEdge1.pos, dEdge2.pos);
		Vector3D perpendDir = Vector3D.prodVect(vEdge, dEdge1.Norm);
		float pScalRef = Vector3D.prodScal(Vector3D.VecteurDir(dTri.pos, dEdge1.pos),perpendDir);
		float pScalTest = Vector3D.prodScal(Vector3D.VecteurDir(dToTest.pos, dEdge1.pos),perpendDir);
		// Code : pScalRef*pScalTest<0 -> 0 = success, you can!
	    return pScalRef*pScalTest<0;
		//return true;
	}	
}
