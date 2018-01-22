package eu.kiaru.limeseg.opt;

import ij.ImagePlus;
import java.util.ArrayList;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import eu.kiaru.limeseg.LimeSeg;
import eu.kiaru.limeseg.cudaHelper.CUDAUtils;
import eu.kiaru.limeseg.struct.CellT;
import eu.kiaru.limeseg.struct.DotN;
import eu.kiaru.limeseg.struct.Vector3D;
import jcuda.*;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import static jcuda.driver.JCudaDriver.*;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealRandomAccessible;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.interpolation.randomaccess.NearestNeighborInterpolatorFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;


/**
 * Heart of LimeSeg - CPU side
 * @author Nicolas Chiaruttini
 */
public class Optimizer {
	
    LimeSeg lms;   // associated limeseg for GUI I/O and interaction
    
    // Pieces of membrane = surfel : position + normal to this position (list)
    // CPU
    public ArrayList<DotN> dots;
    public ArrayList<CellT> cellTInOptimizer = new ArrayList<>();
    // GPU
    GPUDots gpuDots;
    
    /**
     * Keys to the list of parameters that can be set to customize the optimizer
     */
    public final static String[] paramList = {"ka","pa","pr","kr",
    		"k_grad","k_bend", "k_align","d_0",
    		"fillHoles","rmOutliers","attractToMax",
    		"radiusTresholdInteract",
    		"NStepPerR0","maxDisplacementPerStep",
    		"ageMinGenerate","rmIfNeighborBelow",
    		"rmIfNeighborAbove","generateDotIfNeighborEquals",
    		"radiusSearch","normalForce","radiusRes",
    		"radiusDelta","searchMode",
    		"convergenceTimestepSampling",
    		"convergenceDistTreshold",
    		"convergenceNormTreshold",
    		"radiusRelaxed", "ZScale", 
    		"MinX", "MaxX", "MinY", "MaxY", "MinZ", "MaxZ"};
    
    /* Optimizer parameters, private variables 
     * They should be accessed via setParam and getParam */
    public float d_0=2f;       // Equilibrium distance (in pixel) between surfels 
    float k_align=0.05f;
    float k_bend=0.1f;
    float k_grad=0.03f;            // Force towards max intensity (gradient
    boolean attracted_to_Maximum=true;
    boolean rmOutliers=true;
    boolean fHoles=true;
    float ka=0.01f;
    float pa=5;    
    float kr; // not independent : set by d0, pr, pa, ka
    float pr=9;
    float radiusTresholdInteract=1.75f;
    float NStepPerR0=5000f;
    float maxDisplacementPerStep=0.3f;   
    float ageMinGenerate=10;//normally 10
    float normalForce=0;    
    int rmIfNeighborBelow=5;
    int rmIfNeighborAbove=11;
    int generateDotIfNeighborEquals=6;   
    public float limitInteractAttract;
    int convergenceTimestepSampling=20;
    float convergenceDistTresholdSquared=0.1f*0.1f; // in d_0^2 units
    float convergenceNormTresholdSquared=0.1f*0.1f;
    float radiusRelaxed=1f;    
    float radiusRes=0.5f; // increment in pixel
    float radiusSearch=5f;
    float radiusDelta=0f; // delta with the maximum intensity
    long MinX,MinY,MinZ;
    long MaxX,MaxY,MaxZ;
    float ZScale=1f;
    
    /**
     * Number of iterations performed by the Optimizer
     */
    public int NIterations;
    
    // For tree representation of the dots
    // CPU
    public ArrayList<BlockOfDots> allBlocks; // made public for cell tesselation
    ArrayList<BlockOfDots> ignoredBlocks;
    ArrayList<BlockOfDots> workingBlocks;
    // GPU
    GPUBlockList gpuAllBlocks;
    GPUBlockList gpuIgnoredBlocks;
    WGPUBlockList gpuWorkingBlocks_in;
    WGPUBlockList gpuWorkingBlocks_out;
    
    long[] GPUBlockTimeReport;    
    
    // Variable for superdots
    /*if a block contains dots which have all converged, 
     * and the number of dots is higher than this value, then LimeSeg will try to make a super dot 
     */
    int thresholdSuperDotCreationWorthTest=300; 
    /*
     * If a potential superdots contains more than this value of superdots
     * Then the super dot is created
     */ 
    int thresholdSuperDotCreationWorthDo=100;
    /*
     * Tries to find superdots only 1 timestep every numberOfIterationBetweenEachSuperdotTest steps
     */
    int numberOfIterationBetweenEachSuperdotTest = 50;
    boolean writeTimeInfos=false;//true;
    boolean requestFlattenSuperDots;
    
    
    public boolean requestResetDotsConvergence=false;
    /*
     * For 3D Viewer synchronization
     */
    public boolean requestFillBufferRenderer=false;
    
    // Avoids memory allocation in computationally intensive part of the program
    float[] fRep,fAtt;    
    Vector3D dir, SumN;
    
    public String lastImgLoaded="-";
    public int lastFrameLoaded=0;
    
    //----------------- CUDA Variables
    public boolean CUDAEnabled=false;
    static int MAX_THREADS_PER_BLOCK;    
    static int CUDA_DEVICE_ID=0;
    int CUDATreshold = 24000/10;
    
    static CUfunction fComputePairsOfDot;
    static CUfunction fAVGBlocks;
    static CUfunction fKDTree_FindFurthestBlocks;
    static CUfunction fCalcDir;
    static CUfunction fMapDots;
    static CUfunction fSortBlocks;
    static CUfunction fDispatchDots;
    
    /**
     * 3D image LimeSeg works on
     */
    public RealRandomAccessible image3DInfinite;
    
    public Optimizer() {
        allBlocks = new ArrayList<>();
        dots = new ArrayList<>();        
        
        // Avoids memory allocation in computationally intensive part of the program
        dir = new Vector3D(0.0f, 0.0f, 0.0f);
        SumN = new Vector3D(0.0f, 0.0f, 0.0f);
        this.reComputeParameters();        
        this.setCUDAContext();
            // Only needed to recompute kernels
            /*try {
                CUDAUtils.preparePtxFile("src/MbLipidSeg/computeAVGBlocks.cu");
                CUDAUtils.preparePtxFile("src/MbLipidSeg/computeKDTree_FindFurthest.cu");
                CUDAUtils.preparePtxFile("src/MbLipidSeg/gpuCalcDir.cu");
                CUDAUtils.preparePtxFile("src/MbLipidSeg/gpuMapDots.cu");
                CUDAUtils.preparePtxFile("src/MbLipidSeg/gpuSortBlocks.cu");
                CUDAUtils.preparePtxFile("src/MbLipidSeg/gpuDispatchDots.cu");
            } catch (IOException ex) {
                Logger.getLogger(Optimizer.class.getName()).log(Level.SEVERE, null, ex);
            }*/
    }
    
    public Optimizer(LimeSeg lms_in){
    	this();
    	lms=lms_in;
    }
    
    public < T extends RealType< T > & NativeType< T >> void setWorkingImage(RandomAccessibleInterval<T> img, int channel, int frame) {
    	System.out.println("Il y a "+img.numDimensions()+" dimensions dans l'image.");
    	if (img.numDimensions()<3) {
            //IJ.error("The image has less than 3 dimensions");
            lastImgLoaded="Undefined";
            lastFrameLoaded=-1;
            return;
        } else if (img.numDimensions()==4) {
            // Is the fourth dimension time or channel ?...
            //if (imp.getNChannels()==1) {
                // We assume it is frame
                img = Views.hyperSlice(img, 3, frame-1 ); // IJ1 numerotation 
            //} else {
                // we assume it is channel
            //   img = Views.hyperSlice(img, 2, channel-1 ); // IJ1 numerotation 
            //}
        } else if (img.numDimensions()==5) {                
            img = Views.hyperSlice(img, 4, frame-1 );   //IJ1 numerotation 
            img = Views.hyperSlice(img, 2, channel-1 ); //IJ1 numerotation 
        } else if (img.numDimensions()>5) {
            //IJ.error("This plugin do not handle more than 5D images.");
            lastImgLoaded="Undefined";
            lastFrameLoaded=-1;
            return;
        }
        image3DInfinite = Views.interpolate(Views.extendMirrorSingle( img ), new NearestNeighborInterpolatorFactory() );
        MinX=img.min(0);MaxX=img.max(0);
        MinY=img.min(1);MaxY=img.max(1);
        MinZ=img.min(2);MaxZ=img.max(2);     
        System.out.println("MaxZ="+MaxZ);
        lastImgLoaded="Ch="+channel+"; Fr="+frame+"; Img="+img.toString();//.getTitle();
        lastFrameLoaded=frame;
    }
    
    public < T extends RealType< T > & NativeType< T >> void setWorkingImage(ImagePlus imp, int channel, int frame) {
    	if (imp!=null) {            
            // Here we assume 0 = X; 1 = Y; 2 = Z; 3 = Frames; 4 = Channels
            System.out.println("Channel="+channel);
            System.out.println("Frame="+frame);
            RandomAccessibleInterval<T> img = ImageJFunctions.wrap( imp );
            if (img.numDimensions()<3) {
                //IJ.error("The image has less than 3 dimensions");
                lastImgLoaded="Undefined";
                lastFrameLoaded=-1;
                return;
            } else if (img.numDimensions()==4) {
                // Is the fourth dimension time or channel ?...
                if (imp.getNChannels()==1) {
                    // We assume it is frame
                    img = Views.hyperSlice(img, 3, frame-1 ); // IJ1 numerotation 
                } else {
                    // we assume it is channel
                    img = Views.hyperSlice(img, 2, channel-1 ); // IJ1 numerotation 
                }
            } else if (img.numDimensions()==5) {                
                img = Views.hyperSlice(img, 4, frame-1 ); //IJ1 numerotation 
                img = Views.hyperSlice(img, 2, channel-1 ); //IJ1 numerotation 
            } else if (img.numDimensions()>5) {
                //IJ.error("This plugin do not handle more than 5D images.");
                lastImgLoaded="Undefined";
                lastFrameLoaded=-1;
                return;
            }
            //NearestNeighborInterpolatorFactory< T > factory1 = ;
            image3DInfinite = Views.interpolate(Views.extendMirrorSingle( img ), new NearestNeighborInterpolatorFactory<>() );
            MinX=img.min(0);MaxX=img.max(0);
            MinY=img.min(1);MaxY=img.max(1);
            MinZ=img.min(2);MaxZ=img.max(2);  
            lastImgLoaded="Ch="+channel+"; Fr="+frame+"; Img="+imp.getTitle();
            lastFrameLoaded=frame;
        } else {            
            lastImgLoaded="Undefined";
            lastFrameLoaded=-1;
        }
    }
    
    public void filterOptDots(Predicate<DotN> test) {
    	dots.removeIf(test);
    	for (CellT ct: cellTInOptimizer) {
            ct.dots.removeIf(test);
        } 
    }
    
    public void setOptParam(String paramName, float value) {
        switch (paramName) {
            case "ka":ka=value;reComputeParameters();break;
            case "pa":pa=value;reComputeParameters();break;
            case "pr":pr=value;reComputeParameters();break;
            case "k_grad":k_grad=value;break;
            case "k_bend":k_bend=value;break;
            case "k_align":k_align=value;break;
            case "d_0":if (value!=d_0) {
                            d_0=value;
                            reComputeParameters();
                        }break;
            case "fillHoles":fHoles=(value==1);break;
            case "rmOutliers":rmOutliers=(value==1);break;
            case "attractToMax":attracted_to_Maximum=(value==1);break;
            case "radiusTresholdInteract":radiusTresholdInteract=value;reComputeParameters();break;
            case "NStepPerR0":NStepPerR0=value;break;
            case "maxDisplacementPerStep":maxDisplacementPerStep=value;break;
            case "ageMinGenerate":ageMinGenerate=value;break;
            case "rmIfNeighborBelow":rmIfNeighborBelow=(int) value;break;
            case "rmIfNeighborAbove":rmIfNeighborAbove=(int) value;break;
            case "generateDotIfNeighborEquals":generateDotIfNeighborEquals=(int) value;break;
            case "radiusSearch":radiusSearch=value;break;
            case "radiusRes":radiusRes=value;break;
            case "radiusDelta":radiusDelta=value;break;
            case "normalForce":normalForce=value;break;
            case "convergenceTimestepSampling":convergenceTimestepSampling=(int)value;break;
            case "convergenceDistTreshold":convergenceDistTresholdSquared=value*value;break;
            case "convergenceNormTreshold":convergenceNormTresholdSquared=value*value;break;
            case "radiusRelaxed":radiusRelaxed=value;break;
            case "ZScale":ZScale=value;break;
            case "MinX":MinX=(long)value;break;
            case "MaxX":MaxX=(long)value;break;
            case "MinY":MinY=(long)value;break;
            case "MaxY":MaxY=(long)value;break;
            case "MinZ":MinZ=(long)value;break;
            case "MaxZ":MaxZ=(long)value;break;
            default:
                System.out.println("on set "+paramName+" at value ="+value+" but this parameter is not recognized!");
            break;
        }
    }
    
    void resetDotsConvergence() {
        this.flattenSuperDots();
        dots.parallelStream().forEach(nd ->  
        { 
            if ((nd.userDestroyable)||(nd.userGenerate)||(nd.userMovable)||(nd.userRotatable))
            {
            	nd.allNeighborsHaveConverged=false;
            	nd.hasConverged=false;
            }
        });
    }
    
    public double getOptParam(String paramName) {
        float value;
        switch (paramName) {
            case "ka":value=ka;break;
            case "pa":value=pa;break;
            case "pr":value=pr;break;
            case "kr":value=kr;break;
            case "k_grad":value=k_grad;break;
            case "k_bend":value=k_bend;break;
            case "k_align":value=k_align;break;
            case "d_0":value=d_0;break;
            case "fillHoles":if (fHoles) {value=1;} else {value=0;}break;
            case "rmOutliers":if (rmOutliers) {value=1;} else {value=0;}break;
            case "attractToMax":if (attracted_to_Maximum) {value=1;} else {value=0;}break;
            case "radiusTresholdInteract":value=radiusTresholdInteract;break;
            case "NStepPerR0":value=NStepPerR0;break;
            case "maxDisplacementPerStep":value=maxDisplacementPerStep;break;
            case "ageMinGenerate":value=ageMinGenerate;break;
            case "rmIfNeighborBelow":value=rmIfNeighborBelow;break;
            case "rmIfNeighborAbove":value=rmIfNeighborAbove;break;
            case "generateDotIfNeighborEquals":value=generateDotIfNeighborEquals;break;
            case "radiusSearch":value=radiusSearch;break;
            case "normalForce":value=normalForce;break;
            case "radiusRes":value=radiusRes;break;
            case "radiusDelta":value=radiusDelta;break;
            case "convergenceTimestepSampling":value=convergenceTimestepSampling;break;
            case "convergenceDistTreshold":value=(float)java.lang.Math.sqrt(convergenceDistTresholdSquared);break;
            case "convergenceNormTreshold":value=(float)java.lang.Math.sqrt(convergenceNormTresholdSquared);break;
            case "radiusRelaxed":value=radiusRelaxed;break;
            case "ZScale":value=ZScale;break;
            case "MinX":value=MinX;break;
            case "MaxX":value=MaxX;break;
            case "MinY":value=MinY;break;
            case "MaxY":value=MaxY;break;
            case "MinZ":value=MinZ;break;
            case "MaxZ":value=MaxZ;break;
            default:value=Float.NaN;break;
        }
        return (double) (value);
    }
    
    private void reComputeParameters() {         
        kr=(float) (pa*ka/pr*java.lang.Math.pow(1,pr-pa));
        this.computeForcesAttractRepul();        
        this.limitInteractAttract=radiusTresholdInteract*radiusTresholdInteract*d_0*d_0;
    }
        
    void computeForcesAttractRepul() {
        int NPt=(int) ((radiusTresholdInteract+1)*(float)(NStepPerR0)); // weird!
        fRep = new float[NPt];
        fAtt = new float[NPt];
        fAtt[0]=0;
        int iTr=-1;
        for (int i=NPt-1;i>0;i--) {
            fAtt[i]=(float) (pa*ka/java.lang.Math.pow((float)(i)/(float)(NStepPerR0), pa+1));           
            fRep[i]=(float) (-pr*kr/java.lang.Math.pow((float)(i)/(float)(NStepPerR0), pr+1));
            if (fAtt[i]+fRep[i]<0) {
                if (iTr==-1) {iTr=i;}
                // Smooth repulsion
                fAtt[i]=fAtt[iTr]*0+(float) (pr*kr/java.lang.Math.pow(1f, pr+1));
                fRep[i]=-fAtt[iTr]*0+(float) (-pr*kr/java.lang.Math.pow(1f, pr+1))-(1-(float)(i)/((float)(NStepPerR0)))*maxDisplacementPerStep;
            }
        }
    }

    void computeForcesperDot() {      
         dots.parallelStream().forEach(nd -> {
            if (!nd.allNeighborsHaveConverged) {
                nd.reInit(); // set Force to Zero
                if ((attracted_to_Maximum)&&(nd.age>ageMinGenerate)){
                    nd.computeGradForce_Max(image3DInfinite.realRandomAccess(),	k_grad, 
                    		radiusRelaxed, radiusRes, radiusDelta, radiusSearch, 
                    		ZScale, MinX, MaxX, MinY, MaxY, MinZ, MaxZ);
                } 
            }
        });
    }
    
    void computeForcesPair(DotN nd1, DotN nd2){
        if ((!nd1.isSuperDot)&&(!nd2.isSuperDot)) {
            float dx, dy, dz, norme2;
            dx=nd2.pos.x-nd1.pos.x;
            dy=nd2.pos.y-nd1.pos.y;
            dz=nd2.pos.z-nd1.pos.z;
            norme2=(dx*dx+dy*dy+dz*dz);        
            if (norme2<limitInteractAttract) { 
                float NF, fx, fy, fz;
                float f_attract, f_rep;
                int POS; 
                float norme;
                nd1.allNeighborsHaveConverged=nd1.allNeighborsHaveConverged&&nd2.hasConverged; 
                nd2.allNeighborsHaveConverged=nd2.allNeighborsHaveConverged&&nd1.hasConverged; 
                boolean sameCell=nd1.ct.idInt==nd2.ct.idInt;
                dir.x=dx;dir.y=dy;dir.z=dz;
                norme=(float)(java.lang.Math.sqrt(norme2));
                dir.x/=norme;dir.y/=norme;dir.z/=norme;
                POS=(int)(norme/d_0*(float)(NStepPerR0));
                f_rep=fRep[POS];                           
                fx=f_rep*dir.x;fy=f_rep*dir.y;fz=f_rep*dir.z;
                if (sameCell) {               
                    f_attract=fAtt[POS];
                    nd1.repForce.x+=fx;nd1.repForce.y+=fy;nd1.repForce.z+=fz;
                    nd2.repForce.x-=fx;nd2.repForce.y-=fy;nd2.repForce.z-=fz;
                    NF=(f_attract+f_rep);
                    fx=NF*dir.x;fy=NF*dir.y;fz=NF*dir.z;

                    nd1.N_Neighbor++;
                    nd2.N_Neighbor++;
                    // They are neighbor : we should align them compute their force
                    // Computation of a force that tend to make a U shape between  Norm1/dir/Norm2 (moment / k_align)
                    // So a force along Normal to flatten it proportional to the prodscal
                    
                    SumN.x=nd1.Norm.x+nd2.Norm.x;SumN.y=nd1.Norm.y+nd2.Norm.y;SumN.z=nd1.Norm.z+nd2.Norm.z;                    
                    float iFlatten=k_align*Vector3D.prodScal(SumN, dir);
                    
                    nd1.force.x+=nd1.Norm.x*iFlatten;
                    nd1.force.y+=nd1.Norm.y*iFlatten;
                    nd1.force.z+=nd1.Norm.z*iFlatten;
                    
                    nd2.force.x-=nd2.Norm.x*iFlatten;
                    nd2.force.y-=nd2.Norm.y*iFlatten;
                    nd2.force.z-=nd2.Norm.z*iFlatten;
                    // We compute the perpendicular with the vector dir
                    float iPerpend1=-k_bend*Vector3D.prodScal(nd1.Norm,dir);
                    nd1.moment.x+=iPerpend1*dir.x;                    
                    nd1.moment.y+=iPerpend1*dir.y;                    
                    nd1.moment.z+=iPerpend1*dir.z;
                    float iPerpend2=-k_bend*Vector3D.prodScal(nd2.Norm,dir); 
                    nd2.moment.x+=iPerpend2*dir.x;                    
                    nd2.moment.y+=iPerpend2*dir.y;                    
                    nd2.moment.z+=iPerpend2*dir.z;                
                } else {
                    nd1.relaxed=1f;
                    nd2.relaxed=1f;
                }
                nd1.force.x+=fx;nd1.force.y+=fy;nd1.force.z+=fz;
                nd2.force.x-=fx;nd2.force.y-=fy;nd2.force.z-=fz;
            }
        } else {
            // One point at least is a superDot
            if ((nd1.isSuperDot)&&(nd2.isSuperDot)) {                
                // at least one should not be a SuperDot, because they have always converged
            } else {
                DotN sDot,nDot;
                if (nd1.isSuperDot) {
                    sDot=nd1;nDot=nd2;
                } else {
                    sDot=nd2;nDot=nd1;                   
                }
                float dx, dy, dz, norme2;
                dx=nd2.pos.x-nd1.pos.x;
                dy=nd2.pos.y-nd1.pos.y;
                dz=nd2.pos.z-nd1.pos.z;
                norme2=dx*dx+dy*dy+dz*dz-sDot.superDotRadiusSquared;
                if ((norme2<0)&&(nDot.allNeighborsHaveConvergedPreviously==false)) {
                    sDot.allNeighborsHaveConverged=false;
                }                
            }
        }
    }
    
    public ArrayList<BlockOfDots> makeFirstBlocks() {        
        ArrayList<BlockOfDots> iniBlocks = new ArrayList<>();
        BlockOfDots firstBlock=new BlockOfDots(0,dots.size());
        firstBlock.dotsInBlock0=dots;
        // Gives index for GPU purposes
        for (int i=0;i<dots.size();i++) {
            DotN dn;
            dn = dots.get(i);
            dn.dotIndex=i;
        }
        iniBlocks.add(firstBlock);
        return iniBlocks;
    }
    
    void buildTreeCPU(ArrayList<BlockOfDots> iniBlocks, int minNumberOfInteractionsPerBlock) {
        float sqLimitInteractAttract = (float) java.lang.Math.sqrt(limitInteractAttract);        
        ignoredBlocks = new ArrayList<>();
        workingBlocks = iniBlocks;//makeFirstBlocks();
        while (workingBlocks.size()>0) {
              Map<Integer, List<BlockOfDots>> map = workingBlocks
                      .parallelStream()
                      .collect(Collectors.groupingBy(block -> block.splitBlock(sqLimitInteractAttract, minNumberOfInteractionsPerBlock)));
              
              workingBlocks.clear();
              if (map.get(BlockOfDots.BLOCK_SPLIT)!=null)
              map.get(BlockOfDots.BLOCK_SPLIT).forEach((block) -> {
                      workingBlocks.addAll(block.childrenBlocks);
              });
              if (map.get(BlockOfDots.BLOCK_KEEP)!=null)
              map.get(BlockOfDots.BLOCK_KEEP).forEach((block) -> {
                      allBlocks.addAll(block.childrenBlocks);
              });
              if (map.get(BlockOfDots.BLOCK_DISCARD)!=null)
              map.get(BlockOfDots.BLOCK_DISCARD).forEach((block) -> {
                      if ((block.numberOfDotsInBlock()> this.thresholdSuperDotCreationWorthTest)) {
                          ignoredBlocks.addAll(block.childrenBlocks);
                      }
              });
        }
    }
    
    void computeForcesperPairsOfDotCPU() {
        allBlocks.forEach((block) -> {            
            if (block.blockLevel==0) {
                int blSize = block.dotsInBlock0.size();
                for (int i=0;i<blSize-1;i++) {
                    DotN dn1 = block.dotsInBlock0.get(i);
                    for (int j=i+1;j<blSize;j++) {
                        DotN dn2 = block.dotsInBlock0.get(j);
                        computeForcesPair(dn1,dn2);
                    }
                }                
            } else {
                for (DotN dn1:block.dotsInBlock0) {
                    for (DotN dn2:block.dotsInBlock1) {
                        computeForcesPair(dn1,dn2);
                    }
                }
            }
        });
    }
    
    public void buildTreeForNeighborsSearch(ArrayList<BlockOfDots> iniBlocks, int blockSize) {
    	tic();
    	int targetGPUBlockSize=0;// = java.lang.Math.min(512, Optimizer.MAX_THREADS_PER_BLOCK);
        if ((CUDAEnabled)&&(dots.size()>CUDATreshold)) {
            targetGPUBlockSize = java.lang.Math.min(256, Optimizer.MAX_THREADS_PER_BLOCK);            
            this.setCUDAContext();
            buildTreeGPU(iniBlocks, blockSize);//Optimizer.MAX_THREADS_PER_BLOCK);
            allBlocks = gpuAllBlocks.pop(dots);
            //if (writeTimeInfos) System.out.print("BuildTreeGPU=\t"+(toc()/1000)+"\t");
        } else {
            buildTreeCPU(iniBlocks,blockSize);
            //if (writeTimeInfos) System.out.print("BuildTreeCPU=\t"+(toc()/1000)+"\t");
        }
        // the output is in allBlocks
    }

    public void nextStep() {
        if (requestFlattenSuperDots) {
            requestFlattenSuperDots=false;
            flattenSuperDots();
        }
        if (requestResetDotsConvergence) {
            requestResetDotsConvergence=false;
            this.resetDotsConvergence();
        }
        NIterations++;          
        if (writeTimeInfos) System.out.print("N=\t"+dots.size()+"\t Remaining=\t"+(toc()/1000)+"\t");
        if (writeTimeInfos) tic();

        dots.parallelStream().forEach(nd ->  
        {   
            if (!nd.allNeighborsHaveConverged) {
                nd.updatePosition(normalForce, d_0, maxDisplacementPerStep, MinX, MaxX, MinY, MaxY, MinZ, MaxZ, ZScale);
            }
            nd.checkConvergence(d_0, convergenceDistTresholdSquared, convergenceTimestepSampling, convergenceNormTresholdSquared);
            nd.allNeighborsHaveConvergedPreviously=nd.allNeighborsHaveConverged;
        }
        );        
        if (writeTimeInfos) System.out.print("UpdPos=\t"+(toc()/1000)+"\t");
        //reInitDots();
        if (writeTimeInfos) tic();
        int targetGPUBlockSize=0;// = java.lang.Math.min(512, Optimizer.MAX_THREADS_PER_BLOCK);
        if ((CUDAEnabled)&&(dots.size()>CUDATreshold)) {
            targetGPUBlockSize = java.lang.Math.min(256, Optimizer.MAX_THREADS_PER_BLOCK);
            buildTreeGPU(makeFirstBlocks(),targetGPUBlockSize);//Optimizer.MAX_THREADS_PER_BLOCK);
            ignoredBlocks = gpuIgnoredBlocks.pop(dots);
            if (writeTimeInfos) System.out.print("BuildTreeGPU=\t"+(toc()/1000)+"\t");
        } else {
            buildTreeCPU(makeFirstBlocks(), 200);
            if (writeTimeInfos) System.out.print("BuildTreeCPU=\t"+(toc()/1000)+"\t");
        }
        if (writeTimeInfos) tic();
        computeForcesperDot();
        if (writeTimeInfos) System.out.print("CompFPDot=\t"+(toc()/1000)+"\t");
        if (writeTimeInfos) tic();
        //System.out.print("NInteractComputed=\t"+NInteractToBeComputedFinal+"\t");
        //if ((NInteractToBeComputedFinal>CUDATresholdComputeForces)&&(CUDAEnabled)) {
        if ((CUDAEnabled)&&(dots.size()>CUDATreshold)) {
            computeForcesperPairsOfDotGPU(targetGPUBlockSize);
            this.resetGPUFlags();
            if (writeTimeInfos) System.out.print("CompFPPairsDotGPU=\t"+(toc()/1000)+"\t");
        } else {
            computeForcesperPairsOfDotCPU();
            if (writeTimeInfos) System.out.print("CompFPPairsDotCPU=\t"+(toc()/1000)+"\t");
        }
        allBlocks.clear();
        if (NIterations%numberOfIterationBetweenEachSuperdotTest==0) {
            if (writeTimeInfos) tic(); 
            for (BlockOfDots block:ignoredBlocks) {
                createsASuperDot(block);
            }
            if (writeTimeInfos) System.out.print("CreateSDot=\t"+(toc()/1000)+"\t");
        }
        if (writeTimeInfos) tic();
        if (rmOutliers) removeOutliers();
        if (fHoles) fillHoles(); 
        if (writeTimeInfos) System.out.print("FillRemove=\t"+(toc()/1000)+"\t");
        if (writeTimeInfos) tic();
        removeSuperDots();
        if (writeTimeInfos) System.out.print("RM_SDots=\t"+(toc()/1000)+"\t");        
        if (writeTimeInfos) tic();
        if (writeTimeInfos) System.out.print("activedots=\t"+((int)((1-getRatioOfNonActiveDots())*dots.size()))+"\n");
        if ((requestFillBufferRenderer)&&(lms!=null)) {
        	if (lms.jcr.getViewMode()>=8) {
                lms.jcr.fillBufferCellRenderer_PC(dots);
            } else {
                lms.fillBufferCellRenderer();
            }
            requestFillBufferRenderer=false;
        }
    }
    
    void removeSuperDots() {
        // needs to remove all interacting superdots
        ArrayList<DotN> toBePutInDots = null;
        for (int i=0;i<dots.size();i++) {
            DotN dn = dots.get(i);
            if ((dn.isSuperDot)&&(dn.allNeighborsHaveConverged==false)) {
                if (toBePutInDots == null) {
                    toBePutInDots = new ArrayList<>(dn.dotsContained.size());
                } 
                dots.remove(i);
                toBePutInDots.addAll(dn.dotsContained);  
                i--; //removes 1 because of the super dot that's being removed
            }
        }
        if (toBePutInDots!=null) {
            dots.addAll(toBePutInDots);
        }
    }

    void createsASuperDot(BlockOfDots block) {
        //System.out.println("NBlocksToDo = "+countNumberOfBlocksWorthTryingSuperDot);
        // needs to create a single superdot if worth it        
        float sqLimitInteractAttract = (float) java.lang.Math.sqrt(limitInteractAttract);
        // It is worth trying
        // First we'll look for the surfel which has all neighbord converged = false the nearest
        if (!block.centerComputed) {block.computeCenterAndConvergence();}
        float minDistNotConvergedToCenter = Float.MAX_VALUE;
        for (DotN dn : dots) {
            if (dn.allNeighborsHaveConvergedPreviously==false) {
                float d = Vector3D.dist(dn.pos, block.center);
                if (d<minDistNotConvergedToCenter) {
                    minDistNotConvergedToCenter=d;
                }
            }                
        }
        //System.out.println("minDistNotConvergedToCenter="+minDistNotConvergedToCenter);
        // we got the min dist. So now we need to collect all the points that are in the good range
        ArrayList<DotN> dotsInSuperDot = new ArrayList<>(thresholdSuperDotCreationWorthTest);
        //System.out.println("Le min dist vaut \t "+minDistNotConvergedToCenter+" \t dots");
        float dMargin;
        float count=0;
        for (int i=0;i<dots.size();i++) {
            DotN dn = dots.get(i);
            float d = Vector3D.dist(dn.pos, block.center);
            if (dn.isSuperDot) {
                dMargin=sqLimitInteractAttract+dn.superDotRadius;
            } else {
                dMargin=sqLimitInteractAttract;
            }
            if (d<(minDistNotConvergedToCenter-dMargin)) {
                dotsInSuperDot.add(dn);
                dots.remove(i);
                i--;
                count++;
            }
        }
        if (count<thresholdSuperDotCreationWorthDo) {
            // Useless
            //System.out.println("pas assez de points");
            dots.addAll(dotsInSuperDot);
        } else {
            //System.out.println("ON devrait pouvoir faire un super dot avec \t "+dotsInSuperDot.size()+" \t dots");
            Vector3D p = new Vector3D(0,0,0);
            Vector3D n = new Vector3D(0,0,0);
            p.x=block.center.x;
            p.y=block.center.y;
            p.z=block.center.z;
            DotN sDot = new DotN(p,n);
            sDot.hasConverged=true;
            sDot.allNeighborsHaveConverged=true;
            sDot.allNeighborsHaveConvergedPreviously=true;
            sDot.isSuperDot=true;
            sDot.superDotRadius=minDistNotConvergedToCenter-sqLimitInteractAttract+this.maxDisplacementPerStep*d_0;
            sDot.superDotRadiusSquared=sDot.superDotRadius*sDot.superDotRadius;
            sDot.dotsContained=dotsInSuperDot;
            dots.add(sDot);
        }
    }
    
    public double getRatioOfDotsConverged() {
        // returns the number of dots in the optimizer that are relaxed, in percentage
        double ans=0;
        for (DotN nd : dots) {
            if (nd.hasConverged){
                ans+=1;                
            }
        }
        return ans/dots.size();        
    }
    
    double getRatioOfNonActiveDots() {
        // returns the number of dots in the optimizer that are relaxed, in percentage
        double ans=0;
        for (DotN nd : dots) {
            if (nd.allNeighborsHaveConverged){
                ans+=1;                
            }
        }
        return ans/dots.size();        
    }

    void removeOutliers() {
        // Collect CellT objects in optimizer
        // removeIf Much faster on an ArrayList
        
        Predicate<DotN> dotNPredicate = nd -> ((!nd.isSuperDot)&&(nd.isOptimized))&&(nd.userDestroyable)&&((nd.N_Neighbor<rmIfNeighborBelow)||(nd.N_Neighbor>rmIfNeighborAbove));
        
        
        dots.removeIf(dotNPredicate);
        
        for (CellT ct: cellTInOptimizer) {
            ct.dots.removeIf(dotNPredicate);
        }    
    }
    
    void flattenSuperDots() {
        // removes superdots 
        Predicate<DotN> isSuperDot = dn -> dn.isSuperDot;
        while (dots.stream().anyMatch(isSuperDot)) {
            ArrayList<DotN> toBePutInDots = null;
            for (int i=0;i<dots.size();i++) {
                DotN dn = dots.get(i);
                if (dn.isSuperDot) {
                    if (toBePutInDots == null) {
                        toBePutInDots = new ArrayList<>(dn.dotsContained.size());
                    } 
                    dots.remove(i);
                    toBePutInDots.addAll(dn.dotsContained);  
                    i--;// removes 1 because of the super dot that's being removed
                }
            }
            if (toBePutInDots!=null) {
                dots.addAll(toBePutInDots);
            }
        }        
    }
        
    public void fillHoles() {
        for (int i=0;i<dots.size();i++){
            DotN nd = dots.get(i);
            if ((nd.N_Neighbor==generateDotIfNeighborEquals)&&(nd.age>ageMinGenerate)&&(nd.isOptimized)&&(nd.userGenerate)) {                
                dots.add(nd.generate(d_0));
            }
        }
    }
    
    public void addDots(CellT ct) {
        if (ct!=null) {
            for (int i=0;i<ct.dots.size();i++) {
                this.dots.add(ct.dots.get(i));
            }
            this.cellTInOptimizer.add(ct);
        }
    }
    
    public void removeDots(CellT ct) {
        if (ct!=null) {
            for (int i=0;i<ct.dots.size();i++) {
                this.dots.remove(ct.dots.get(i));
            }
            this.cellTInOptimizer.remove(ct);
        }       
    }
    
    public void removeAllDots() {
        this.dots.clear();
        this.dots=new ArrayList<>();
        this.cellTInOptimizer.clear();
    }
    
    public void tic() {
        startTime = System.nanoTime();
    }
    
    long startTime,endTime,duration; 
    
    public long toc(String message) {
        endTime = System.nanoTime();
        duration = (endTime - startTime);  
        System.out.println(message+":\t"+(duration/1000)+"\t us");  
        return duration;
    }
    
    public long toc() {
        endTime = System.nanoTime();
        duration = (endTime - startTime);  
        return duration;
    }
    
    public void setOptDotsConvergence(boolean flag) {
         for (int i=0;i<dots.size();i++){
                DotN nd = dots.get(i);       
                nd.allNeighborsHaveConverged=flag;
                nd.hasConverged=flag;
         }        
    }
    
    //---------------------------------------- CUDA FUNCTIONS
    CUcontext context;
    public void setCUDAContext() {
        if ((this.CUDAEnabled)&&(CUDAUtils.get_NCuda_Devices()>0)) {
        	//System.out.println("setting CUDA context");
            //if (context==null) {
            	//System.out.println("context was null");
                cuInit(0);
                CUdevice device = new CUdevice();
                cuDeviceGet(device, CUDA_DEVICE_ID);
                context = new CUcontext();
                cuCtxCreate(context, 0, device);        

                fComputePairsOfDot = new CUfunction();
                fAVGBlocks = new CUfunction();
                fKDTree_FindFurthestBlocks = new CUfunction();
                fCalcDir = new CUfunction();
                fMapDots = new CUfunction();
                fSortBlocks = new CUfunction();
                fDispatchDots = new CUfunction();
                
                CUmodule module = new CUmodule();
                try {
                    byte [] ptxData = CUDAUtils.toZeroTerminatedStringByteArray(Optimizer.class.getClassLoader().getResourceAsStream("CUDAkernels/gpuForcesCompute.ptx"));
                    cuModuleLoadData(module, ptxData);
                    cuModuleGetFunction(fComputePairsOfDot, module, "forceCompute");    

                    ptxData = CUDAUtils.toZeroTerminatedStringByteArray(Optimizer.class.getClassLoader().getResourceAsStream("CUDAkernels/computeAVGBlocks.ptx"));
                    cuModuleLoadData(module, ptxData);
                    cuModuleGetFunction(fAVGBlocks, module, "computeAVG");

                    ptxData = CUDAUtils.toZeroTerminatedStringByteArray(Optimizer.class.getClassLoader().getResourceAsStream("CUDAkernels/computeKDTree_FindFurthest.ptx"));
                    cuModuleLoadData(module, ptxData);
                    cuModuleGetFunction(fKDTree_FindFurthestBlocks, module, "findFurthest"); 

                    ptxData = CUDAUtils.toZeroTerminatedStringByteArray(Optimizer.class.getClassLoader().getResourceAsStream("CUDAkernels/gpuCalcDir.ptx"));
                    cuModuleLoadData(module, ptxData);
                    cuModuleGetFunction(fCalcDir, module, "calcDir");

                    ptxData = CUDAUtils.toZeroTerminatedStringByteArray(Optimizer.class.getClassLoader().getResourceAsStream("CUDAkernels/gpuMapDots.ptx"));
                    cuModuleLoadData(module, ptxData);
                    cuModuleGetFunction(fMapDots, module, "mapDots");

                    ptxData = CUDAUtils.toZeroTerminatedStringByteArray(Optimizer.class.getClassLoader().getResourceAsStream("CUDAkernels/gpuSortBlocks.ptx"));
                    cuModuleLoadData(module, ptxData);
                    cuModuleGetFunction(fSortBlocks, module, "sortBlocks");

                    ptxData = CUDAUtils.toZeroTerminatedStringByteArray(Optimizer.class.getClassLoader().getResourceAsStream("CUDAkernels/gpuDispatchDots.ptx"));
                    cuModuleLoadData(module, ptxData);
                    cuModuleGetFunction(fDispatchDots, module, "dispatchDots");   

                } catch (IOException ex) {
                    Logger.getLogger(Optimizer.class.getName()).log(Level.SEVERE, null, ex);
                }
            /*} else {
            	System.out.println("retrieve context...");
            	//context.
                cuCtxSetCurrent(context);
            }*/
        } else {
        	CUDAEnabled=false;
        	//System.out.println("CUDA disabled!");
        	//System.out.println("If you know you have a CUDA capable device check this:");
        	//System.out.println("\t Does it have Compute Capabilities > 3 ?");
        	//System.out.println("\t Did you install CUDA download toolkit (https://developer.nvidia.com/cuda-downloads) ?");
        	//System.out.println("\t Please report in case of troubles in forum.imagej.net.");
        }
        if (CUDAEnabled) {
            //IJ.log("CUDA is enabled.");
        	//System.out.println("CUDA Enabled.");
            MAX_THREADS_PER_BLOCK=CUDAUtils.getDeviceAttribute(CUDA_DEVICE_ID, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
        }
    }
    
    void buildTreeGPU(ArrayList<BlockOfDots> iniBlocks, int minNumberOfInteractionsPerBlock){  
        // 1 - PUSH dots if not alreadyDone
        if (gpuDots==null) {gpuDots = new GPUDots();}
        if (gpuDots.hasBeenPushed==false) {gpuDots.push(dots);}
        // 2 - Allocate and create gpuBlocks if needed
        if (gpuAllBlocks==null) {gpuAllBlocks = new GPUBlockList();}
        if (gpuWorkingBlocks_in==null) {gpuWorkingBlocks_in = new WGPUBlockList();}
        if (gpuWorkingBlocks_out==null) {gpuWorkingBlocks_out = new WGPUBlockList();}
        if (gpuIgnoredBlocks==null) {gpuIgnoredBlocks = new GPUBlockList();}
        gpuAllBlocks.resetBlocks();
        // 3 - Initialize blocks     
        gpuWorkingBlocks_in.push(iniBlocks);
        int gpuBlockSize=Optimizer.MAX_THREADS_PER_BLOCK;
        //gpuWorkingBlocks_in.writeGPUBlockInfos();
        // 3 - LOOP  
        //          Split blocks
        //          Feed allBlocks and ignoredBlocks
        //     UNTIL workingBlocks are empty
        //GPUSplitBlocks(gpuWorkingBlocks_in,gpuWorkingBlocks_out,gpuAllBlocks,gpuIgnoredBlocks, gpuBlockSize, minNumberOfInteractionsPerBlock, this.thresholdSuperDotCreationWorthTest);
        //int loopNumber=0;
        GPUBlockTimeReport = new long[4];
        //System.out.println("GPU Tree details");
        while (gpuWorkingBlocks_in.nBlocks>0) {
            // roundNumber++;
            // Split and dispatch
            //System.out.println("\n loopNumber ="+(++loopNumber));
            int avgBlock = gpuWorkingBlocks_in.getAverageNumberOfRequiredThreadPerBlock(GPUBlockList.ONE_THREAD_PER_DOT);
            int gpuNewSize  = 1;
            if (avgBlock>=Optimizer.MAX_THREADS_PER_BLOCK) {
                gpuNewSize = Optimizer.MAX_THREADS_PER_BLOCK;
            } else {
                while (gpuNewSize<avgBlock) {gpuNewSize*=2;}
            } 
            //System.out.println("roundNumber=\t "+roundNumber+"\t avgBlock=\t "+avgBlock+ "\t GPUBlockSize = "+gpuNewSize);
            gpuBlockSize=gpuNewSize;
            gpuWorkingBlocks_in.mapBlocksToGPU(GPUBlockList.ONE_THREAD_PER_DOT, gpuBlockSize);//, true);
            GPUSplitBlocks(gpuWorkingBlocks_in,gpuWorkingBlocks_out,gpuAllBlocks,gpuIgnoredBlocks, gpuBlockSize, minNumberOfInteractionsPerBlock, this.thresholdSuperDotCreationWorthTest);        
            // Switch in and out working blocks
            //System.out.println("gpuWorkingBlocks_in.nBlocks=\t "+gpuWorkingBlocks_in.nBlocks+"\t gpuWorkingBlocks_in.nDots=\t "+gpuWorkingBlocks_in.nDots);
            //System.out.println("gpuAllBlocks.nBlocks=\t "+gpuAllBlocks.nBlocks+"\t gpuAllBlocks.nDots=\t "+gpuAllBlocks.nDots);
            //System.out.println("gpuIgnoredBlocks.nBlocks=\t "+gpuIgnoredBlocks.nBlocks+"\t gpuIgnoredBlocks.nDots=\t "+gpuIgnoredBlocks.nDots);
            WGPUBlockList temp = gpuWorkingBlocks_in;
            gpuWorkingBlocks_in = gpuWorkingBlocks_out;
            gpuWorkingBlocks_out = temp;

        }
        gpuAllBlocks.hasBeenPushed=true;
        // POP ignored Blocks
        //System.out.println("Center = \t"+(GPUBlockTimeReport[0]/1000.)+"\t Dir = \t"+(GPUBlockTimeReport[1]/1000.)+"\t Map = \t"+(GPUBlockTimeReport[2]/1000.)+"\t Dispatch = \t"+(GPUBlockTimeReport[3]/1000.));
    }
    
    void GPUSplitBlocks(WGPUBlockList gBlockListIn, 
                               WGPUBlockList gBlockListOut,
                               GPUBlockList gBlockListFinal,
                               GPUBlockList gBlockListIgnored, 
                               int gpuBlockSize,
                               int minInteract, 
                               int minPointsToKeep) {
        // I need per point : px, py, pz, radius, converged to be efficient
        // The most annoying function to write
        // We always assume 4 children blocks per block as output
        // 1 - GPU : computeCenterAndConvergence in gBlockListIn
        //tic();
        gBlockListIn.computeCenters(gpuDots, gpuBlockSize);
        //GPUBlockTimeReport[0]+=toc();
        // 2 - GPU : findDirOfFurthestPoint in gBlockListIn
        //tic();
        gBlockListIn.computeDirFurthestPoint(gpuDots, gpuBlockSize);
        //GPUBlockTimeReport[1]+=toc();
        // 3 - GPU : mapDotsIntoNewBlocksAndGetRank
        //tic();
        gBlockListIn.mapDotsIntoNewBlocksAndGetRank(gpuDots, gpuBlockSize, limitInteractAttract);
        //GPUBlockTimeReport[2]+=toc();
        // 4 - CPU : 
        //    - collectRankingData 
        //    - decide for each block : KEEP, DISCARD or SPLIT
        //    - compute new memory location for all points of all block
        // 5 - GPU : copy data to new blocks location
        //tic();
        gBlockListIn.dispatchBlocks(gpuDots, gBlockListOut, gBlockListFinal, gBlockListIgnored, gpuBlockSize, minInteract, minPointsToKeep);
        //GPUBlockTimeReport[3]+=toc();
        
    }
    
    void computeForcesperPairsOfDotGPU(int gpuBlockSize) {
        // PLAN : 
        // 1 - PUSH dots if not alreadyDone
        if (gpuDots==null) {        
            gpuDots = new GPUDots();
        }
        if (gpuDots.hasBeenPushed==false) {    
            //tic();
            gpuDots.push(dots);
            //System.out.print("GPUDotsPushed=\t"+(toc()/1000)+"\t");
        }
        gpuDots.setAllNeighborsConvergenceTo1(); // a bit dirty...
        // 2 - Ensure gpuAllblocks is pushed
        if (gpuAllBlocks==null) {gpuAllBlocks = new GPUBlockList();}
        /*if (gpuAllBlocks.hasBeenPushed==false) {
            gpuAllBlocks.push(allBlocks);
        }*/
        //int gpuBlockSize=Optimizer.MAX_THREADS_PER_BLOCK;
        gpuAllBlocks.mapBlocksToGPU(GPUBlockList.ONE_THREAD_PER_INTERACTION,gpuBlockSize);//, false);
        // 2 - Compute Forces
        GPUComputeForces(gpuDots, gpuAllBlocks, gpuBlockSize);    
        // 3 - POP dots
        gpuDots.pop(dots);
    }
    
    void GPUComputeForces(GPUDots gDots, GPUBlockList gBlocks, int gpuBlockSize) {
        Pointer kernelParameters = Pointer.to(
                    // Dots properties
                    Pointer.to(gDots.iGA_Float[GPUDots.PX].gpuArray), Pointer.to(gDots.iGA_Float[GPUDots.PY].gpuArray), Pointer.to(gDots.iGA_Float[GPUDots.PZ].gpuArray), 
                    Pointer.to(gDots.iGA_Float[GPUDots.NX].gpuArray), Pointer.to(gDots.iGA_Float[GPUDots.NY].gpuArray), Pointer.to(gDots.iGA_Float[GPUDots.NZ].gpuArray), 
                    Pointer.to(gDots.iGA_Float[GPUDots.FX].gpuArray), Pointer.to(gDots.iGA_Float[GPUDots.FY].gpuArray), Pointer.to(gDots.iGA_Float[GPUDots.FZ].gpuArray), 
                    Pointer.to(gDots.iGA_Float[GPUDots.RFX].gpuArray), Pointer.to(gDots.iGA_Float[GPUDots.RFY].gpuArray), Pointer.to(gDots.iGA_Float[GPUDots.RFZ].gpuArray),
                    Pointer.to(gDots.iGA_Float[GPUDots.MX].gpuArray), Pointer.to(gDots.iGA_Float[GPUDots.MY].gpuArray), Pointer.to(gDots.iGA_Float[GPUDots.MZ].gpuArray),
                    Pointer.to(gDots.iGA_Float[GPUDots.SUPER_DOT_RADIUS_SQUARED].gpuArray),
                    Pointer.to(gDots.iGA_Float[GPUDots.RELAXED].gpuArray),
                    Pointer.to(gDots.iGA_Int[GPUDots.N_NEIGH].gpuArray),
                    Pointer.to(gDots.iGA_Int[GPUDots.CELL_ID].gpuArray),                   
                    Pointer.to(gDots.iGA_Int[GPUDots.HAS_CONVERGED].gpuArray),
                    Pointer.to(gDots.iGA_Int[GPUDots.ALL_NEIGHBORS_HAVE_CONVERGED].gpuArray),                  
                    Pointer.to(gDots.iGA_Int[GPUDots.ALL_NEIGHBORS_HAVE_CONVERGED_PREVIOUSLY].gpuArray),                    
                    // Blocks Properties
                    Pointer.to(gBlocks.iGA_addrStartBlock0.gpuArray),Pointer.to(gBlocks.iGA_nPtBlock0.gpuArray),
                    Pointer.to(gBlocks.iGA_addrStartBlock1.gpuArray),Pointer.to(gBlocks.iGA_nPtBlock1.gpuArray),
                    Pointer.to(gBlocks.iGA_blockLevel.gpuArray),
                    Pointer.to(gBlocks.iGA_idBlock.gpuArray),
                    Pointer.to(gBlocks.iGA_offsIntBlock.gpuArray),
                    Pointer.to(gBlocks.iGA_arrayDotsIndexes.gpuArray),    
                    // Optimizer / Forces properties
                    Pointer.to(new float[]{d_0}),
                    Pointer.to(new float[]{k_align}),
                    Pointer.to(new float[]{k_bend}),
                    Pointer.to(new float[]{radiusTresholdInteract}),
                    Pointer.to(new float[]{ka}),
                    Pointer.to(new float[]{pa}),
                    Pointer.to(new float[]{pr}),
                    Pointer.to(new float[]{maxDisplacementPerStep})
            );
        cuLaunchKernel(fComputePairsOfDot, 
                           gBlocks.nGPUBlocks,  1, 1,             // Grid dimension 
                           gpuBlockSize, 1, 1,      // Block dimension
                           6*Sizeof.INT, null,                         // Shared memory size and stream 
                           kernelParameters, null                      // Kernel- and extra parameters
        ); 
        cuCtxSynchronize(); 
    }
    
    void resetGPUFlags() {
        // dots pushed alreadydone set to false
        if (!(gpuDots==null)) {
            gpuDots.hasBeenPushed=false;
        }
        if (!(gpuAllBlocks==null)) {
            gpuAllBlocks.hasBeenPushed=false;
            gpuAllBlocks.nDots=0;
            gpuAllBlocks.nBlocks=0;
        }
        if (!(gpuWorkingBlocks_in==null)) {
            gpuWorkingBlocks_in.nBlocks=0;
            gpuWorkingBlocks_in.nDots=0;
        }
        if (!(gpuWorkingBlocks_out==null)) {
            gpuWorkingBlocks_out.nBlocks=0;
            gpuWorkingBlocks_out.nDots=0;
        }
        if (!(gpuIgnoredBlocks==null)) {
            gpuIgnoredBlocks.nDots=0;
            gpuIgnoredBlocks.nBlocks=0;
        }
    }
    
    public void freeGPUMem(){
    	this.resetGPUFlags();
        if (gpuAllBlocks!=null) gpuAllBlocks.freeMem();
        if (gpuIgnoredBlocks!=null) gpuIgnoredBlocks.freeMem();
        if (gpuWorkingBlocks_in!=null) gpuWorkingBlocks_in.freeMem();
        if (gpuWorkingBlocks_out!=null) gpuWorkingBlocks_out.freeMem();
        if (gpuDots!=null) gpuDots.freeMem();
    }
}