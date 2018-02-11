package eu.kiaru.limeseg;


import ij.ImagePlus;
import ij.WindowManager;

import ij.gui.Overlay;
import ij.gui.PointRoi;
import ij.gui.Roi;
import ij.macro.Functions;
import ij.process.FloatPolygon;
import net.imglib2.RandomAccessibleInterval;

import java.awt.Color;
import java.text.DecimalFormat;
import java.util.Iterator;
import java.util.function.Predicate;
import java.util.ArrayList;

import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import eu.kiaru.limeseg.gui.JFrameLimeSeg;
import eu.kiaru.limeseg.gui.JOGL3DCellRenderer;
import eu.kiaru.limeseg.gui.JTableCellsExplorer;
import eu.kiaru.limeseg.ij1script.HandleIJ1Extension;
import eu.kiaru.limeseg.ij1script.IJ1ScriptableMethod;
import eu.kiaru.limeseg.io.IOXmlPlyLimeSeg;
import eu.kiaru.limeseg.opt.Optimizer;
import eu.kiaru.limeseg.struct.Cell;
import eu.kiaru.limeseg.struct.CellT;
import eu.kiaru.limeseg.struct.DotN;
import eu.kiaru.limeseg.struct.PolygonSkeleton;
import eu.kiaru.limeseg.struct.Skeleton2D;
import eu.kiaru.limeseg.struct.Vector3D;
import eu.kiaru.limeseg.demos.SegCElegans;
import eu.kiaru.limeseg.gui.DisplayableOutput;
/**
 * Helper class for LimeSeg segmentation
 * The segmentation objects are called Cells
 * They are contained within the Array allCells
 * Each Cell contains one or Many CellT, which is a Cell at a timepoint
 * Each CellT contains many dots. It a represent a 3D object
 * This class contains many functions to:
 * - store / retrieves these objects (I/O section)
 * - display them in 2D on ImagePlus image
 * - display them in 3D in a custom 3D viewer
 * - prepare the segmentation (optimizer preparation : parameters and seeds)
 * - launch the segmentation
 * - analyse the CellT objects (mesh reconstruction / surface and volume measurement)
 * - manipulate the object (remove / generate / copy paste)
 * @author Nicolas Chiaruttini 
 */
@Plugin(type = Command.class, menuPath = "Plugins>LimeSeg>Show GUI")
public class LimeSeg implements Command {   
	 
    static public Optimizer opt;                   			// Associated Optimizer, filled with dots currently being optimized
    static public JOGL3DCellRenderer jcr;					// Associated 3D Viewer
    
    static public ArrayList<Cell> allCells;  				// List of cells currently stored by LimeSeg
    static public boolean notifyCellExplorerCellsModif;		// Flags any change of allCells to the cellExplorer tab (Swing GUI)
   
    static public ArrayList<DotN> dots_to_overlay;			// Set of dots being overlayed on workingImP
    
    static public Cell currentCell;    						// Current selected Cell of limeseg
    
    static public DotN currentDot;							// Current selected Dot of limeseg
   
	static public ImagePlus workingImP;            			// Current working image in IJ1 format - Used for IJ1 interaction (ROI, and JOGLCellRenderer synchronization)
	
	static public int currentChannel=1;						// specifies the Channel of the workingImP to work with
    	
	static public int currentFrame=1;						// specifies the Frame of the workingImP to work with
    
    public static ArrayList<DotN> copiedDots;				// Internal equivalent of Clipboard for set of dots in limeseg
    static Skeleton2D skeleton;  							// Current skeleton of Limeseg. Serves to generate initial shapes in 3D
    
    static public boolean optimizerIsRunning=false;			// Flags if optimizer is running
    public static boolean requestStopOptimisation=false;	// Flags is a stop optimisation command has been requested
    
    // GUI
    static JFrameLimeSeg jfs3Di;							// LimeSeg GUI (Swing)
    static JTableCellsExplorer cExplorer;					// Cell Explorer (Swing table)
    
    // Macro extensions IJ1
    static boolean extensionsHaveBeenRegistered = false;	// IJ1 scripting with macro extensions
    // End of static variables    
    
    /**
     * Initializes LimeSeg:
     * 	- creates an optimizer
     *  - creates list of dots to overlay
     */
    public void initialize() {
    	if (copiedDots==null) {
    		copiedDots= new ArrayList<>();
    	}
    	if (allCells==null) {
            allCells=new ArrayList<>();
            notifyCellExplorerCellsModif=true;
            notifyCellRendererCellsModif=true;
        }
        if (dots_to_overlay==null) {
        	dots_to_overlay = new ArrayList<>();
        }
    	initOptimizer();
    }    
    
    void initOptimizer() {
        if (opt==null) {        
                opt = new Optimizer(this);
        }
    } 
    
    /**
     *  Tabs in LimeSeg GUI
     */
    final static public String 
    		TS=",", // target separator, do not use comma in other
    		IO="I/O",
    		VIEW_2D="2D View",
    		VIEW_3D="3D View",
    		OPT="Optimizer",
    		CURRENT_CELL="Current Cell",
    		STATE="STATE",
    		CURRENT_CELLT="Current CellT",
    		CURRENT_DOT="Current Dot",
    		CLIPPED_DOTS="Clipped Dots",
    		BENCHMARK="BenchMark";		
    		
    
    /**
     * returns the state of LimeSeg as a String:
     * @return a string containing working image + ZScale + currentframe +currenchannel
     */
    @DisplayableOutput(pr=0)
    static public String getLmsState() {
    	String str="";
    	str+="State:\n";
    	if (workingImP==null) {
        	str+="\t img     = null\n";
    	} else {
        	str+="\t img = "+workingImP.getTitle()+"\n";
    	}
    	str+="\t ZScale  = "+opt.getOptParam("ZScale")+"\n";
    	str+="\t frame   = "+currentFrame+"\n";
    	str+="\t channel = "+currentChannel+"\n";
    	str+="Cells:\n";
    	str+="\t #cells="+allCells.size()+"\n";
    	return str;    	
    }
    
    /**
     * get optimizer state as a String
     * @return number of dots in optimizer, if it is optimizing and the image
     */
    @DisplayableOutput(pr=5)
    static public String getOptState() {
    	String str="";
    	str+="Optimizer:\n";
    	str+="\t #dots="+opt.dots.size()+"\n";
    	//str+="\t "+"\n";
    	str+="\t img="+((opt.image3DInfinite==null)?"-":opt.image3DInfinite.toString())+"\n";
    	str+="\t isOptimizing="+optimizerIsRunning+"\n";
    	return str;
    }
    
    /**
     * get current Cell state
     * @return  infos about the current selected cell as a String
     */
    @DisplayableOutput(pr=1)
    static public String getCellState() {
    	String str="";
    	str+="Cell:\n";
    	str+="\t current="+((currentCell==null)?"null":currentCell.id_Cell)+"\n";
    	str+="\t #cellT="+((currentCell==null)?"-":currentCell.cellTs.size())+"\n";
    	str+="\t channel="+((currentCell==null)?"-":currentCell.cellChannel)+"\n";
    	str+="\t color="+((currentCell==null)?"[]\n":"["+(new DecimalFormat("#.##").format(currentCell.color[0]))+";"
    												  +(new DecimalFormat("#.##").format(currentCell.color[1]))+";"
    												  +(new DecimalFormat("#.##").format(currentCell.color[2]))+";"
    												  +(new DecimalFormat("#.##").format(currentCell.color[3]))+"]"+"\n");
    	return str;
    }
    
    /**
     * get copied dots properties
     * @return infos within a String about the dots that have been copied
     */
    @DisplayableOutput(pr=6)
    static public String getClippedDotsState() {
    	String str="";
    	str+="ClippedDots:\n";
    	if (copiedDots!=null) {
    		str+="\t #dots="+copiedDots.size()+"\n";
    	}
    	return str;
    }
    
    /**
     * get CellT state
     * @return infos as a String about the current cell timepoint (number of dots + has the mesh been reconstructed ?)
     */
    @DisplayableOutput(pr=2)
    static public String getCellTState() {
    	String str="CellT:\n";
    	if (currentCell!=null) {
            CellT ct = currentCell.getCellTAt(currentFrame);
            if (ct!=null) {
            	str+="\t frame="+currentFrame+"\n";
            	str+="\t #dots="+ct.dots.size()+"\n";
            	str+="\t tesselated="+ct.tesselated+"\n";
            }
        }   else {
        	str+="\t null \n";
        }
    	return str;
    }
    
    /**
     * Generates a grid of surfels in 2D
     * @param d_0 equilibrium distance between surfels in pixels
     * @param pxi x position start
     * @param pyi y position start
     * @param pxf x position end
     * @param pyf y position end
     * @param pz  z position of the sheet
     * @return List of dots generated
     */
    static public ArrayList<DotN> makeXYSheet(float d_0, float pxi, float pyi, float pxf, float pyf, float pz) {
        ArrayList<DotN> ans = new ArrayList<DotN>();
        for (float x=pxi;x<pxf;x+=d_0) {
            for (float y=pyi;y<pyf;y+=d_0) {
                Vector3D pos = new Vector3D(x,y,pz);
                Vector3D normal = new Vector3D(0,0,1);
                DotN nd=new DotN(pos,normal);                            
                ans.add(nd);      
            }
        }
        return ans;
    }
    
    /**
     * Generates a sphere surface made of surfels. All units in pixels
     * @param d_0 equilibrium distance between surfels in pixels
     * @param px sphere center X
     * @param py sphere center Y
     * @param pz sphere center Z
     * @param radius sphere radius
     * @return
     */
    static public ArrayList<DotN> makeSphere(float d_0, float px, float py, float pz, float radius) {
        ArrayList<DotN> ans = new ArrayList<DotN>();
        float dlat=d_0/radius;
        float lat_i=(float) (-java.lang.Math.PI/2);
        float lat_f=(float) (java.lang.Math.PI/2);//-dlat;               
        for (float lat=lat_i+dlat; lat<(lat_f); lat=lat+dlat) {
            // We put points around a circle of radius = radius.cos(lat)
            float R=(float) (radius*java.lang.Math.cos(lat));            
            float N=(int)(java.lang.Math.PI*2.0*R/d_0);
            float dAngle=(float) (java.lang.Math.PI*2.0/N);
            for (float i=0;i<N;i++) {
                Vector3D normal = new Vector3D((float)(R*java.lang.Math.sin(i*dAngle)),
                                                 (float)(R*java.lang.Math.cos(i*dAngle)),
                                                 (float)(radius*java.lang.Math.sin(lat)));
                Vector3D pos = new Vector3D(px+normal.x,
                                              py+normal.y,
                                              pz+normal.z);
                DotN nd=new DotN(pos,normal);                                       
                ans.add(nd);               
            }
        }
        return ans;
    }
    
    /*
     * ------------------- Methods and attributes used to avoid conflicts between segmentation threads and 3D viewer thread
     */
    static public boolean notifyCellRendererCellsModif;		// Flags any change of allCells to the 3D Viewer (JOGLCellRenderer)
															// When notified, the renderer asks the segmentation thread to give his updated data
    static private boolean bufferHasBeenFilled;

    /**
     * 	Function that triggers an update of the 3D viewer
     *  - Located in this class because this has to be done in the thread of the optimizer, 
     *  and not in the thread of the viewer...
     *  a bit ugly though
     */
    public static void requestFillBufferCellRenderer() {
        notifyCellRendererCellsModif=false;
        if (optimizerIsRunning) {            
            opt.requestFillBufferRenderer=true; //System.out.println("La seg tourne, on demande de remplir le buffer");
        } else {
            fillBufferCellRenderer(); //System.out.println("La seg tourne pas, on appelle le remplissage de buffer");
        }        
    }    
    
    /*
     *  Helper function for 3D viewer 
     */
    public static boolean getBuffFilled() {
        return bufferHasBeenFilled;
    }
    
    /*
     * Helper function for 3D viewer
     */
    public static void setBuffFilled(boolean flag) {
        bufferHasBeenFilled=flag;
    }
    
    /*
     * Helper function for 3D viewer
     */
    public static void fillBufferCellRenderer() {
        jcr.fillBufferCellRenderer_PC();
        jcr.fillBufferCellRenderer_TR();
    }
    
    public static void fillBufferCellRenderer(ArrayList<DotN> aList) {
        jcr.fillBufferCellRenderer_PC(aList);
    }
    
    /*
     * ------------------- End of methods and attributes used to avoid conflicts between segmentation threads and 3D viewer thread
     */
    
    /**
     * adds a cell timepoint to list of points which will be overlayed
     * @param ct cell timepoint to add
     */
    static public void addToOverlay(CellT ct) {
        for (int i=0;i<ct.dots.size();i++) {
            DotN dn= ct.dots.get(i);
            dots_to_overlay.add(dn);           
        }
    }
    
    /**
     * adds a cell to list of points which will be overlayed (i.e. all associated cell timepoints)
     * @param c cell timepoint to add
     */
    static public void addToOverlay(Cell c) {
        for (int i=0;i<c.cellTs.size();i++) {
            CellT ct= c.cellTs.get(i);
            addToOverlay(ct);
        }
    }  
    
    /**
     * adds all points of all all cellt found to be at the specified frame
     * @param frame
     */
    public void addToOverlay(int frame) {
        for (int i=0;i<allCells.size();i++) {
            Cell c= allCells.get(i);
            CellT ct = c.getCellTAt(frame);
            if (ct!=null) {
                addToOverlay(ct);
            }
        }
    }  
    
    /**
     * Sets the image which will be used by the optimizer
     * @param imageName image which has to be found within the static WindowManager
     */
    @IJ1ScriptableMethod(target=STATE, ui="ImageChooser", tt="(String imageName)", pr=-2)
    public static void setWorkingImage(String imageName) {
    	System.out.println("imageName="+imageName);
   		ImagePlus imtest = WindowManager.getImage(imageName);
   		if (imtest!=workingImP) {
   			workingImP=imtest;
   			LimeSeg.notifyCellExplorerCellsModif=true;
   			updateWorkingImage();    
   		}
    }
    
    static void updateWorkingImage() {
    	if ((!optimizerIsRunning)&&(workingImP!=null)) {
            setWorkingImage(workingImP,currentChannel,currentFrame);
    	} else {
    		//IJ.log("Cannot change image : the Optimizer is running");
    	}
    }
    
    /**
     * Sets image to be used by the optimizer using imageID IJ1 reference
     * Can affect to currentframe and currentchannel synchronization
     * @param imageID
     * @param channel
     * @param frame
     */
    static public void setWorkingImage(int imageID, int channel, int frame) {
        setWorkingImage(WindowManager.getImage(imageID),channel, frame);
    }
    
    /**
     * Sets image to be used by the optimizer using ImagePlus IJ1 reference
     * Can affect to currentframe and currentchannel synchronization
     * @param imageID
     * @param channel
     * @param frame
     */
    static public void setWorkingImage(ImagePlus img, int channel, int frame) {        
        opt.setWorkingImage(img, channel, frame);  
        workingImP = img;
    }
    
    /**
     * Sets image to be used by the optimizer using RandomAccessibleInterval ImgLib2 Object
     * Can affect to currentframe and currentchannel synchronization
     * @param imageID
     * @param channel
     * @param frame
     */
    static public void setWorkingImage(RandomAccessibleInterval img, int channel, int frame) {        
        if (!optimizerIsRunning) {
        	opt.setWorkingImage(img, channel, frame);  
        	workingImP = null;
        }
    }
    
    /**
     * Adds a cell in the 3D viewer
     * @param c cell to be displayed by the 3D viewer
     */
    static public void putCellTo3DDisplay(Cell c) {
        make3DViewVisible();
        jcr.addCellToDisplay(c);
        notifyCellRendererCellsModif=true;
    }    
    
    /**
     * Displays a swing table with current Cells and corresponding CellTs present in LimeSeg
     */
    @IJ1ScriptableMethod(target=STATE, ui="STD", pr=-3)
    static public void showTable() {
    	//System.out.println("Unsupported showTable operation");
        if (cExplorer==null) {cExplorer=new JTableCellsExplorer();//workingImP);
                java.awt.EventQueue.invokeLater(new Runnable() {
                    public void run() {
                        cExplorer.setVisible(true);
                    }
                });
        } else {
            cExplorer.setVisible(true);
        }
    }
    
    //----------------- Optimizer
    
    static ArrayList<CellT> savedCellTInOptimizer = new ArrayList<>(); // For Optimizer cancellation
    
    /**
     * UNSTABLE WITH GPU MODE - save the state of the optimizer 
     * - can be restored with restoreOptState method 
     */
    @IJ1ScriptableMethod(target=OPT, ui="STD", tt="()", pr=1)
    static public void saveOptState() {
    	if (opt!=null) {
    		//savedOptDots.clear();
    		savedCellTInOptimizer.clear();
    		for (CellT ct:opt.cellTInOptimizer) {
    			CellT nct = ct.clone();
    			savedCellTInOptimizer.add(nct);
    		}
    	}
    }
    
    /**
     * UNSTABLE WITH GPU MODE - restore the state of the optimizer 
     * that was saved with saveOptState command 
     */
    @IJ1ScriptableMethod(target=OPT, ui="STD", tt="()", pr=1)
    static public void restoreOptState() {
    	opt.freeGPUMem();    	
    	if ((opt!=null)) {
    		ArrayList<CellT> ctInOptimizer = opt.cellTInOptimizer;    		
    		for (CellT ct:ctInOptimizer) {
    			Cell c = ct.c;
    			CellT ctToRemove = ct.c.getCellTAt(ct.frame);
    			System.out.println("c = "+ct.c.id_Cell);
    			ct.c.cellTs.remove(ctToRemove);
    			if (ct.c.cellTs.size()==0) {
    				boolean remove=true;
    				for (CellT ct2:savedCellTInOptimizer) {
    					if (ct.c==ct2.c) {
    						remove=false;
    					}    					
    				}
    				if (remove) {LimeSeg.allCells.remove(c);}
    			}
    		}
    		LimeSeg.clearOptimizer();    		

    		for (CellT ct:savedCellTInOptimizer) {
    			for (DotN dn:ct.dots) {
    				dn.ct=ct;
    			}
    			CellT ctToRemove = ct.c.getCellTAt(ct.frame);
    			ct.c.cellTs.remove(ctToRemove);
    			ct.c.cellTs.add(ct);
    			opt.addDots(ct);
    		}
    		LimeSeg.notifyCellExplorerCellsModif=true;
    		LimeSeg.notifyCellRendererCellsModif=true;
    	}
    }
    
    /**
     * Put all cellt from cells in the optimizer with respect of currentFrame and currentChannel 
     */
    @IJ1ScriptableMethod(target=OPT, ui="STD", tt="()", pr=6)
    static public void putAllCellsToOptimizerRespectChannel() {
    	for (Cell c:allCells) {
    		if ((c.cellChannel==LimeSeg.currentChannel)&&(c.getCellTAt(LimeSeg.currentFrame)!=null)) {
    			LimeSeg.opt.addDots(c.getCellTAt(LimeSeg.currentFrame));    		
    		}
    	}
    }
    
    /**
     * Set the working channel for 5D images
     * @param cChannel
     */
    @IJ1ScriptableMethod(target=STATE, ui="STD", tt="(int cChannel)", pr=1)
    static public void setCurrentChannel(int cChannel) {
    	currentChannel=cChannel;
    	updateWorkingImage();
    }
    
    /**
     * Set the spacing ratio between Z spacing and XY spacing (X and Y should be isotropic)
     * For instance if the image was acquired with pixels spacing = 0.2 um and z spacing = 0.6
     * zscale should be equal to 0.6/0.2 = 3
     * @param zscale
     */
    @IJ1ScriptableMethod(target=STATE, ui="STD", tt="(float zscale) // ratio between z spacing and xy spacing (x and y are assumed to isotropic). <br>"
    		+ "For instance if x and y pixel size are 0.2 microns and z sampling is 0.8 microns, then this property should be set to 0.8/0.2 = 4. ", pr=-1)
    static public void setZScale(float zscale) {
        opt.setOptParam("ZScale", zscale);
    }
    
    /**
     * Set the bounding box of the Optimizer. 
     * A string should be given formatted as : xmin, xmax, ymin, ymax, zmin, zmax
     * Units are pixels for xy and slice index for z : xmin, xmax
     * @param str
     */
    @IJ1ScriptableMethod(target=OPT, ui="STD", tt="(String str) // Format MinX,MaxX,MinY,MaxY,MinZ,MaxZ", pr=4)
    static public void setOptBounds(String str) {
        String[] parts=str.substring(0, str.length()).split(",");
            if (parts.length==6){
                int xmin=Integer.parseInt(parts[0]);                    
                int xmax=Integer.parseInt(parts[1]);
                int ymin=Integer.parseInt(parts[2]);                    
                int ymax=Integer.parseInt(parts[3]);
                int zmin=Integer.parseInt(parts[4]);                    
                int zmax=Integer.parseInt(parts[5]);
                opt.setOptParam("MinX", xmin);
                opt.setOptParam("MaxX", xmax);
                opt.setOptParam("MinY", ymin);
                opt.setOptParam("MaxY", ymax);
                opt.setOptParam("MinZ", zmin);
                opt.setOptParam("MaxZ", zmax);
            }
    }
    
    /**
     * Put all cellt from cells in the optimizer with respect to currentFrame
     */
    @IJ1ScriptableMethod(target=OPT, ui="STD", pr=2)
    static public void putAllCellTsToOptimizer() {
        for (int i=0;i<allCells.size();i++) {
            currentCell = allCells.get(i);
            putCurrentCellTToOptimizer();
        }
    }
    
    /**
     * Clears all dots contained in the optimizer
     */
    @IJ1ScriptableMethod(target=OPT, ui="STD", pr=2)
    static public void clearOptimizer() {
        if (opt!=null) {
        	opt.dots.clear();// = new ArrayList<>();//.clear();
        	opt.cellTInOptimizer.clear();
        }
    }
    
    /**
     * Asynchroneous request to stop the optimizer
     */
    @IJ1ScriptableMethod(target=OPT, ui="STD", pr=1)
    static public void stopOptimisation() {
        requestStopOptimisation = true;
    }
    
    /**
     * Sets optimizer parameters
     * Default values are put in paranthesis
     * Parameters are :
     *  - paramName : 
     *  		description 
     *  		(default_value; [min...max])
     *  - "d_0": 
     *  		equilibrium distance between surfels d_0 
     *  		(2; [0.5... ])
     *  - "normalForce": 
     *  		pressure applied on each surfel 
     *  		(0; [-0.04...0.04])
     * 	- "ka": 
     * 			amplitude of attractive force between surfels pair 
     * 			(when d>d_0)
     * 			(0.01)
     *  - "pa": 
     *  		power of attractive force 
     *  		(5)
     *  - "pr": 
     *  		power of repulsive force 
     *  		(9) 
     *  - "k_grad": 
     *  		amplitude of attractive force towards image maximum 
     *  		(0.03)
     *  - "k_bend":
     *  		(0.1) 
     *  - "k_align":
     *  		(0.05)
     *  - "fillHoles":
     *  		0 : no new surfel is generated
     *  		1 : if number of neighbors == generateDotIfNeighborEquals then a new surfel is generated
     *  		(1)
     *  - "rmOutliers":
     *  		0 : no surfel is removed
     *  		1 : if number of neighbors not in [rmIfNeighborBelow...rmIfNeighborAbove] then the surfel is removed
     *  		(1)
     *  - "attractToMax":
     *  		0 : no influence of the image (local maximum is not searched)
     *  		1 : surfels are attracted to local maxima
     *  		(1)
     *  - "radiusTresholdInteract":
     *  		in units of d_0, radius of surfels sphere of influence
     *  		(1.75)
     *  - "NStepPerR0":
     *  		sampling of f_dist
     *  		(5000)
     *  - "maxDisplacementPerStep":
     *  		in units of d_0, maximum displacement of surfels between two steps
     *  		(0.3)
     *  - "ageMinGenerate":
     *  		number of equilibrium iteration steps before a surfel is "active" (i.e. can be removed or generate a new one)
     *  		(10)
     *  - "rmIfNeighborBelow":
     *  		lower threshold to remove a surfel
     *  		(5)
     *  - "rmIfNeighborAbove":
     *  		high threshold to remove a surfel
     *  		(11)
     *  - "generateDotIfNeighborEquals":
     *  		if number of neighbors == generateDotIfNeighborEquals
     *  		then the surfel generated a new surfel
     *  		(6)
     *  - "radiusSearch":
     *  		in number of pixels, distance over which a local image maximum is search for
     *  		(5)
     *  - "radiusRes":
     *  		in number of pixels, sampling (=step size) over which maximum is looked for
     *  		(0.5) 
     *  - "radiusDelta":
     *  		in number of pixels along the normal vector, shift between the surfel position and the image range search
     *  		(0) 
     *  - "searchMode":
     *  		for future features (now: 0 = max; expect 1 = min, 2 = grad ...)
     *  		(0)
     *  - "convergenceTimestepSampling":
     *  		number of iterations performed before looking for convergence
     *  		(20)
     *  - "convergenceDistTreshold":
     *  		in d_0 units, distance traveled between two convergence steps below which is surfel is considered as having converged
     *  		(provided norm is also ok)
     *  		(0.1)
     *  - "convergenceNormTreshold":
     *  		if norm (change of normal) below threshold then the surfel could have converged
     *  		(provided dist is also ok)
     *  		(0.1) 
     *  - "radiusRelaxed":
     *  		if number of pixels, distance between the surfel and the image maximum below which
     *  		the surfel is relaxed (i.e. Fpressure and Fsignal are ignored = no normal force exerted)  
     *  		(1)
     *  - "ZScale":
     *  		ratio between Z spacing and XY spacing 
     *  		for instance, if Z slice spacing = 1 um and pixels of camera = 0.2 um, zscale = 1/0.2 = 5 
     *  		(1)
     *  - "MinX":
     *  		x minimal position of surfels (pixel)
     *  		(set by image dimension)
     *  - "MaxX":
     *  		x maximal position of surfels (pixel)
     *  		(set by image dimension)
     *  - "MinY":
     *  		y minimal position of surfels (pixel)
     *  		(set by image dimension)
     *  - "MaxY":
     *  		y maximal position of surfels (pixel)
     *  		(set by image dimension)
     *  - "MinZ":
     *  		minimal z slice position of surfels
     *  		(set by image dimension)
     *  - "MaxZ":
     *  		maximal z slice position of surfels
     *  		(set by image dimension)
     * @param paramName
     * @param value
     */
    @IJ1ScriptableMethod(target=OPT, ui="STD", tt="(String paramName, double value)", pr=3)
    static public void setOptimizerParameter(String paramName, double value) {
    	assert opt!=null;
        opt.setOptParam(paramName, (float)value);
    } 
    
    /**
     * Get current optimizer parameter (see setOptimizerParameter)
     * @param paramName
     * @param value is a table to fit with IJ1 macroextension
     */
    @IJ1ScriptableMethod(target=OPT, tt="(String paramName, Double value)")
    static public void getOptimizerParameter(String paramName, Double[] value) {
    	assert opt!=null;
        value[0]=opt.getOptParam(paramName);
    } 
    
    /**
     * Get current optimizer parameter (see setOptimizerParameter)
     * @param paramName
     * @return
     */
    public double getOptParam(String paramName) {
    	assert opt!=null;
        return opt.getOptParam(paramName);
    }
    
    /**
     * Runs NStep of iteration for the optimizer
     * Stops before if:
     * 	- all surfels have converged
     *  - requestStopOptimisation is set to true
     *  - no dots is present anymore
     *  if NStep is set to negative values, then the optimizer ignores NStep
     * @param NStep
     * @return true is the optimizer has converged, false otherwise
     */
    @IJ1ScriptableMethod(target=OPT, tt="(int NStep)", ui="STD", pr=-5, newThread=true)
    static public int runOptimisation(int NStep) {
        boolean hasConverged=false;
        if (optimizerIsRunning) {
           //IJ.log("Cannot run Optimisation : it is already running");
        } else {
            opt.setCUDAContext();
            long tInit=System.currentTimeMillis();
            optimizerIsRunning = true;
            int i=0;
            while (((i<NStep)||(NStep<0))&&(requestStopOptimisation==false)&&(opt.dots.size()>0))  {
                opt.nextStep();
                notifyCellRendererCellsModif=true;
                if (opt.getRatioOfDotsConverged()==1f) {
                     System.out.println("Everything has converged in "+i+" steps.");  
                     hasConverged=true;
                     break;
                }
                i++;
            }            
            optimizerIsRunning = false;
            requestStopOptimisation = false;
            opt.freeGPUMem();
            long tEnd=System.currentTimeMillis();
            System.out.println("Optimization time = \t "+((tEnd-tInit)/1000)+" s");
        }
        return hasConverged?1:0;
    }
    
    //----------------- 2D View
    /**
     * Put current cell to overlays (requires updateOverlay to be effective)
     */
    @IJ1ScriptableMethod(target=VIEW_2D, ui="STD", pr=0)
    static public void putCurrentCellToOverlay() { 
    	if (currentCell!=null) {
            addToOverlay(currentCell);
    	}    			
    }
    
    /**
     * Put dots of current user selected slice to overlays (requires updateOverlay to be effective)
     */
    @IJ1ScriptableMethod(target=VIEW_2D, ui="STD", pr=0)
    static public void putCurrentSliceToOverlay() { 
    	if (workingImP!=null) {
			float ZS=(float) opt.getOptParam("ZScale");
			int zSlice; //= workingImP.getZ();

	        if ((workingImP.getNFrames()!=1)||(workingImP.getNChannels()!=1)) {
	        	zSlice = workingImP.getZ();
	        } else {
	        	zSlice = workingImP.getSlice();
	        }
	    	for (Cell c:allCells) {
	    		CellT ct = c.getCellTAt(LimeSeg.currentFrame);
	    		if (ct!=null) {
	    			for (DotN dn:ct.dots) {
	    				if ((int)(dn.pos.z/ZS)==zSlice-1) {
	    					LimeSeg.dots_to_overlay.add(dn);
	    				}
	    			}
	    		}
	    	}
    	}
    }
    
    @IJ1ScriptableMethod(target=VIEW_2D, ui="STD", pr=0)
    static public void putCurrentTimePointToOverlay() { 
    	if (workingImP!=null) {
	    	for (Cell c:allCells) {
	    		CellT ct = c.getCellTAt(LimeSeg.currentFrame);
	    		if (ct!=null) {
	    			for (DotN dn:ct.dots) {
	    					LimeSeg.dots_to_overlay.add(dn);
	    			}
	    		}
	    	}
    	}
    }
    
    /**
     *  Clears image overlay (requires updateOverlay to be effective)
     */
    @IJ1ScriptableMethod(target=VIEW_2D, ui="STD", pr=0)
    static public void clearOverlay() {
        dots_to_overlay.clear();
    }
    
    /**
     * Adds all cells into image overlay (requires updateOverlay to be effective
     */
    @IJ1ScriptableMethod(target=VIEW_2D, ui="STD", pr=1)
    static public void addAllCellsToOverlay() {
        for (int i=0;i<allCells.size();i++) {
            Cell c= allCells.get(i);
            addToOverlay(c);
        }
    } 
    
    /**
     * updates Overlay of the working image with registeres dots to be overlayed
     */
    @IJ1ScriptableMethod(target=VIEW_2D, ui="STD", pr=2)
    static public void updateOverlay() {
        Overlay ov = new Overlay();
        if (workingImP!=null) {
	        workingImP.setOverlay(ov);
	        Iterator<DotN> i=dots_to_overlay.iterator();
	        float ZS=(float) opt.getOptParam("ZScale");
	        if ((workingImP.getNFrames()!=1)||(workingImP.getNChannels()!=1)) {
	            while (i.hasNext()) {
	                DotN nd = i.next();
	                PointRoi roi;
	                roi = new PointRoi(nd.pos.x,nd.pos.y);//,c);
	                Color color = new Color((int)(nd.ct.c.color[0]*255),(int)(nd.ct.c.color[1]*255),(int)(nd.ct.c.color[2]*255));
	                roi.setStrokeColor(color);
	                int zpos=1+(int)(nd.pos.z/ZS);
	                if ((zpos>0)&&(zpos<=workingImP.getNSlices())) {
	                    roi.setPosition(nd.ct.c.cellChannel, zpos, nd.ct.frame);
	                    ov.addElement(roi); 
	                }   
	            }   
	        } else {
	            while (i.hasNext()) {
	                DotN nd = i.next();
	                PointRoi roi;
	                roi = new PointRoi(nd.pos.x,nd.pos.y);//,c);   
	                Color color = new Color((int)(nd.ct.c.color[0]*255),(int)(nd.ct.c.color[1]*255),(int)(nd.ct.c.color[2]*255));
	                roi.setStrokeColor(color);
	                int zpos=1+(int)((float) (nd.pos.z)/(float) (ZS));
	                if ((zpos>0)&&(zpos<=workingImP.getNSlices())) {
	                    roi.setPosition(zpos);
	                    ov.addElement(roi);  
	                }
	            }
	        }
	        workingImP.updateAndDraw();
        }
    } 
    
    //----------------- 3D View
    /**
     * Updates the 3D viewer:
     * 	- triggers a notification that updates the buffered dots in 3D
     */
    @IJ1ScriptableMethod(target=VIEW_3D, ui="STD", pr=2)
    static public void update3DDisplay() {
        notifyCellRendererCellsModif=true;
    }
    
    /**
     * Displays the 3D viewer
     */
    @IJ1ScriptableMethod(target=VIEW_3D, ui="STD", newThread=true, pr=0)
    static public void make3DViewVisible() {
        if (jcr==null) {           
        	try {
				Thread.sleep(3000);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
            jcr = new JOGL3DCellRenderer();
            jcr.launchAnim();
        } else {
            if (jcr.glWindow.isVisible()==false) {
                jcr.glWindow.setVisible(true);
                jcr.animator.start();
            }
        }
    }
    
    /**
     * Puts current cell to 3D display
     */
    @IJ1ScriptableMethod(target=VIEW_3D, ui="STD", pr=2)
    static public void putCurrentCellTo3DDisplay() {
    	if (jcr!=null) {
    		putCellTo3DDisplay(currentCell);
    	}
    }
    
    /**
     * Sets the center of the 3D viewer
     * @param px
     * @param py
     * @param pz (in slice number / corrected by ZScale)
     */
    @IJ1ScriptableMethod(target=VIEW_3D, ui="STD", tt="(float px,float py, float pz)", pr=6)
    static public void set3DViewCenter(float px, float py, float pz) {
    	//make3DViewVisible();
    	if (jcr!=null) {
    		jcr.lookAt.x=px;            
        	jcr.lookAt.y=py;            
        	jcr.lookAt.z=pz*(float)opt.getOptParam("ZScale"); 
    	}
    }
    
    /**
     * Sets the rotation ( point of view of the 3D viewer)
     * @param rx
     * @param ry
     * @param rz
     */
    @IJ1ScriptableMethod(target=VIEW_3D, ui="STD", tt="(float rx,float ry, float rz)", pr=7)
    static public void set3DViewRot(float rx, float ry, float rz) {
    	//make3DViewVisible();            
        if (jcr!=null) {
        	jcr.view_rotx=rx;            
        	jcr.view_roty=ry;            
        	jcr.view_rotz=rz;
        }
    }  
    
    /**
     * Clears all objects in the 3D display
     */
    @IJ1ScriptableMethod(target=VIEW_3D, ui="STD", pr=3)
    static public void clear3DDisplay() {    	
        if (jcr!=null) {
        	notifyCellRendererCellsModif=true;
        	jcr.clearDisplayedCells(); 
        }
    }
    
    /**
     * Puts all the cells in the 3D display
     */
    @IJ1ScriptableMethod(target=VIEW_3D, ui="STD", pr=1, newThread=true)
    static public void putAllCellsTo3DDisplay() {
        //make3DViewVisible();
        if ((jcr!=null)&&(allCells!=null)) {
    	for (int i=0;i<allCells.size();i++) {
            Cell c= allCells.get(i);
            jcr.addCellToDisplay(c);
        }
        notifyCellRendererCellsModif=true;
        }
    }
    
    /**
     * Sets 3D View mode:
     * @param vMode
     * 	0 : Full view
     *  1 : Shows only dots below the displayed slice  
     *  2 : Shows only dots above the displayed slice  
     *  3 : Shows only dots within the displayed slice
     *  
     * 	8 : Full view of dots within the optimizer
     *  9 : Shows only dots of the optimizer below the displayed slice  
     *  10 : Shows only dots of the optimizer above the displayed slice  
     *  11 : Shows only dots of the optimizer within the displayed slice
     */
    @IJ1ScriptableMethod(target=VIEW_3D, ui="STD", tt="(int vMode)", pr=4)
    static public void set3DViewMode(int vMode) {
    	if (jcr!=null) {
    		jcr.setViewMode(vMode);
    	}
    }
    
    /**
     * Sets the zoom value of the 3D Viewer
     * @param zoom
     */
    @IJ1ScriptableMethod(target=VIEW_3D, ui="STD", tt="(float zoom)", pr=5)
    static public void set3DViewZoom(float zoom) {
    	if (jcr!=null) {
    		jcr.RatioGlobal=zoom;
    	}
    }
	
    //----------------- Current Cell
    /**
     * Sets cell display mode : dots or triangles (if tesselated)
     * @param vMode
     * 0: dots
     * 1: mesh
     */
    @IJ1ScriptableMethod(target=VIEW_3D+TS+CURRENT_CELL, ui="STD", tt="(int vMode)", pr=8)
	static public void setCell3DDisplayMode(int vMode) {
		if (currentCell!=null) {   
			currentCell.display_mode=vMode;
			notifyCellRendererCellsModif=true;
		}
	}
    
    /**
     * Gives a string identifier to the currentCell
     * @param id
     */
    @IJ1ScriptableMethod(target=CURRENT_CELL, ui="STD", tt="(String id)", pr=0)
	static public void setCellId(String id) {
    	if (currentCell!=null) {
		    currentCell.id_Cell = id;
		    notifyCellExplorerCellsModif=true;
    	}
	}
	
    /**
     * Creates a new cell
     */
	@IJ1ScriptableMethod(target=STATE, ui="STD", pr=2)
	static public void newCell() {
	    currentCell = new Cell(currentChannel);        
	    allCells.add(currentCell);          
	    notifyCellExplorerCellsModif=true;
	    notifyCellRendererCellsModif=true;
	}
    
	/**
	 * Select cell by its index
	 * @param index
	 */
	@IJ1ScriptableMethod(target=STATE, ui="STD", tt="(int index)", pr=3)
	static public void selectCellByNumber(int index) {
   		if ((index>=0)&&(index<allCells.size()))
			currentCell=allCells.get(index);
   	}
    
	/**
	 * Set color of current cell
	 * @param r
	 * @param g
	 * @param b
	 * @param a (alpha channel currently unsupported)
	 */
    @IJ1ScriptableMethod(target=CURRENT_CELL, ui="STD", tt="(float r, float g, float b, float a) : Set cell color (parameters are between 0 and 1).", pr=2)
    static public void setCellColor(float r, float g, float b, float a) {
    	if (currentCell!=null) {
            currentCell.color[0]=r;
            currentCell.color[1]=g;
            currentCell.color[2]=b;
            currentCell.color[3]=a;
        }
	    notifyCellRendererCellsModif=true;
    }
    
    /**
     * !Warning! removes all cells of LimeSeg
     */
    @IJ1ScriptableMethod(target=STATE, ui="STD", tt="")
    static public void clearAllCells() {
    	clearCell("ALL");
    	clearOverlay();
    }
    
    /**
     * Clear the cell named by arg from LimeSeg
     * @param arg
     */
    @IJ1ScriptableMethod(target=STATE, tt="(String arg)")
    static public void clearCell(String arg) {        
        if (arg.toUpperCase().equals("ALL")) {
            allCells.clear();      
            if ((jcr!=null)) {
            	jcr.clearDisplayedCells();
            }
            notifyCellExplorerCellsModif=true;
            notifyCellRendererCellsModif=true;
        } else if (findCell(arg)!=null) {
            Cell c = findCell(arg);           
            if (c!=null) {
                allCells.remove(c);                
                if ((jcr!=null)) {jcr.removeDisplayedCell(c);}
                notifyCellExplorerCellsModif=true;
                notifyCellRendererCellsModif=true;
            }
        } else {
            Cell c = currentCell;            
            if (c!=null) {
                allCells.remove(c);
                if ((jcr!=null)) {jcr.removeDisplayedCell(c);}
                notifyCellExplorerCellsModif=true;
                notifyCellRendererCellsModif=true;
            }
            currentCell=null;
        }            
    }
    
    /**
     * Clears current Cell
     */
    @IJ1ScriptableMethod(target=CURRENT_CELL, ui="STD", tt="")
    static public void clearCurrentCell() {
    	clearCell("");      
    }
    
    /**
     * Sets cell named by id as the current Cell
     * @param id
     */
    @IJ1ScriptableMethod(target=STATE, ui="STD", tt="(String id)", pr=4)
    static public void selectCellById(String id) {
    	currentCell=findCell(id);
    }

    //----------------- Current CellT
    /**
     * If a cell timepoint from a cell has been tesselated, value contains the surface of the cell
     * value is a table because of IJ1 macroextension compatibility
     * @param value
     */
    @IJ1ScriptableMethod(target=CURRENT_CELLT, tt="", pr=0)
    static public void getCellSurface(Double[] value) {
		value[0]=(double)0;
    	if (currentCell!=null) {
    		if (currentCell.getCellTAt(currentFrame)!=null) {
    			if (currentCell.getCellTAt(currentFrame).dots!=null) {
    				CellT ct = currentCell.getCellTAt(currentFrame);
    				if (ct.tesselated) {
    					value[0] = ct.getSurface();
    				} else {
    					value[0] = Double.NaN;
    				}
    			} else {
    				value[0]=0.0;
    			}
    		}
    	}
    }
    
    /**
     * If a cell timepoint from a cell has been tesselated, value contains the volume of the cell
     * value is a table because of IJ1 macroextension compatibility
     * @param value
     */
    @IJ1ScriptableMethod(target=CURRENT_CELLT, tt="", pr=0)
    static public void getCellVolume(Double[] value) {
		value[0]=(double)0;
    	if (currentCell!=null) {
    		if (currentCell.getCellTAt(currentFrame)!=null) {
    			if (currentCell.getCellTAt(currentFrame).dots!=null) {
    				CellT ct = currentCell.getCellTAt(currentFrame);
    				if (ct.tesselated) {
    					value[0] = ct.getVolume();
    				} else {
    					value[0] = Double.NaN;
    				}
    			} else {
    				value[0]=0.0;
    			}
    		}
    	}
    }
    
    /**
     * Set current working frame
     * @param cFrame
     */
    @IJ1ScriptableMethod(target=STATE, ui="STD", tt="(int cFrame)", pr=0)
    static public void setCurrentFrame(int cFrame) {
    	currentFrame=cFrame;
    	updateWorkingImage();
    }
    
    /**
     * Puts the cellt of currentCell at currentFrame into the optimizer
     */
    @IJ1ScriptableMethod(target=CURRENT_CELLT, ui="STD", pr=0)
    static public void putCurrentCellTToOptimizer() {
        if ((opt!=null)&&(currentCell!=null)) {
        	opt.addDots(currentCell.getCellTAt(currentFrame)); 
        }
    }
    
    /**
     * Constructs the mesh of the current Cell at the current frame
     * @return the number of free edges (hopefully 0)
     */
    @IJ1ScriptableMethod(target=CURRENT_CELLT, ui="STD", pr=4)
    static public int constructMesh() {
    	//System.out.println("Unsupported tesselation operation");
    	//return 0;
        int ans=-1;
        if (currentCell!=null) {
            CellT ct=currentCell.getCellTAt(currentFrame);
            if (ct!=null) {
                ans=ct.constructMesh();
                ct.modified=true;
                LimeSeg.notifyCellRendererCellsModif=true;
                LimeSeg.notifyCellExplorerCellsModif=true;
            }            
            currentCell.modified=true;
        }
        return ans;
    }
    
    /**
     * Puts dots of current CellT into the "clipboard"
     */
    @IJ1ScriptableMethod(target=CURRENT_CELLT, ui="STD", pr=1)
    static public void copyDotsFromCellT() {
        if (currentCell==null) {
            return;
        }
        CellT ct = currentCell.getCellTAt(currentFrame);
        if (ct==null) {
            return;
        }
        ArrayList<DotN> dTemp=ct.dots;
        copiedDots=new ArrayList<>();
        for (int i=0;i<dTemp.size();i++){
            DotN nd = dTemp.get(i);
            DotN nd_copy = new DotN(new Vector3D(nd.pos.x,nd.pos.y,nd.pos.z),new Vector3D(nd.Norm.x,nd.Norm.y,nd.Norm.z));
            nd_copy.N_Neighbor=nd.N_Neighbor;
            nd_copy.userDestroyable=nd.userDestroyable;
            nd_copy.userMovable=nd.userMovable;
            nd_copy.userGenerate=nd.userGenerate;
            nd_copy.userRotatable=nd.userRotatable;
            copiedDots.add(nd_copy);
        }
    }
    
    /**
     * Paste dots of "clipboard" into the current cellt
     */
    @IJ1ScriptableMethod(target=CURRENT_CELLT, ui="STD", pr=2)
    static public void pasteDotsToCellT() {      
        if (currentCell==null) {
            return;
        }
        currentCell.addDots(currentFrame, copiedDots);
        notifyCellExplorerCellsModif=true;
        notifyCellRendererCellsModif=true;
    }
    
    public static int getMaxFrames() {
    	int maxFrames=1;
    	if (workingImP!=null) {
    		maxFrames = workingImP.getNFrames();
    	}
    	if (allCells!=null) {
    		for (Cell c:allCells) {
    			for (CellT ct:c.cellTs) {
    				if (ct.frame>maxFrames) {
    					maxFrames=ct.frame;
    				}
    			}
    		}
    	}
    	return maxFrames;
    }
    
    /**
     * removes dots of current cell at current frame
     */
    @IJ1ScriptableMethod(target=CURRENT_CELLT, ui="STD", pr=3)
    static public void clearDotsFromCellT() {
        if (currentCell==null) {
            return;
        }
        CellT ct = currentCell.getCellTAt(currentFrame);
        if (ct==null) {
            return;
        } else {
            ct.tesselated=false;
            ct.triangles=new ArrayList<>();
            ct.dots=new ArrayList<>();
            ct.modified=true;
            currentCell.modified=true;
        }
        notifyCellExplorerCellsModif=true;
        notifyCellRendererCellsModif=true;
    }
    
	/**
	 * Removes flagged dots of current cell at current frame
	 */
	@IJ1ScriptableMethod(target=CURRENT_CELLT, ui="STD", pr=5)
	static public void removeFlaggedDots() {
	    Predicate<DotN> dotNPredicate = nd -> (nd.userDefinedFlag);
	    if (currentCell!=null) {
	    	CellT ct = currentCell.getCellTAt(currentFrame);
		    if (ct!=null) {
		        if (ct.dots.stream().anyMatch(dotNPredicate)) {
		            ct.dots.removeIf(dotNPredicate);
		        }
		    }
            notifyCellExplorerCellsModif=true;
            notifyCellRendererCellsModif=true;
	    }  	                         
	}
    
    //----------------- Current Dot    
	/**
	 * Select the dot number id as current dot
	 * @param id
	 */
	@IJ1ScriptableMethod(target=CURRENT_DOT, ui="STD", tt="(int id)", pr=0)
	static public void selectDot(int id) {
		currentDot = currentCell.getCellTAt(currentFrame).dots.get(id);
	}
	/**
	 * Sets current dot normal vector
	 * @param nx
	 * @param ny
	 * @param nz
	 */
	@IJ1ScriptableMethod(target=CURRENT_DOT, ui="STD", tt="(float nx, float ny, float nz)", pr=2)
	static public void setDotNorm(float nx, float ny, float nz) { 
	    if (currentDot!=null) {
	        currentDot.Norm.x=nx;
	        currentDot.Norm.y=ny;
	        currentDot.Norm.z=nz;
	    }   
	}
	/**
	 * Set current dot normal position
	 * @param px
	 * @param py
	 * @param pz
	 */
	@IJ1ScriptableMethod(target=CURRENT_DOT, ui="STD", tt="(float px, float py, float pz)", pr=1)
	static public void setDotPos(float px, float py, float pz) { 
	    if (currentDot!=null) {
	        currentDot.pos.x=px;
	        currentDot.pos.y=py;
	        currentDot.pos.z=pz;
	    }   
	}
	/**
	 * Set current dot property
	 * @param mov_
	 * @param rot_
	 * @param des_
	 * @param gen_
	 */
	@IJ1ScriptableMethod(target=CURRENT_DOT, ui="STD", tt="(int mov_,int rot_, int des_, int gen)", pr=3)
	static public void setDotProps(int mov_, int rot_, int des_, int gen_) {
        boolean mov=(mov_==1);
        boolean rot=(rot_==1);
        boolean des=(des_==1);
        boolean gen=(gen_==1);
        if (currentDot!=null) {
            currentDot.userMovable=mov;
            currentDot.userRotatable=rot;
            currentDot.userDestroyable=des;
            currentDot.userGenerate=gen;
        }		
	}
	/**
	 * Set current dot flag -> can then use removedflaggeddots to remove them
	 * @param flag
	 */
	@IJ1ScriptableMethod(target=CURRENT_DOT, ui="STD", tt="(int flag)", pr=4)
	static public void setDotFlag(int flag) {
	    if (currentDot!=null) {
	        currentDot.userDefinedFlag = (flag==1);
	    }   
	}
	/**
	 * Removes the current dot
	 */
	@IJ1ScriptableMethod(target=CURRENT_DOT, ui="STD", pr=5)
	static public void removeDot() {		
	    if (opt.dots.contains(currentDot)) {
	        opt.dots.remove(currentDot);
	        currentCell.getCellTAt(currentFrame).dots.remove(currentDot);
	        currentDot=null;
	    } else {
	        currentCell.getCellTAt(currentFrame).dots.remove(currentDot);
	        currentDot=null;
	    }   
	}

	
	//------------- Clipped Dots
    /**
     * normal vectors of clipped dots *=-1
     */
    @IJ1ScriptableMethod(target=CLIPPED_DOTS, ui="STD", pr=6)
    static public void invertClippedDotsPolarity() {
        if (copiedDots!=null) {
            for (DotN dn : copiedDots) {
                dn.Norm.x*=-1;
                dn.Norm.y*=-1;
                dn.Norm.z*=-1;
            }
        }
    }
    /**
     * Set properties of clipped dots (movable, ratable, destroyable, generate
     * @param mov_
     * @param rot_
     * @param des_
     * @param gen_
     */
    @IJ1ScriptableMethod(target=CLIPPED_DOTS, ui="STD", tt="(int mov_,int rot_, int des_, int gen_)", pr=5)
    static public void setClippedDotsProps(int mov_, int rot_, int des_, int gen_) {
        boolean mov=(mov_==1);
        boolean rot=(rot_==1);
        boolean des=(des_==1);
        boolean gen=(gen_==1);
        if (copiedDots!=null)
        for (int i=0;i<copiedDots.size();i++) {
        	DotN dn=copiedDots.get(i);
            dn.userMovable=mov;
            dn.userRotatable=rot;
            dn.userDestroyable=des;
            dn.userGenerate=gen;
        }
    }
    /**
     * Translate clipped dots
     * @param tx
     * @param ty
     * @param tz
     */
    @IJ1ScriptableMethod(target=CLIPPED_DOTS, ui="STD", tt="(float tx,float ty,float tz)", pr=4)
    static public void translateDots(float tx,float ty,float tz) {
        float ZS=(float)opt.getOptParam("ZScale");
        if (copiedDots!=null)
        for (int i=0;i<copiedDots.size();i++) {
                DotN dn=copiedDots.get(i);
                dn.pos.x+=tx;
                dn.pos.y+=ty;
                dn.pos.z+=tz*ZS;
            }
    }
    /**
     * Starts a new skeleton
     */
  	@IJ1ScriptableMethod(target=CLIPPED_DOTS, ui="STD", pr=1)
  	static public void newSkeleton() {
        skeleton = new Skeleton2D();
    }
  	
  	/**
  	 * Adds current ROI to skeleton
  	 */
  	@IJ1ScriptableMethod(target=CLIPPED_DOTS, ui="STD", pr=2)
  	static public void addRoiToSkeleton() {
          if (skeleton!=null) {
              Roi roi = workingImP.getRoi();
              addRoiToSkeleton(roi);
          }
    }
    
  	/**
  	 * Adds input roi to skeleton
  	 * @param roi
  	 */
  	static public void addRoiToSkeleton(Roi roi) {
  		 if (roi!=null) {
             FloatPolygon pol = roi.getFloatPolygon();
             int zpos=roi.getZPosition();      
             if ((workingImP.getNFrames()==1)||(workingImP.getNChannels()==1)) {
                 // Bug ? getZPosition returns always 0
                 zpos = workingImP.getSlice(); // +1 or -1 ?
             }
             PolygonSkeleton polSk = new PolygonSkeleton();
             ArrayList<Vector3D> vp = new ArrayList<>();
             float ZS=(float)opt.getOptParam("ZScale");
             for (int i=0;i<pol.npoints;i++) {
                 vp.add(new Vector3D(pol.xpoints[i],pol.ypoints[i],zpos*ZS));
             }
             if (pol.npoints>1) {
                 // closing polygon
                 vp.add(new Vector3D(pol.xpoints[0],pol.ypoints[0],zpos*ZS));
             }
             polSk.setDots(vp);
             skeleton.pols.add(polSk);
         }
  	}
  	
  	static public void addRoiToSkeleton(Roi roi, int zpos) {
 		 if (roi!=null) {
            FloatPolygon pol = roi.getFloatPolygon();
            PolygonSkeleton polSk = new PolygonSkeleton();
            ArrayList<Vector3D> vp = new ArrayList<>();
            float ZS=(float)opt.getOptParam("ZScale");
            for (int i=0;i<pol.npoints;i++) {
                vp.add(new Vector3D(pol.xpoints[i],pol.ypoints[i],zpos*ZS));
            }
            if (pol.npoints>1) {
                // closing polygon
                vp.add(new Vector3D(pol.xpoints[0],pol.ypoints[0],zpos*ZS));
            }
            polSk.setDots(vp);
            skeleton.pols.add(polSk);
        }
 	}
  	
  	/**
  	 * Dispatches surfels on skeleton with respect to d_0 value
  	 * And puts them in clipped dots
  	 */
    @IJ1ScriptableMethod(target=CLIPPED_DOTS, ui="STD", pr=3)
    static public void generateAndCopySkeletonSurface() {
          if ((skeleton!=null)&&(opt!=null)) {
              copiedDots=skeleton.getSurface(opt.d_0);
          } else {
              //IJ.error("generateAndCopySkeletonSurface: Skeleton or Optimizer (necessary to get r0 parameter) not initialized.");
          }        
    }
      /**
       * Makes a sphere with specified properties and puts it in clipped dots
       * @param px
       * @param py
       * @param pz
       * @param radius
       */
      @IJ1ScriptableMethod(target=CLIPPED_DOTS, ui="STD", tt="(double px, double py, double pz, double radius)", pr=0)
      static public void makeSphere(float px, float py, float pz, float radius) {
          copiedDots = makeSphere((float)opt.getOptParam("d_0"),px,py,(pz*(float)opt.getOptParam("ZScale")),radius);
      }
      
      /**
       * Makes a sheet of specified properties and puts it in clipped dots
       * @param pxi
       * @param pyi
       * @param pxf
       * @param pyf
       * @param pz
       */
      @IJ1ScriptableMethod(target=CLIPPED_DOTS)
      static public void makeXYSheet(float pxi, float pyi, float pxf, float pyf, float pz) {       
          copiedDots = makeXYSheet((float)opt.getOptParam("d_0"), (float)pxi, (float)pyi, (float)pxf, (float)pyf, (float)pz);
      }
    
     
      
    //------------------- I/O
    /**
     * Writes current data in XML/Ply files
     * @param path
     */
    //@IJ1ScriptableMethod(target=IO, ui="PathWriter", tt="(String path)", pr=1)
    static public void saveStateToXmlPlyv1(String path) {
        IOXmlPlyLimeSeg.saveState(new LimeSeg(), "0.1", path);
    }
    @IJ1ScriptableMethod(target=IO, ui="PathWriter", tt="(String path)", pr=1)
    static public void saveStateToXmlPly(String path) {
        IOXmlPlyLimeSeg.saveState(new LimeSeg(), "0.2", path);
    }
    /**
     * Loads current data in XML/Ply files
     * @param path
     */
    @IJ1ScriptableMethod(target=IO, ui="PathOpener", tt="(String path)", pr=0)
    static public void loadStateFromXmlPly(String path) {
    	    if (jcr!=null) {clear3DDisplay();}            
            IOXmlPlyLimeSeg.loadState(new LimeSeg(), path);      
            notifyCellRendererCellsModif=true;
            notifyCellExplorerCellsModif=true;
    }  
    /**
     * get currentChannel parameter (IJ1 macroextension style)    
     * @param value
     */
    @IJ1ScriptableMethod(target=OPT, tt="(String paramName, Double value)")
    static public void getCurrentChannel(Double[] value) {
        value[0]=(double) currentChannel;
    } 
    
    /**
     * get currentFrame parameter (IJ1 macroextension style)    
     * @param value
     */
    @IJ1ScriptableMethod(target=OPT, tt="(String paramName, Double value)")
    static public void getCurrentFrame(Double[] value) {
        value[0]=(double) currentFrame;
    }
    
    /**
     * get number of cells present in LimeSeg (IJ1 macroextension style)    
     * @param value
     */
    @IJ1ScriptableMethod(target=OPT, tt="(String paramName, Double value)")
    static public void getNCells(Double[] value) {
    	assert allCells!=null;
        value[0]=(double) allCells.size();
    } 
    /**
     * Gets 3D view center position (IJ1 macroextension style)
     * @param pX
     * @param pY
     * @param pZ
     */
    @IJ1ScriptableMethod(target=VIEW_3D, tt="(String paramName, Double value)")
    static public void get3DViewCenter(Double[] pX, Double[] pY, Double[] pZ) {        
        make3DViewVisible();
        pX[0]=(double)(jcr.lookAt.x);
        pY[0]=(double)(jcr.lookAt.y);
        pZ[0]=(double)(jcr.lookAt.z/(float)opt.getOptParam("ZScale"));
    }
    /**
     * Get current cell color (IJ1 macroextension style)
     * @param r
     * @param g
     * @param b
     * @param a
     */
    @IJ1ScriptableMethod(target=CURRENT_CELL, tt="(String paramName, Double value)")
    static public void getCellColor(Double[] r, Double[] g, Double[] b, Double[] a) {
	    if (currentCell!=null) {
	        r[0] = new Double(currentCell.color[0]);
	        g[0] = new Double(currentCell.color[1]);
	        b[0] = new Double(currentCell.color[2]);
	        a[0] = new Double(currentCell.color[3]);
    	} else {
	        r[0] = new Double(0);
	        g[0] = new Double(0);
	        b[0] = new Double(0);
	        a[0] = new Double(0);
    	}
    } 
    /**
     * See {@link #set3DViewMode(int)}
     * @param value
     */
    @IJ1ScriptableMethod(target=VIEW_3D, tt="(String paramName, Double value)")
    static public void get3DViewMode(Double[] value) {
        value[0]=(double) jcr.getViewMode();
    }
    /**
     * See {@link #setCell3DDisplayMode(int)}
     * @param vMode
     */
    @IJ1ScriptableMethod(target=CURRENT_CELL+TS+VIEW_3D, tt="(int vMode)")
    static public void getCell3DDisplayMode(Double[] vMode) {
		if (currentCell!=null) {
			vMode[0]=(double)currentCell.display_mode;
		}
	}    
    /**
     * Get current dot pos (IJ1 macro extension style)
     * @param px
     * @param py
     * @param pz
     */
    @IJ1ScriptableMethod(target=CURRENT_CELL, tt="(String paramName, Double value)")
    static public void getDotPos(Double[] px, Double[] py, Double[] pz) {
	    if (currentDot!=null) {
	        px[0] = new Double(currentDot.pos.x);
	        py[0] = new Double(currentDot.pos.y);
	        pz[0] = new Double(currentDot.pos.z);///(float)opt.getOptParam("ZScale"));
    	} else {
	        px[0] = new Double(0);
	        py[0] = new Double(0);
	        pz[0] = new Double(0);
    	}
    }
    /**
     * Get norm of current dot (IJ1 Macro extension style)
     * @param nx
     * @param ny
     * @param nz
     */
    @IJ1ScriptableMethod(target=CURRENT_CELL, tt="(String paramName, Double value)")
    static public void getDotNorm(Double[] nx, Double[] ny, Double[] nz) {
	    if (currentDot!=null) {
	        nx[0] = new Double(currentDot.Norm.x);
	        ny[0] = new Double(currentDot.Norm.y);
	        nz[0] = new Double(currentDot.Norm.z);///(float)opt.getOptParam("ZScale"));
    	} else {
	        nx[0] = new Double(0);
	        ny[0] = new Double(0);
	        nz[0] = new Double(0);
    	}
    }
    /**
     * Gets number of dots contained in current cell at current frame (IJ1 macro extension style)
     * @param value
     */
    @IJ1ScriptableMethod(target=CURRENT_CELLT, tt="(Double value)")
    static public void getNDots(Double[] value) {
		value[0]=(double)0;
    	if (currentCell!=null) {
    		if (currentCell.getCellTAt(currentFrame)!=null) {
    			if (currentCell.getCellTAt(currentFrame).dots!=null) {
    				value[0]=new Double(currentCell.getCellTAt(currentFrame).dots.size());
	   			}
    		}
    	}
    }
    /**
     * Get current cell at current frame center
     * @param px
     * @param py
     * @param pz
     */
    @IJ1ScriptableMethod(target=CURRENT_CELLT, tt="(Double[] px, Double[] py, Double[] pz)")
    static public void getCellCenter(Double[] px, Double[] py, Double[] pz) {
        px[0] = new Double(0);
        py[0] = new Double(0);
        pz[0] = new Double(0);
    	if (currentCell!=null) {
    		if (currentCell.getCellTAt(currentFrame)!=null) {
    			if (currentCell.getCellTAt(currentFrame).dots!=null) {
    				CellT ct = currentCell.getCellTAt(currentFrame);
    				ct.updateCenter();
    		        px[0] = new Double(ct.center.x);
    		        py[0] = new Double(ct.center.y);
    		        pz[0] = new Double(ct.center.z/(float)opt.getOptParam("ZScale"));
    			}
    		}
    	}
    }
    
    /**
     * Displays LimeSeg GUI
     */
	@IJ1ScriptableMethod
	public static void showGUI() {
		if (jfs3Di==null) {
            jfs3Di=new JFrameLimeSeg(new LimeSeg());
            jfs3Di.setVisible(true);
        } else {
            jfs3Di.setVisible(true);            
        } 
	}
	/**
	 * C Elegans segmentation benchmark
	 * !! Takes a huge amount of time to unzip data
	 */
	@IJ1ScriptableMethod(target=BENCHMARK, ui="STD", newThread=true) 
	static public void benchCElegans() {			
		SegCElegans.TestSegEmbryoCElegansFull();			
	}

    static public String handleExtension(String name, Object[] args) {  
    	HandleIJ1Extension.staticHandleExtension(name, args);
    	return null;
    }
    
    public DotN findDotNearTo(float px,float py,float pz) {
        DotN ans=null;        
        if (currentCell!=null) {
            CellT ct = currentCell.getCellTAt(currentFrame);
            if (ct!=null) {
                float minDist=Float.MAX_VALUE;
                Vector3D v = new Vector3D(px,py,pz);
                if (ct.dots.size()>0) {ans=ct.dots.get(0);}
                for (int i=1;i<ct.dots.size();i++) {
                    DotN dn = ct.dots.get(i);
                    if (Vector3D.dist2(dn.pos, v)<minDist) {
                        ans=dn;
                        minDist=Vector3D.dist2(dn.pos, v);
                    }                    
                }
            }
        }
        return ans;        
    }
    
    static Cell findCell(String id) {
        Cell ans=null;
        for (Cell c : allCells) {
            if (c.id_Cell.equals(id)) {ans=c;}
        }
        return ans;
    }
        
	@Override
	public void run() {
        if ((!extensionsHaveBeenRegistered)) {     	
        	HandleIJ1Extension.addAClass(this.getClass());        	
            Functions.registerExtensions(new HandleIJ1Extension());
        }
        if (allCells==null) {
            allCells=new ArrayList<>();
            notifyCellExplorerCellsModif=true;
            notifyCellRendererCellsModif=true;
        }
        if (dots_to_overlay==null) {
        	dots_to_overlay = new ArrayList<>();
        }
        if (opt==null) {
            initOptimizer(); 
        }
        if (workingImP==null) {
        	// Initialize the plugin with the current open image
        	workingImP = WindowManager.getCurrentImage();
        }                
        //=============================================     
        showGUI();
	}		
}