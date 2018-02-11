package eu.kiaru.limeseg.commands;

import java.util.ArrayList;

import org.scijava.app.StatusService;
import org.scijava.command.Command;
import org.scijava.io.DefaultIOService;
import org.scijava.io.IOService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.service.ServiceHelper;
import org.scijava.util.ColorRGB;

import eu.kiaru.limeseg.LimeSeg;
import eu.kiaru.limeseg.demos.DemoHelper;
import eu.kiaru.limeseg.struct.CellT;
import ij.ImagePlus;
import ij.gui.OvalRoi;
import ij.gui.Roi;
import ij.measure.ResultsTable;
import ij.plugin.filter.Analyzer;
import ij.plugin.frame.RoiManager;
import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.display.ImageDisplay;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.display.imagej.ImageJFunctions;
/**
 * Sphere seg with many options
 * @author Nicolas Chiaruttini
 *
 */
@Plugin(type = Command.class, menuPath = "Plugins>LimeSeg>Sphere Seg (Advanced)")
public class SphereSegAdvanced implements Command {
	
	@Parameter(persist=true, stepSize="0.1", min="0")
	float d_0=2.0f;
	
	@Parameter(persist=true, stepSize="0.005", min="-0.04", max="0.04")
	float f_pressure = 0.015f;
	
	@Parameter(persist=true)
	float z_scale = 1f;

    @Parameter(persist=true)
    private ImagePlus imp;
    
    @Parameter(persist=true)
    float range_in_d0_units = 2;
    
    static int index=0;
    
    @Parameter(persist=true)
    ColorRGB color;
    
    @Parameter(persist=true)
    boolean sameCell;
    
    @Parameter(persist=true)
    boolean showOverlayOuput;
    
    @Parameter(persist=true)
    boolean show3D;
    
    @Parameter(persist=true)
    boolean constructMesh;
    
    @Parameter(persist=true)
    int numberOfIntegrationStep=-1;
    
    @Parameter(persist=true)
    boolean randomColors;
    
    @Parameter(persist=true)
    boolean appendMeasures;
    
    @Parameter(persist=true)
    float realXYPixelSize=1f;
    
	@Parameter
	StatusService sts;
	
	@Parameter(persist=true)
	boolean clearOptimizer;
	
	@Parameter(persist=true)
	boolean stallDotsPreviouslyInOptimizer=false;
    
	@Override
	public void run() {
        RoiManager roiManager = RoiManager.getRoiManager();
        if (roiManager==null) {
        	System.err.println("No roi manager found - command aborted.");
        	return;
        }
		LimeSeg lms = new LimeSeg();
        lms.initialize();
		LimeSeg.saveOptState();
        if (clearOptimizer) {LimeSeg.clearOptimizer();}
        if ((!clearOptimizer)) {
        	if ((stallDotsPreviouslyInOptimizer)) {
        		
	        	LimeSeg.opt.dots.forEach(dn -> {
	        		dn.stallDot();
	        	});
        	} else {
        		LimeSeg.opt.dots.forEach(dn -> {
        			dn.freeDot();
            	});
        	}
        }
    	LimeSeg.opt.setOptParam("ZScale", z_scale);
        LimeSeg.opt.setOptParam("d_0",d_0);
        LimeSeg.opt.setOptParam("radiusSearch",d_0*range_in_d0_units);
        LimeSeg.opt.setOptParam("normalForce",f_pressure);

        float avgX=0;
        float avgY=0;
        float avgZ=0;
        int NCells=0;
        ArrayList<CellT> currentlyOptimizedCellTs = new ArrayList<>();
        if (sameCell) {
        	LimeSeg.newCell();
        }
        
		for (Roi roi:roiManager.getRoisAsArray()) {
			if (roi.getClass().equals(OvalRoi.class)) {
				OvalRoi circle = (OvalRoi) roi;
				float r0 = (float) ((circle.getFloatWidth()/2 + circle.getFloatHeight()/2)/2);
				LimeSeg.currentChannel = circle.getCPosition();
				LimeSeg.currentFrame = circle.getTPosition();
				if (LimeSeg.currentFrame==0) {LimeSeg.currentFrame=1;}
				float z0 = circle.getZPosition();
				float x0 = (float)(circle.getXBase()+circle.getFloatWidth()/2);
				float y0 = (float)(circle.getYBase()+circle.getFloatHeight()/2);
				
				avgX+=x0;
				avgY+=y0;
				avgZ+=z0;
				NCells++;

		    	if (!sameCell) {
		    		LimeSeg.newCell();
		    	}
		        if (!this.randomColors) {
		        	LimeSeg.setCellColor((float) (color.getRed()/255.0),
		        					 	 (float) (color.getGreen()/255.0),
		        					 	 (float) (color.getBlue()/255.0),
		        					 	 1.0f);
		        } else {
		        	LimeSeg.setCellColor((float) (java.lang.Math.random()),
    					 				 (float) (java.lang.Math.random()),
    					 				 (float) (java.lang.Math.random()),
    					 				 1.0f);
		        }		       
		        LimeSeg.makeSphere(x0,y0,z0,r0);      
		        LimeSeg.pasteDotsToCellT();
		        if (!sameCell) {
		        	LimeSeg.putCurrentCellTToOptimizer();
		        }
		    	currentlyOptimizedCellTs.add(LimeSeg.currentCell.getCellTAt(LimeSeg.currentFrame));
			}			
		}
		LimeSeg.setWorkingImage(imp, LimeSeg.currentChannel, LimeSeg.currentFrame);
    	
		if (sameCell) {
			LimeSeg.putCurrentCellTToOptimizer();
		}
		
    	if (show3D) {
    		LimeSeg.make3DViewVisible();
    		LimeSeg.putAllCellsTo3DDisplay();
    		LimeSeg.set3DViewCenter(avgX/NCells,avgY/NCells,avgZ/NCells);
    		//LimeSeg.jcr.setViewMode(8+LimeSeg.jcr.getViewMode()%8);
    	}
 	    float k_grad=(float) LimeSeg.opt.getOptParam("k_grad");
        LimeSeg.opt.setOptParam("k_grad",0.0f);
        LimeSeg.opt.setOptParam("normalForce",0);        
       	//LimeSeg.opt.setCUDAContext();
       	LimeSeg.runOptimisation(500);
        LimeSeg.opt.requestResetDotsConvergence=true;
        LimeSeg.opt.setOptParam("k_grad",k_grad);
        LimeSeg.opt.setOptParam("normalForce",f_pressure);
        LimeSeg.runOptimisation(numberOfIntegrationStep); 	
       	if (constructMesh) {
       	   	for (CellT ct : currentlyOptimizedCellTs) {
       	   		ct.constructMesh();
       	   	}
       	   	//LimeSeg.constructMesh();
       	   	if (show3D) {
       	   		for (CellT ct : currentlyOptimizedCellTs) {
       	   			LimeSeg.setCell3DDisplayMode(1);
               		LimeSeg.currentCell=ct.c;
   	    	 	}
       	   	}
       	}

       	LimeSeg.notifyCellExplorerCellsModif=true;
       	if (showOverlayOuput) {
       	   	for (CellT ct : currentlyOptimizedCellTs) {
       	   		LimeSeg.addToOverlay(ct);
       	   	}
       	   	LimeSeg.updateOverlay();
       	}
       	
       	if (appendMeasures) {
       		//
       		ResultsTable rt = Analyzer.getResultsTable();
       	 
       		if (rt == null) {
       		        rt = new ResultsTable();
       		        Analyzer.setResultsTable(rt);
       		}
       		for (CellT ct : currentlyOptimizedCellTs) {
       		    rt.incrementCounter();
       		    int i = rt.getLastColumn()+1;
       			ct.updateCenter();
       		    rt.addValue("Cell Name", ct.c.id_Cell);
       			rt.addValue("Number of Surfels", ct.dots.size());
       			rt.addValue("Center X", ct.center.x);
       			rt.addValue("Center Y", ct.center.y);
       			rt.addValue("Center Z", (ct.center.z/LimeSeg.opt.getOptParam("ZScale"))+1);
       			rt.addValue("Frame", ct.frame);
       			rt.addValue("Channel", ct.c.cellChannel);       			
       			rt.addValue("Mesh ?", (constructMesh)?"YES":"NO");
       			if (constructMesh) {
       				rt.addValue("Euler characteristic", ct.dots.size()-3.0/2.0*ct.triangles.size()+ct.triangles.size());
       				rt.addValue("Free edges", ct.freeEdges);
       				rt.addValue("Surface", ct.getSurface());
       				rt.addValue("Volume", ct.getVolume());
       				rt.addValue("Real Surface", ct.getSurface()*(this.realXYPixelSize*this.realXYPixelSize));
       				rt.addValue("Real Volume", ct.getVolume()*(this.realXYPixelSize*this.realXYPixelSize*this.realXYPixelSize));
       			}
       		}       		
       		rt.show("Results");
       	}
       
	}
	
	/**
     * This main function serves for development purposes.
     *
     * @param args whatever, it's ignored
     * @throws Exception
     */
    public static void main(final String... args) throws Exception {
		final ImageJ ij = new ImageJ();
	    ij.ui().showUI();
	    final ServiceHelper sh = new ServiceHelper(ij.getContext());
        final IOService io = sh.loadService(DefaultIOService.class);
        final Dataset datasetIn = DemoHelper.getDatasetFromResources(io,"images/Dub-WilliamMohler-Tp33-Half.zip");
        final ImageDisplay imageDisplay =
                (ImageDisplay) ij.display().createDisplay(datasetIn);

        ImagePlus myImpPlus = ImageJFunctions.wrap((RandomAccessibleInterval)datasetIn,"CElegans");
        RoiManager rM = RoiManager.getRoiManager();
        OvalRoi circle = new OvalRoi(321,112,20,20);
        circle.setPosition(1, 13, 1);
        rM.add(myImpPlus, circle, 13);
        
        circle = new OvalRoi(121,112,20,20);
        circle.setPosition(1, 13, 1);
        rM.add(myImpPlus, circle, 13);
        
        ij.command().run(SphereSegAdvanced.class, true);
    }

}
