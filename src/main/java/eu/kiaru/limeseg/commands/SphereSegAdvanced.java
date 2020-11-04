package eu.kiaru.limeseg.commands;

import java.util.ArrayList;

import ij.IJ;
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
	double z_scale = 1f;

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
    boolean showOverlayOuput = true;
    
    @Parameter(persist=true)
    boolean show3D = true;
    
    @Parameter(persist=true)
    boolean constructMesh=true;
    
    @Parameter(persist=true)
    int numberOfIntegrationStep=-1;
    
    @Parameter(persist=true)
    boolean randomColors;
    
    @Parameter(persist=true)
    boolean appendMeasures;
    
    @Parameter(persist=true)
    float realXYPixelSize=1f;
	
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
    	LimeSeg.opt.setOptParam("ZScale", (float)z_scale);
        LimeSeg.opt.setOptParam("d_0",d_0);
        LimeSeg.opt.setOptParam("radiusSearch",d_0*range_in_d0_units);
        LimeSeg.opt.setOptParam("normalForce",f_pressure);

        float avgX=0;
        float avgY=0;
        float avgZ=0;
        int NCells=0;
        ArrayList<CellT> currentlyOptimizedCellTs = new ArrayList<>();
		
		if (imp == null) imp = IJ.openImage(); // TODO : not sure about this
		
        LimeSeg.currentChannel = imp.getChannel();
        if (sameCell) {
        	LimeSeg.newCell();
        }
        int nRois=roiManager.getRoisAsArray().length;
		for (Roi roi:roiManager.getRoisAsArray()) {
			if (roi.getClass().equals(OvalRoi.class)) {
				OvalRoi circle = (OvalRoi) roi;
				float r0 = (float) ((circle.getFloatWidth()/2 + circle.getFloatHeight()/2)/2);
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
		        if ((this.sameCell)||(nRois==1)) {
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
    	}
 	    float k_grad=(float) LimeSeg.opt.getOptParam("k_grad");
        LimeSeg.opt.setOptParam("k_grad",0.0f);
        LimeSeg.opt.setOptParam("normalForce",0);        
       	//LimeSeg.opt.setCUDAContext();
       	LimeSeg.runOptimisation(500);
        LimeSeg.opt.requestResetDotsConvergence=true;
        LimeSeg.opt.setOptParam("k_grad",k_grad);
        LimeSeg.opt.setOptParam("normalForce",f_pressure);
        boolean RadiusDeltaChanged=false;
        if ((LimeSeg.opt.cellTInOptimizer.size()>1)&&(LimeSeg.opt.getOptParam("radiusDelta")==0)) {
        	RadiusDeltaChanged=true;
        	LimeSeg.opt.setOptParam("radiusDelta", d_0/2);
        }
        LimeSeg.runOptimisation(numberOfIntegrationStep);
        if (RadiusDeltaChanged) {
        	LimeSeg.opt.setOptParam("radiusDelta", 0);
        }
       	if (constructMesh) {
       	   	for (CellT ct : currentlyOptimizedCellTs) {
       	   		ct.constructMesh();
       	   	}
       	   	for (CellT ct : currentlyOptimizedCellTs) {
       	   		LimeSeg.setCell3DDisplayMode(1);
            	LimeSeg.currentCell=ct.c;
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
       		CommandHelper.displaySegmentationOutput(currentlyOptimizedCellTs, 
       												this.realXYPixelSize, 
       												this.constructMesh);
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
