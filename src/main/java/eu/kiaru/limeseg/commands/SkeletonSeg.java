package eu.kiaru.limeseg.commands;


import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.util.ColorRGB;

import eu.kiaru.limeseg.LimeSeg;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.plugin.frame.RoiManager;
/**
 * Starts a segmentation from a skeleton:
 * 	- a skeleton is a list of roi, stored within the ROIManager.
 *  - this list should be ordered by slice and should be started and ended by a point
 * @author Nicolas Chiaruttini
 */
@Plugin(type = Command.class, menuPath = "Plugins>LimeSeg>Skeleton Seg")
public class SkeletonSeg implements Command{
	@Parameter(stepSize="0.1", min="0")
	float d_0=2.0f;
	
	@Parameter(stepSize="0.005", min="-0.04", max="0.04")
	float f_pressure = 0.015f;
	
	@Parameter
	float z_scale = 1f;

    @Parameter
    private ImagePlus imp;
    
    @Parameter
    float range_in_d0_units = 2;
       
    @Parameter
    ColorRGB color;
    
    boolean showOverlayOuput=true;
    
    @Parameter
    boolean show3D;
    
    boolean constructMesh=true;
    
    @Parameter
    int numberOfIntegrationStep=-1;
    
	boolean appendMeasures=true;
	    
	@Parameter
	float realXYPixelSize=1f;
    
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
        LimeSeg.clearOptimizer();
    	LimeSeg.opt.setOptParam("ZScale", z_scale);
        LimeSeg.opt.setOptParam("d_0",d_0);
        LimeSeg.opt.setOptParam("radiusSearch",d_0*range_in_d0_units);
        LimeSeg.opt.setOptParam("normalForce",f_pressure);
        float avgX=0;
        float avgY=0;
        float avgZ=0;
        int NRois=0;
        LimeSeg.newSkeleton();
        LimeSeg.currentChannel = imp.getChannel();
		for (Roi roi:roiManager.getRoisAsArray()) {
			LimeSeg.addRoiToSkeleton(roi,roi.getZPosition());	
			avgZ+=roi.getZPosition();
			avgX+=roi.getXBase();
			avgY+=roi.getYBase();
			LimeSeg.currentFrame = roi.getTPosition();
			if (LimeSeg.currentFrame==0) {LimeSeg.currentFrame=1;}
			NRois++;
		}
		LimeSeg.setWorkingImage(imp, LimeSeg.currentChannel, LimeSeg.currentFrame);
    	
		if (NRois<2) {
			System.err.println("At leasts two rois should be found in the manager.");
			return;
		}
		
		LimeSeg.generateAndCopySkeletonSurface();
		LimeSeg.newCell();
		LimeSeg.pasteDotsToCellT();
		LimeSeg.putCurrentCellTToOptimizer();
		LimeSeg.setCellColor((float) (color.getRed()/255.0),
			 	 (float) (color.getGreen()/255.0),
			 	 (float) (color.getBlue()/255.0),
			 	 1.0f);
		
		
    	if (show3D) {
    		LimeSeg.make3DViewVisible();
    		LimeSeg.putAllCellsTo3DDisplay();
    		LimeSeg.set3DViewCenter(avgX/NRois,avgY/NRois,avgZ/NRois);
    	}
    	
 	    float k_grad=(float) LimeSeg.opt.getOptParam("k_grad");
        LimeSeg.opt.setOptParam("k_grad",0.0f);
        LimeSeg.opt.setOptParam("normalForce",0);  
       	LimeSeg.opt.setCUDAContext();
       	double rmOutLiers = LimeSeg.opt.getOptParam("rmOutliers");
       	double attractToMax = LimeSeg.opt.getOptParam("attractToMax");
       	LimeSeg.setOptimizerParameter("rmOutliers", 0);
       	LimeSeg.runOptimisation(25);
       	LimeSeg.setOptimizerParameter("rmOutliers", rmOutLiers);
       	LimeSeg.runOptimisation(25);
       	LimeSeg.setOptimizerParameter("attractToMax", attractToMax);
       			
       	LimeSeg.opt.requestResetDotsConvergence=true;
       	LimeSeg.opt.setOptParam("k_grad",k_grad);
       	LimeSeg.opt.setOptParam("normalForce",f_pressure);
       	LimeSeg.runOptimisation(numberOfIntegrationStep); 	
       	if (constructMesh) {
       	   	LimeSeg.constructMesh();
       	   	if (show3D) {       	        		
       	   		LimeSeg.setCell3DDisplayMode(1);
       	   	}
       	}

       	LimeSeg.notifyCellExplorerCellsModif=true;
       	if (showOverlayOuput) {
       	   	LimeSeg.putCurrentCellToOverlay();       	        	
       	  	LimeSeg.updateOverlay();
       	}	
       	
    	if (appendMeasures) {
       		CommandHelper.displaySegmentationOutput(LimeSeg.opt.cellTInOptimizer, 
       												this.realXYPixelSize, 
       												this.constructMesh);
       	}
	}
}
