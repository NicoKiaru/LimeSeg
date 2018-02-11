package eu.kiaru.limeseg.commands;


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
		for (Roi roi:roiManager.getRoisAsArray()) {
			LimeSeg.addRoiToSkeleton(roi,roi.getZPosition());	
			avgZ+=roi.getZPosition();
			avgX+=roi.getXBase();
			avgY+=roi.getYBase();
			LimeSeg.currentFrame = roi.getTPosition();
			LimeSeg.currentChannel = roi.getCPosition();
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
    		//LimeSeg.jcr.setViewMode(8);
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
       		//
       		ResultsTable rt = Analyzer.getResultsTable();
       	 
       		if (rt == null) {
       		        rt = new ResultsTable();
       		        Analyzer.setResultsTable(rt);
       		}
       		CellT ct = LimeSeg.currentCell.getCellTAt(LimeSeg.currentFrame);
       		{
       		    rt.incrementCounter();
       		    int i = rt.getLastColumn()+1;
       			ct.updateCenter();
       		    rt.addValue("Cell Name", ct.c.id_Cell);
       			rt.addValue("Number of Surfels", ct.dots.size());
       			rt.addValue("Center X", ct.center.x);
       			rt.addValue("Center Y", ct.center.y);
       			rt.addValue("Center Z", ct.center.z);
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
}
