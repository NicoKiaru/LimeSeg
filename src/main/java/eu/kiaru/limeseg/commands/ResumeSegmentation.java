package eu.kiaru.limeseg.commands;

import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import eu.kiaru.limeseg.LimeSeg;
import eu.kiaru.limeseg.struct.CellT;
import ij.plugin.frame.RoiManager;
/**
 * Resumes a segmentation started within the optimizer
 * @author Nicolas Chiaruttini
 * 
 */
@Plugin(type = Command.class, menuPath = "Plugins>LimeSeg>Resume Seg")
public class ResumeSegmentation implements Command {
	
	@Parameter(persist=true, stepSize="0.005", min="-0.04", max="0.04")
	float f_pressure = 0.015f;
    
    @Parameter(persist=true)
    float range_in_d0_units = 2;
    
    boolean showOverlayOuput=true;
    
    @Parameter(persist=true)
    boolean show3D;
    
    @Parameter(persist=true)
    boolean resetConvergence;
    
    boolean constructMesh=true;
    
    @Parameter(persist=true)
    int numberOfIntegrationStep=-1;
    
    boolean appendMeasures=true;
    
    @Parameter(persist=true)
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
		float d_0 = (float) LimeSeg.opt.getOptParam("d_0");
        LimeSeg.opt.setOptParam("radiusSearch",d_0*range_in_d0_units);
        LimeSeg.opt.setOptParam("normalForce",f_pressure);

    	if (show3D) {
    		LimeSeg.make3DViewVisible();
    		LimeSeg.putAllCellsTo3DDisplay();
    	}
    	
        if (resetConvergence) {
        	LimeSeg.opt.requestResetDotsConvergence=true;
        }
        LimeSeg.runOptimisation(numberOfIntegrationStep); 	
       	if (constructMesh) {
       	   	for (CellT ct : LimeSeg.opt.cellTInOptimizer) {
       	   		ct.constructMesh();
       	   	}
       	   	for (CellT ct : LimeSeg.opt.cellTInOptimizer) {
       	   		LimeSeg.setCell3DDisplayMode(1);
            	LimeSeg.currentCell=ct.c;
   	    	}
       	}

       	LimeSeg.notifyCellExplorerCellsModif=true;
       	if (showOverlayOuput) {
       	   	for (CellT ct : LimeSeg.opt.cellTInOptimizer) {
       	   		LimeSeg.addToOverlay(ct);
       	   	}
       	   	LimeSeg.updateOverlay();
       	}
       	
    	if (appendMeasures) {
       		CommandHelper.displaySegmentationOutput(LimeSeg.opt.cellTInOptimizer, 
       												this.realXYPixelSize, 
       												this.constructMesh);
       	}
	}
}
