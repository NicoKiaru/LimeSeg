package eu.kiaru.limeseg.commands;

import org.scijava.command.Command;
import org.scijava.io.DefaultIOService;
import org.scijava.io.IOService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.service.ServiceHelper;

import eu.kiaru.limeseg.LimeSeg;
import eu.kiaru.limeseg.demos.DemoHelper;
import eu.kiaru.limeseg.struct.CellT;
import ij.ImagePlus;
import ij.gui.OvalRoi;
import ij.plugin.frame.RoiManager;
import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.display.ImageDisplay;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.display.imagej.ImageJFunctions;
/**
 * Smooth change of d_0 of the objects already present in the optimizer
 * @author Nicolas Chiaruttini
 */
@Plugin(type = Command.class, menuPath = "Plugins>LimeSeg>Coarsen/Refine Seg")
public class CoarsenRefineSegmentation implements Command{
	
	@Parameter(persist=false, initializer = "inid0Ini")
	float d_0_Ini;

	@Parameter(persist=true)
	float d_0_End;
	
	//@Parameter(persist=false, initializer = "iniFPressure")
	float f_pressure = 0.0f;
	
    @Parameter(persist=true)
    float range_in_d0_units = 2;
	
	private void inid0Ini() {
		if (LimeSeg.opt!=null) {
			d_0_Ini = (float) LimeSeg.opt.getOptParam("d_0");
		}		
	}
	
	@Override
	public void run() {
		if ((d_0_Ini!=d_0_End)&&(!LimeSeg.optimizerIsRunning)&&(d_0_End>0)) {
			// removes the mesh if it was existing, and notifies it to the 3D display
			for (CellT ct : LimeSeg.opt.cellTInOptimizer) {
       	   		ct.tesselated=false;
       	   		if (ct.c.display_mode==1) {ct.c.display_mode=0;}
       	   		LimeSeg.notifyCellRendererCellsModif=true;
       	   	}
			LimeSeg.setOptimizerParameter("normalForce", 0);//f_pressure = 0 during refinement
			LimeSeg.setOptimizerParameter("d_0", d_0_Ini);
			LimeSeg.opt.setOptParam("radiusSearch",d_0_Ini*range_in_d0_units);
			LimeSeg.saveOptState();			
			double currentd0=d_0_Ini;
			LimeSeg.opt.setCUDAContext();
		    LimeSeg.optimizerIsRunning = true;
		    // logarithmic decay
		    double maxRatioIncreaseBetweenSteps = 1.0075;
		    double maxRatioDecreaseBetweenSteps = 1/maxRatioIncreaseBetweenSteps;
			while (currentd0!=d_0_End) {
				if (currentd0>d_0_End) {
					currentd0=currentd0*maxRatioDecreaseBetweenSteps;
					if (currentd0<d_0_End) {
						currentd0=d_0_End;
					}
				} else {
					currentd0=currentd0*maxRatioIncreaseBetweenSteps;
					if (currentd0>d_0_End) {
						currentd0=d_0_End;
					}
				}
				LimeSeg.opt.requestResetDotsConvergence=true;
				LimeSeg.opt.setOptParam("radiusSearch",(float)(currentd0*range_in_d0_units));
				LimeSeg.opt.setOptParam("d_0",(float)(currentd0));
				LimeSeg.opt.nextStep();
				LimeSeg.notifyCellRendererCellsModif=true;
				if ((LimeSeg.requestStopOptimisation==true)||(LimeSeg.opt.dots.size()==0)) {
					break;
				}
			}
			LimeSeg.opt.setOptParam("d_0",(float)(d_0_End));
			LimeSeg.optimizerIsRunning = false;
			LimeSeg.requestStopOptimisation = false;
			LimeSeg.opt.freeGPUMem();
		}

       	LimeSeg.notifyCellExplorerCellsModif=true;
	}
	
}
