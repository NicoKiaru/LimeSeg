package eu.kiaru.limeseg.commands;

import org.scijava.command.Command;
import org.scijava.command.CommandService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.util.ColorRGB;

import ij.ImagePlus;

@Plugin(type = Command.class, menuPath = "Plugins>LimeSeg>Sphere Seg")
public class SphereSeg implements Command{

	@Parameter
	CommandService cs;
	
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
    
    @Parameter(persist=true)
    ColorRGB color;
    
    @Parameter(persist=true)
    boolean sameCell;
    
    @Parameter(persist=true)
    boolean show3D;
    
    @Parameter(persist=true)
    int numberOfIntegrationStep=-1;
    
    @Parameter(persist=true)
    float realXYPixelSize=1f;	
	
	public void run() {
        cs.run(SphereSegAdvanced.class,true,
        		// Fixed parameters for "simple" version
							           "showOverlayOuput",true,
							           "constructMesh",true, 
							           "clearOptimizer",true,
							           "appendMeasures",true,
							           "randomColors",true,
							           "stallDotsPreviouslyInOptimizer",false,
				// Communicated parameters (not necessary for GUI but for transparent scriptability)
							           "d_0",d_0,
							           "realXYPixelSize",realXYPixelSize,
							           "numberOfIntegrationStep",numberOfIntegrationStep,
							           "show3D",show3D,
							           "sameCell",sameCell,
							           "color",color,
							           "range_in_d0_units",range_in_d0_units,
							           "imp",imp,
							           "z_scale",z_scale,
							           "f_pressure",f_pressure
							       );		
	}

}
