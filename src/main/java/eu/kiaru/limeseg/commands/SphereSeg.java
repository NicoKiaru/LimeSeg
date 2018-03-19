package eu.kiaru.limeseg.commands;

import java.util.concurrent.ExecutionException;

import org.scijava.Initializable;
import org.scijava.command.Command;
import org.scijava.command.DynamicCommand;
import org.scijava.command.CommandService;
import org.scijava.module.MutableModuleItem;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.util.ColorRGB;

import ij.ImagePlus;
import net.imagej.ImageJ;

@Plugin(type = Command.class, menuPath = "Plugins>LimeSeg>Sphere Seg")
public class SphereSeg extends DynamicCommand implements Initializable{

	@Parameter
	CommandService cs;
	
	@Parameter(persist=true, stepSize="0.1", min="0")
	float d_0=2.0f;
	
	@Parameter(persist=true, stepSize="0.005", min="-0.04", max="0.04")
	float f_pressure = 0.015f;
	
	@Parameter(persist=false)//true)
	double z_scale;

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
	
    @Override
    public void initialize() {
    /*	System.out.println("coucou! j'initiliaeipot ="+imp.getCalibration().pixelDepth);
		final MutableModuleItem<Double> zScaleItem=
				getInfo().getMutableInput("z_scale", Double.class);
			zScaleItem.setDefaultValue(imp.getCalibration().pixelDepth);
			zScaleItem.setDefaultValue(new Double(3.5));*/
			
    	
    }
    
    
	public void run() {
		System.out.println("zs="+z_scale);
        try {
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
								       ).get();
		} catch (InterruptedException | ExecutionException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}
	
	
	
    public static void main(final String... args) throws Exception {
	    // Creates an IJ instance
	    final ImageJ ij = new ImageJ();
	    ij.ui().showUI();
	    // Code of an IJ1 Macro
	    // Executes the IJ1 Macro and waits to finish (get())
	    ij.script().run("foo.ijm","run('Bat Cochlea Volume (19K)');", true).get();        
	    String setPropVox="run('Properties...', 'channels=1 slices=114 frames=1 unit=pixel pixel_width=1.0000 pixel_height=1.0000 voxel_depth=3.23');";
	    ij.script().run("foo.ijm",setPropVox, true).get(); 
	     
	    // Launch the command which requires an open image
	    ij.command().run(SphereSeg.class, true);
    }

}
