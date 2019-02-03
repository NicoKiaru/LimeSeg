import eu.kiaru.limeseg.LimeSeg;
// Objects used by LimeSeg
import eu.kiaru.limeseg.struct.Cell;
import eu.kiaru.limeseg.struct.CellT;
import eu.kiaru.limeseg.struct.DotN;
import eu.kiaru.limeseg.struct.Vector3D;
import eu.kiaru.limeseg.opt.CurvaturesComputer;
import eu.kiaru.limeseg.gui.DotNColorSupplier;
import eu.kiaru.limeseg.gui.DefaultDotNColorSupplier;

d0=(float) LimeSeg.opt.getOptParam("d_0");

LimeSeg.jcr.colorSupplier = new GaussianCurvatureColorLUT(1000f);
//LimeSeg.jcr.colorSupplier = new MeanCurvatureColorLUT(80f);
//LimeSeg.jcr.colorSupplier = new DefaultDotNColorSupplier();

LimeSeg.update3DDisplay();


// ---------------------- Classes for DotN coloring, based on any DotN property (~ vertex shader)

/**
 * Look Up Table for Gaussian Curvature:
 *  parameter emphasize to emphasize small curvature value
 *  atan function used to map infinity value to 1
 */
 
public class GaussianCurvatureColorLUT extends DotNColorSupplier {

	float emphasize; 
	
	public GaussianCurvatureColorLUT(float emphasize) {
		this.emphasize  = emphasize;		
	}

	public float[] getColor(DotN dn) {
		float curvature=(float) dn.gaussianCurvature;
		if (curvature>0f) {
        	return [0.5f,(float)(0.5f*(1+(Math.atan(emphasize*curvature)*2/Math.PI))),0.5f,1f] as float[];
		}
		if (curvature<0f) {
        	return [(float)(0.5f*(1+(Math.atan(-emphasize*curvature)*2/Math.PI))),0.5f,0.5f,1f] as float[];
        }
    }
	
}

/**
 * Look Up Table for Mean Curvature:
 *  parameter emphasize to emphasize small curvature value
 *  atan function used to map infinity value to 1
 */
 
public class MeanCurvatureColorLUT extends DotNColorSupplier {

	float emphasize; 
	
	public MeanCurvatureColorLUT(float emphasize) {
		this.emphasize  = emphasize;		
	}

	public float[] getColor(DotN dn) {
		float curvature=(float) dn.meanCurvature;
		if (curvature<0f) {
        	return [0.5f,(float)(0.5f*(1+(Math.atan(-emphasize*curvature)*2/Math.PI))),0.5f,1f] as float[];
		}
		if (curvature>0f) {
        	return [(float)(0.5f*(1+(Math.atan(emphasize*curvature)*2/Math.PI))),0.5f,0.5f,1f] as float[];
        }
    }
	
}