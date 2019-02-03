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
range = 4.5f;

for (Cell c:LimeSeg.allCells) {
	LimeSeg.currentCell=c;
	CellT ct = c.getCellTAt(1)
    CurvaturesComputer cc = new CurvaturesComputer(ct.dots,d0,range);
}


LimeSeg.jcr.colorSupplier = new DotNColorSupplier() {
	public float[] getColor(DotN dn) {
		gc=(float) dn.gaussianCurvature;
		mc=(float) dn.meanCurvature;
		//+mc*20f
		if (gc>0f) {
        	return [0.5f,(float)(0.5f+(Math.pow(1+gc,0.25f)-1)*1000f),0.5f,1f] as float[];
		}
		if (gc<0f) {
        	return [(float)(0.5f+(Math.pow(1-gc,0.25f)-1)*1000f),0.5f,0.5f,1f] as float[];
        }
    }
}
//LimeSeg.jcr.colorSupplier = new DefaultDotNColorSupplier();


/*LimeSeg.jcr.colorSupplier = new DotNColorSupplier() {
	public float[] getColor(DotN dn) {
		gc=(float) dn.gaussianCurvature;
		mc=(float) dn.meanCurvature;
		//+mc*20f
		if (mc>0f) {
        	return [(float)(0.5f+(Math.pow(1+mc,0.25)-1)*25f),0.5f,0.5f,1f] as float[];
		}
		if (mc<0f) {
        	return [0.5f,(float)(0.5f+(Math.pow(1-mc,0.25)-1)*25f),0.5f,1f] as float[];
        }
    }
}*/


LimeSeg.update3DDisplay();