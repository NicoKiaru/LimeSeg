import eu.kiaru.limeseg.LimeSeg;
// Objects used by LimeSeg
import eu.kiaru.limeseg.struct.Cell;
import eu.kiaru.limeseg.struct.CellT;
import eu.kiaru.limeseg.struct.DotN;
import eu.kiaru.limeseg.struct.Vector3D;


for (Cell c:LimeSeg.allCells) {
	LimeSeg.currentCell=c;
	CellT ct = c.getCellTAt(1) // Get the Cell at first timepoint
	for (DotN dn:ct.dots) {
		print("P=\t"+dn.pos.x+"\t"+dn.pos.y+"\t"+dn.pos.z+"\t");
		print(" N=\t"+dn.Norm.x+"\t"+dn.Norm.y+"\t"+dn.Norm.z+"\t");
		// Needs to be computed beforehand, see example script ComputeCurvatures.groovy
		print(" MC=\t"+dn.meanCurvature+" GC=\t"+dn.gaussianCurvature+"\t");
		print(" Val="+dn.floatImageValue+"\t");
		print("\n");
	}
}