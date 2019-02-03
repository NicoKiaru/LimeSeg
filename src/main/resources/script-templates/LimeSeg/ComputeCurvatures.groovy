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
range = 3f;

for (Cell c:LimeSeg.allCells) {
	LimeSeg.currentCell=c;
	CellT ct = c.getCellTAt(1)
    CurvaturesComputer cc = new CurvaturesComputer(ct.dots,d0,range);
}

LimeSeg.update3DDisplay();