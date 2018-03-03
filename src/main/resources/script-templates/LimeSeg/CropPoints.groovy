
import eu.kiaru.limeseg.LimeSeg;
// Objects used by LimeSeg
import eu.kiaru.limeseg.struct.Cell;
import eu.kiaru.limeseg.struct.CellT;
import eu.kiaru.limeseg.struct.DotN;
import eu.kiaru.limeseg.struct.Vector3D;

// One way of removing  points -> very inefficient in Groovy
/*for (Cell c:LimeSeg.allCells) {
	CellT ct = c.getCellTAt(1)
	ct.dots.removeAll { it.pos.x<2000 }
}
LimeSeg.update3DDisplay();*/


// This is a hundred times faster because it uses java's internal removeIf function
for (Cell c:LimeSeg.allCells) {
	LimeSeg.currentCell=c;
	CellT ct = c.getCellTAt(1)
	for (DotN dn:ct.dots) {
		if (dn.pos.x<2500) {
			dn.userDefinedFlag = true;
		}
	}
	// Calls fast Java removeIf function
	LimeSeg.removeFlaggedDots();
}
