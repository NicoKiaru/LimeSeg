#@String(choices={"Surfels", "Mesh"}) cellVisualisationMode 
#@String(choices={"Not clipped", "Cut above", "Cut below", "Cut above and below"}) clippingMode

//---- Script helping settings 3D vizualisation settings for all cells
//---- Note that no mesh will be shown if the mesh has not been constructed

import eu.kiaru.limeseg.LimeSeg;
// Objects used by LimeSeg
import eu.kiaru.limeseg.struct.Cell;
import eu.kiaru.limeseg.struct.CellT;
import eu.kiaru.limeseg.struct.DotN;
import eu.kiaru.limeseg.struct.Vector3D;
import eu.kiaru.limeseg.opt.CurvaturesComputer;
import eu.kiaru.limeseg.gui.DotNColorSupplier;
import eu.kiaru.limeseg.gui.DefaultDotNColorSupplier;

switch(clippingMode) {
	case "Not clipped": LimeSeg.set3DViewMode(0);break;
	case "Cut above": LimeSeg.set3DViewMode(1);break;
	case "Cut below": LimeSeg.set3DViewMode(2);break;
	case "Cut above and below": LimeSeg.set3DViewMode(3);break;
}


switch(cellVisualisationMode ) {
	case "Surfels": setAllCellsDisplayMode(0);break;
	case "Mesh": setAllCellsDisplayMode(1);break;
}


LimeSeg.update3DDisplay();


//------ Helper function

void setAllCellsDisplayMode(int mode) {
	for (Cell c:LimeSeg.allCells) {
		LimeSeg.currentCell=c;
		LimeSeg.setCell3DDisplayMode(mode);
	}	
}