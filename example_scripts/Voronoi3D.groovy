// Demo of LimeSeg use -> building of a voronoi partitioning
// Show how to launch LimeSeg optimization without commands

#@ImageJ ij

import eu.kiaru.limeseg.LimeSeg;
import eu.kiaru.limeseg.commands.SphereSeg;
import eu.kiaru.limeseg.commands.ClearAll;

// Objects used by LimeSeg
import eu.kiaru.limeseg.struct.Cell;
import eu.kiaru.limeseg.struct.CellT;
import eu.kiaru.limeseg.struct.DotN;
import eu.kiaru.limeseg.struct.Vector3D;

import org.scijava.util.ColorRGB;
import ij.IJ // imports IJ1
import ij.plugin.frame.RoiManager // imports ROIManager
import ij.gui.OvalRoi
import ij.gui.Roi
import ij.ImagePlus;


// Initializes LimeSeg and displays LimeSeg GUI
(new LimeSeg()).initialize();
//LimeSeg.showGUI();  displays the GUI
// Erases all objects previously in LimeSeg
LimeSeg.clearAllCells();
LimeSeg.clearOptimizer(); 

// Creates a dummy image
IJ.runMacro("newImage('Black', '16-bit black', 300, 300, 300);");
IJ.runMacro("doCommand('Start Animation [\\\\]');");

LimeSeg.setWorkingImage("Black");
LimeSeg.setZScale(1); // isotropic
LimeSeg.currentFrame=1;
LimeSeg.currentChannel=1;


float boxSize = 40;
float r0=boxSize/2;
// Sets d_0 parameter (~surface precision, equilibrium distance between surfels)
LimeSeg.setOptimizerParameter("d_0",boxSize/8);

for (float x=boxSize;x<(300-boxSize);x+=boxSize) {
	for (float y=boxSize;y<(300-boxSize);y+=boxSize) {
		for (float z=boxSize;z<(300-boxSize);z+=boxSize) {
			// Creates Cell object
			if (java.lang.Math.random()<0.25) {
				LimeSeg.newCell(); // stored in LimeSeg.currentCell
				Cell myCell = LimeSeg.currentCell;
				myCell.color[0]=x/300.0;
				myCell.color[1]=y/300.0;
				myCell.color[2]=z/300.0;
				myCell.color[3]=1f;
				LimeSeg.makeSphere(x,y,z,r0);
				LimeSeg.pasteDotsToCellT();
				LimeSeg.putCurrentCellTToOptimizer();
			}
		}
	}
}

LimeSeg.make3DViewVisible();
LimeSeg.set3DViewMode(0);
LimeSeg.clear3DDisplay();
LimeSeg.putAllCellsTo3DDisplay();

LimeSeg.jcr.lookAt.x=150;
LimeSeg.jcr.lookAt.y=150;
LimeSeg.jcr.lookAt.z=150;

// First round of optimization to "relax" spheres
LimeSeg.setOptimizerParameter("normalForce",0.00); // == f_pressure
LimeSeg.setOptimizerParameter("attractToMax",0); // disable image link
LimeSeg.runOptimisation(1000);

LimeSeg.opt.requestResetDotsConvergence=true; // Reset the convergence -> optimization can start again
LimeSeg.setOptimizerParameter("normalForce",0.03); // == f_pressure
LimeSeg.setOptimizerParameter("attractToMax",0); // disable image link
LimeSeg.runOptimisation(2500);
LimeSeg.setOptimizerParameter("attractToMax",1); // Restores link to image for future usage...

LimeSeg.currentFrame=1; // needs to be modified to work on a different timepoint -> irrelevant here
for (Cell c:LimeSeg.allCells) {
	LimeSeg.currentCell=c;
	LimeSeg.constructMesh(); // reconstruct mesh
	
}

LimeSeg.clear3DDisplay();

for (Cell c:LimeSeg.allCells) {
	LimeSeg.setCell3DDisplayMode(1); // displays Mesh instead of dots 
}

LimeSeg.putAllCellsTo3DDisplay();

// LimeSeg cellT points manipulation
// For instance due to the methods, there is some space between surface, let's make this less obvious:

float expansion = LimeSeg.opt.getOptParam("d_0")/2;

for (Cell c:LimeSeg.allCells) { // loops through cells
	LimeSeg.currentCell=c;
	// Fetch CellT object = single 3D object, rather obvious here because there is only one timepoint
	CellT ct = c.getCellTAt(LimeSeg.currentFrame);
	for (DotN dn: ct.dots) { // DotN object = surfel = a position and a normal vector
		dn.pos.x+=dn.Norm.x*expansion;
		dn.pos.y+=dn.Norm.y*expansion;
		dn.pos.z+=dn.Norm.z*expansion;
	}
}
LimeSeg.update3DDisplay(); // notifies 3D viewer that the dots have changed

//2D output
LimeSeg.clearOverlay();
LimeSeg.updateOverlay();
LimeSeg.addAllCellsToOverlay();
LimeSeg.updateOverlay();

// Fun with the 3D viewer
// Set mode for cropping in Z:
LimeSeg.set3DViewMode(1); // Check 0 1 or 2
// The animation was launched at the beginning of this macro... and 2D and 3D synchronization is responsible for the live cropping
for (int i=0;i<400;i++) {
	LimeSeg.jcr.view_rotx+=1/50;
	LimeSeg.jcr.view_roty+=1/25;
	LimeSeg.jcr.view_rotz+=1/100;
	sleep(50);
}

// Getting data out:
for (Cell c:LimeSeg.allCells) { // loops through cells
	CellT my3Dobject = c.getCellTAt(LimeSeg.currentFrame);
	if (my3Dobject.freeEdges!=0) {
		print("Mesh reconstruction problem, wrong surface and volume value!");
		print("Free edges = "+my3Dobject.freeEdges+"\n"); 
	}
	print("Cell    : "+c.id_Cell+"\n");
	print("Surface = "+my3Dobject.getSurface()+"\n");
	print("Volume  = "+ my3Dobject.getVolume()+"\n");
}

