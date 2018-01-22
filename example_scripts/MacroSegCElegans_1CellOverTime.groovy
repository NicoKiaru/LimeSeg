#@ImageJ ij
// This macro shows how to use LimeSeg without using the pre-defined commands

// Commands to initialize and clear LimeSeg state
import eu.kiaru.limeseg.commands.SphereSeg;
import eu.kiaru.limeseg.commands.ClearAll;

// Objects used by LimeSeg
import eu.kiaru.limeseg.LimeSeg;
import eu.kiaru.limeseg.struct.Cell;
import eu.kiaru.limeseg.struct.CellT;
import eu.kiaru.limeseg.struct.DotN;
import eu.kiaru.limeseg.struct.Vector3D;

import ij.IJ // imports IJ1 

// Fetch demo dataset = fluorescent lipid vesicles acquired by Spining Disc confocal microscope
// - Image URL
String testImgRepo = 'https://raw.githubusercontent.com/NicoKiaru/TestImages/master/'
String vesiclesURL = testImgRepo+'CElegans/dub-0.5xy-TP1-22.tif'
// - Open and shows the image (ImagePlus)
imgPlus = IJ.openImage(vesiclesURL)
imgPlus.show()
(new LimeSeg()).initialize();
LimeSeg.clearAllCells(); // Removes all previous objects of LimeSeg


//------------------------------ Definition of 3D image for LimeSeg
// - Sets linked 3D Image
float imgZScale=4;

LimeSeg.workingImP=imgPlus;  // First thing to do
LimeSeg.setZScale(imgZScale);
LimeSeg.currentFrame=1;
LimeSeg.currentChannel=1;
LimeSeg.setWorkingImage(imgPlus,LimeSeg.currentChannel,LimeSeg.currentFrame);



//----------------------------- Creation of a Cell object
// First of all : creation of a Cell object
LimeSeg.clearAllCells();
LimeSeg.newCell(); // Creates a new cell object which is stored in LimeSeg.currentCell static variable
Cell myCell = LimeSeg.currentCell;
myCell.id_Cell= "myCell"; // Renames the cell but take care, renaming this way do not prevent duplicate names


// Now let's populate the Cell with CellT objects (= 3D object at a specific timepoint)
// The easiest way is to go through clipped dots objects
float x0=240;
float y0=120;
float z0=6;
float r0=15;
float d0=4;

LimeSeg.setOptimizerParameter("d_0",d0); // Very important : it is an optimizer parameter but it also sets the "precision" of created clipped dots objects
LimeSeg.makeSphere(x0,y0,z0,r0);
// Now let's select the correct timepoint

// And paste the sphere to the currentCell (newly created) at the currentFrame (set at the beginning)
LimeSeg.pasteDotsToCellT();


//------------------------------- 3D Vizualization
// Let's see in 3D how it looks
LimeSeg.make3DViewVisible();
LimeSeg.clear3DDisplay();
LimeSeg.putAllCellsTo3DDisplay();
// Recenter the 3D view
// LimeSeg.jcr is the 3D viewer from LimeSeg (jcr stands for Jogl Cell Renderer)
// lookAt is a LimeSeg Vector3D object
LimeSeg.jcr.lookAt.x=x0;
LimeSeg.jcr.lookAt.y=y0;
LimeSeg.jcr.lookAt.z=z0*imgZScale;


//---------------------------- Optimizer setting up
// Now sets the parameters for the optimizer (LimeSeg.opt)
// LimeSeg.setOptimizerParameter("d_0",3);
LimeSeg.setOptimizerParameter("normalForce",0.015); // == f_pressure
LimeSeg.setOptimizerParameter("radiusSearch",8);  // in pixel, the range over which a maximum is looked for
   // Link to the image enabled (should be almost always 1)


// Now, puts CellT objects into the Optimizer
LimeSeg.clearOptimizer();    // first clear everything
LimeSeg.currentCell=myCell; // not really necessary here
LimeSeg.currentFrame=1;		 // not really necessary here

LimeSeg.putCurrentCellTToOptimizer(); // That's it! The Cell vesicle a timepoint 1 is into the optimizer

//---------------------------- Optimization
// Sphere relaxation
LimeSeg.setOptimizerParameter("attractToMax",0);
LimeSeg.setOptimizerParameter("normalForce",0.0);
LimeSeg.runOptimisation(1000);

// Ok, now let's optimize
LimeSeg.setOptimizerParameter("normalForce",0.015);
LimeSeg.setOptimizerParameter("attractToMax",1);
LimeSeg.opt.requestResetDotsConvergence=true;
// Optimizer's ready, let's run it
LimeSeg.runOptimisation(1000); // parameter = maximal number of iteration

// Now let's segment the other Timepoints


for (int timePoint=2;timePoint<11;timePoint++) {
	LimeSeg.currentFrame=timePoint;
	// Copy from previous frame to next frame
	LimeSeg.currentFrame=timePoint-1;
	LimeSeg.currentCell=myCell;
	LimeSeg.copyDotsFromCellT();
	LimeSeg.currentFrame=timePoint;
	LimeSeg.pasteDotsToCellT();

	// Now the cell is initialized with the shape from the previous frame
	// Let's feed the Optimizer
	LimeSeg.clearOptimizer();
	LimeSeg.putCurrentCellTToOptimizer();
	// Done
	
	// can be put out of the loop : 
	// basically it says that there's no need to inflate the shape right now:
	// this is true is the shape from one timepoint to another is not too different
	LimeSeg.setOptimizerParameter("normalForce",0.00); 	
	LimeSeg.runOptimisation(1000); // Do the optimisation
}

// After timepoint 10, the cells divides 
// -> the approximation of cell shape not changing too much between time points is not valid anymore

// 2D output:
LimeSeg.clearOverlay();
LimeSeg.addAllCellsToOverlay();
LimeSeg.updateOverlay();
