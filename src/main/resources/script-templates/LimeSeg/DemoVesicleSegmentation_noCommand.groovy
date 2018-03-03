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



// Initializes LimeSeg and displays LimeSeg GUI
(new LimeSeg()).initialize();

// Fetch demo dataset = fluorescent lipid vesicles acquired by Spining Disc confocal microscope
// - Image URL
String testImgRepo = 'https://raw.githubusercontent.com/NicoKiaru/TestImages/master/'
String vesiclesURL = testImgRepo+'Vesicles/Vesicles.tif'
// - Open and shows the image (ImagePlus)
imgPlus = IJ.openImage(vesiclesURL)
imgPlus.show()
//LimeSeg.clearAllCells(); // Removes all previous objects of LimeSeg


//------------------------------ Definition of 3D image for LimeSeg
// - Sets linked 3D Image
float imgZScale=0.340/0.133; // 340 nm per slice, 133 nm per pixel

LimeSeg.workingImP=imgPlus;  // First thing to do
LimeSeg.setZScale(imgZScale);
LimeSeg.currentFrame=1;
LimeSeg.currentChannel=1;
LimeSeg.setWorkingImage(imgPlus,LimeSeg.currentChannel,LimeSeg.currentFrame);


//----------------------------- Creation of a Cell object
// First of all : creation of a Cell object
LimeSeg.clearAllCells();
LimeSeg.newCell(); // Creates a new cell object which is stored in LimeSeg.currentCell static variable
Cell vesicle = LimeSeg.currentCell;
vesicle.id_Cell= "Vesicle"; // Renames the cell but take care, renaming this way do not prevent duplicate names

// Now let's populate the Cell with CellT objects (= 3D object at a specific timepoint)
// The easiest way is to go through clipped dots objects
float x0=75;
float y0=90;
float z0=14;
float r0=15;

LimeSeg.setOptimizerParameter("d_0",4); // Very important : it is an optimizer parameter but it also sets the "precision" of created clipped dots objects
LimeSeg.makeSphere(x0,y0,z0,r0);
// Now let's select the correct timepoint

// And paste the sphere
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
LimeSeg.setOptimizerParameter("d_0",4);
LimeSeg.setOptimizerParameter("normalForce",0.02); // == f_pressure
LimeSeg.setOptimizerParameter("radiusSearch",10);  // in pixel, the range over which a maximum is looked for
LimeSeg.setOptimizerParameter("attractToMax",1);   // Link to the image enabled (should be almost always 1)

// Now, puts CellT objects into the Optimizer
LimeSeg.clearOptimizer();    // first clear everything
LimeSeg.currentCell=vesicle; // not really necessary here
LimeSeg.currentFrame=1;		 // not really necessary here

LimeSeg.putCurrentCellTToOptimizer(); // That's it! The Cell vesicle a timepoint 1 is into the optimizer

//---------------------------- Optimization
// Optimizer's ready, let's run it
LimeSeg.runOptimisation(1000); // parameter = maximal number of iteration

//---------------------------- Output data retrieval
// But the output is a point cloud, not a mesh... To transform the current CellT object into mesh:
// Select which CellT object is active
LimeSeg.currentCell=vesicle; 
LimeSeg.currentFrame=1;		 
// Set the expected distance between points
LimeSeg.setOptimizerParameter("d_0",4);
// Calls construct Mesh method
LimeSeg.constructMesh();

// Get shape surface: !! The mesh has to be reconstructed
// Fetch CellT object:
CellT my3Dobject = vesicle.getCellTAt(LimeSeg.currentFrame);
if (my3Dobject.freeEdges!=0) {
	print("Mesh reconstruction problem, wrong surface and volume value!");
	print("Free edges = "+my3Dobject.freeEdges+"\n"); 
}
print("Surface = "+my3Dobject.getSurface()+"\n");
print("Volume  = "+ my3Dobject.getVolume()+"\n");



// To display the mesh in the 3D Viewer:
LimeSeg.setCell3DDisplayMode(1);

// To display the points in the 3D Viewer:
LimeSeg.setCell3DDisplayMode(0);

// 2D Display:
LimeSeg.putCurrentCellToOverlay();
LimeSeg.updateOverlay();
