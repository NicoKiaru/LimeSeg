#@ImageJ ij

import eu.kiaru.limeseg.commands.SphereSeg;
import eu.kiaru.limeseg.commands.ClearAll;

import org.scijava.util.ColorRGB;
import ij.IJ // imports IJ1
import ij.plugin.frame.RoiManager // imports ROIManager
import ij.gui.OvalRoi
import ij.gui.Roi
import ij.ImagePlus;


// Fetch demo dataset = fluorescent lipid vesicles acquired by Spining Disc confocal microscope
// - Image URL
String testImgRepo = 'https://raw.githubusercontent.com/NicoKiaru/TestImages/master/'
String vesiclesURL = testImgRepo+'Vesicles/Vesicles.tif'
// - Open and shows the image (ImagePlus)
imgPlus = IJ.openImage(vesiclesURL)
imgPlus.show();

// defines a function to "easily" communicate with the Roi manager
def addCircleToRoiManager(ImagePlus img, float x, float y, float z, float radius) {	
	int currentSlice = img.getCurrentSlice();
	RoiManager manager = RoiManager.getInstance();
	if (manager == null)
	    manager = new RoiManager();
	// the first slice is 1 (not 0)
	OvalRoi circle = new OvalRoi(x-radius,y-radius,2*radius,2*radius);
	circle.setPosition(1,(int)z, 1);
	img.setSliceWithoutUpdate((int)z);
	manager.add(img, circle, (int)z);
	img.setSlice(currentSlice);	
}

def clearRoiManager() {
	RoiManager manager = RoiManager.getInstance();
	if (manager == null)
	    manager = new RoiManager();
	manager.reset();
}

clearRoiManager();
addCircleToRoiManager(imgPlus,75,90,14,15); // Vesicle 1
addCircleToRoiManager(imgPlus,89,21,19,15); // Vesicle 2

// ij2 way of launching a command
ij.command().run(ClearAll.class,true) // Clears all previous objects of LimeSeg
ij.command().run(SphereSeg.class,true,
					"d_0",3,
					"f_pressure",0.025,
					"z_scale",340.0/133.0,
					"range_in_d0_units",2,
					"sameCell",false,
					"show3D",true,
					"color", new ColorRGB(200,150,50),
					"numberOfIntegrationStep",-1,
					"realXYPixelSize",0.133);

// ImageJ 1 way of executing the command
//IJ.run(imgPlus, "Clear All");
//IJ.run(imgPlus, "Sphere Seg", "d_0=2.5 f_pressure=0.02 z_scale=2.54 range_in_d0_units=2 color=255,0,0 samecell=true show3d=true numberofintegrationstep=-1 realxypixelsize=0.133");

