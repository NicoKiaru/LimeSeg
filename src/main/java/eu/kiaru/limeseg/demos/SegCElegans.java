package eu.kiaru.limeseg.demos;

import ij.IJ;
import org.scijava.io.DefaultIOService;
import org.scijava.io.IOService;
import org.scijava.service.ServiceHelper;

import eu.kiaru.limeseg.LimeSeg;
import ij.ImagePlus;
import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.display.ImageDisplay;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.display.imagej.ImageJFunctions;

public class SegCElegans {

	static ImagePlus myImpPlus;
    public static void main(final String... args) {
    	final ImageJ ij = new ImageJ();
	    ij.ui().showUI();
	    /*final ServiceHelper sh = new ServiceHelper(ij.getContext());
        final IOService io = sh.loadService(DefaultIOService.class);
        final Dataset datasetIn = DemoHelper.getDatasetFromResources(io,"images/Dub-WilliamMohler-Tp33-Half.zip");
        final ImageDisplay imageDisplay = (ImageDisplay) ij.display().createDisplay(datasetIn);
        ImagePlus myImpPlus = ImageJFunctions.wrap((RandomAccessibleInterval)datasetIn,"CElegans");*/
		myImpPlus = IJ.openImage("src/main/resources/images/Dub-WilliamMohler-Tp33-Half.zip");
		myImpPlus.setTitle("CElegans");
		myImpPlus.show();
        LimeSeg lms = new LimeSeg();
        lms.initialize();
	    TestSegEmbryoCElegansFull();
    }
    
    static void addCell(double x, double y, double z, double r,int add) {
    	if(add==0) {
    		LimeSeg.newCell();
    	}
    	x=x/2f;
    	y=y/2f;
    	if (true){
    		LimeSeg.makeSphere((float)x,(float)y,(float)z,(float)r);
    		LimeSeg.pasteDotsToCellT();
	    	double c=1.5;
	    	double s=0.5;
	    	double red=(x/758*2-s)*c+s;
	    	double green=(y/512*2-s)*c+s;
	    	double blue=(z/33-s)*c+s;
	    	if(red>1) red=1;
	    	if (red<0) red=0;
	    	if(green>1) green=1;
	    	if (green<0) green=0;
	    	if(blue>1) blue=1;
	    	if (blue<0) blue=0;
	    	LimeSeg.setCellColor((float)red,(float)green,(float)blue,1.0f);
    	}
    }
    
    
    public static long TestSegEmbryoCElegansFull() {
    	// Feeds the 3D image to the optimzier
    	// reset previous state
    	LimeSeg.clearCell("ALL");
    	LimeSeg.clearOptimizer();
    	LimeSeg.make3DViewVisible();
        final Dataset dataset = DemoHelper.getDatasetFromResources("images/Dub-WilliamMohler-Tp33-Half.zip");
        LimeSeg.setWorkingImage(myImpPlus, LimeSeg.currentChannel, LimeSeg.currentFrame); // Takes ages!
        LimeSeg.opt.setOptParam("ZScale", 7.0f/2f);
        LimeSeg.opt.setOptParam("d_0",7.0f/2f);
        LimeSeg.opt.setOptParam("normalForce",0.025f);
        LimeSeg.opt.setOptParam("k_grad",0.03f);
        LimeSeg.opt.setOptParam("radiusTresholdInteract", 1.76f);
        LimeSeg.opt.setOptParam("rmIfNeighborAbove",11.0f);
        LimeSeg.opt.setOptParam("ageMinGenerate",8.0f);
        LimeSeg.opt.setOptParam("maxDisplacementPerStep",0.3f);
        LimeSeg.opt.setOptParam("radiusSearch",40.0f);
        LimeSeg.opt.setOptParam("radiusRelaxed",10.0f);
        LimeSeg.opt.setOptParam("radiusDelta",3.5f/2f);
	    float rIni=20/2f;
	    addCell(530.0,298.0,8.0,rIni,0);
	    addCell(483.0,214.0,10.0,rIni,0);
	    addCell(656.0,228.0,13.0,rIni,0);
	    addCell(651.0,315.0,9.0,rIni,0);
	    addCell(584.0,216.0,8.0,rIni,0);
	    addCell(583.0,308.0,16.0,rIni,0);
	    addCell(552.0,374.0,9.0,rIni,0);
	    addCell(617.0,392.0,14.0,rIni,0);
	    addCell(695.0,343.0,15.0,rIni,0);
	    addCell(679.0,269.0,19.0,rIni,0);
	    addCell(546,162,14,rIni,0);
	    addCell(452,359,11,rIni,0);
	    addCell(488,266,17,rIni,0);
	    addCell(614,174,21,rIni,0);
	    addCell(607,243,28,rIni,0);
	    addCell(561,288,22,rIni,0);
	    addCell(497,395,16,rIni,0); // div ici
	    addCell(535,401,22,rIni,1);
	    addCell(430,396,20,rIni,0);
	    addCell(482,168,21,rIni,0);
	    addCell(628,349,26,rIni,0);
	    addCell(433,278,4,rIni,0);
	    addCell(385,185,7,rIni,0);
	    addCell(372,334,7,rIni,0);
	    addCell(377,276,11,rIni,0);
	    addCell(292,343,10,rIni,0);
	    addCell(284,248,6,rIni,0);
	    addCell(424,305,22,rIni,0);
	    addCell(425,213,23,rIni,0);
	    addCell(407,144,14,rIni,0);
	    addCell(324,199,14,rIni,0);
	    addCell(241,269,12,rIni,0);
	    addCell(361,356,24,rIni,0);
	    addCell(381,388,14,rIni,0);
	    addCell(297,378,17,rIni,0);
	    addCell(201,333,8,rIni,0);
	    addCell(167,243,8,rIni,0);
	    addCell(260,166,12,rIni,0);
	    addCell(128,300,13,rIni,0);
	    addCell(90,249,16,rIni,0);
	    addCell(142,184,12,rIni,0);
	    addCell(165,346,16,rIni,0);
	    addCell(226,357,20,rIni,0);
	    addCell(215,190,18,rIni,0);
	    addCell(236,259,23,rIni,0);
	    addCell(128,290,24,rIni,0);
	    addCell(165,203,24,rIni,0);
	    addCell(301,312,24,rIni,0);
	    addCell(337,237,24,rIni,0);
	    addCell(352,173,23,rIni,0);
	    addCell(270,172,22,rIni,0);
	    int NC = LimeSeg.allCells.size();
	    for (int i=0;i<NC;i++) {
	    	LimeSeg.currentCell = LimeSeg.allCells.get(i);
	    	LimeSeg.putCurrentCellTToOptimizer();
	    }
	    LimeSeg.putAllCellsTo3DDisplay();
	    LimeSeg.set3DViewCenter(367f/2f,258f/2f,15f);
	    LimeSeg.set3DViewRot(165f,-35f,10f);
	    LimeSeg.jcr.RatioGlobal=0.01f;
	    LimeSeg.jcr.setViewMode(8);
	    LimeSeg.putAllCellsTo3DDisplay();
	    LimeSeg.opt.tic(); 									   // tip
	    LimeSeg.runOptimisation(-1); 						   // -1 = until convergence normally 741 steps
	    return LimeSeg.opt.toc("Durée de la segmentation : "); // top}
	    
    }   
}
