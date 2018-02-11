package eu.kiaru.limeseg.demos;

import java.io.File;
import java.net.URL;

//import org.apache.commons.io.FileUtils;
import org.scijava.io.DefaultIOService;
import org.scijava.io.IOService;
import org.scijava.service.ServiceHelper;

import eu.kiaru.limeseg.LimeSeg;
import net.imagej.Dataset;
import net.imagej.ImageJ;

public class DemoHelper {
	
	static public Dataset getDatasetFromResources(IOService io, String pathToResource) {
		 /*URL imageURL = LimeSeg.class.getClassLoader().getResource(pathToResource);
	     if (imageURL==null) {
	    	 System.out.println("Could not find resource "+pathToResource);
	    	 return null;
	     }
	     final Dataset dataset;
	     try {        	
	      	File tempDir = new File(System.getProperty("java.io.tmpdir"));
	        String fName = tempDir.getAbsolutePath()+"/PieceOfBrain.zip";
	      	File dest = new File(fName);
	       	FileUtils.copyURLToFile(imageURL, dest);
	       	dataset = (Dataset) io.open(fName);
	     } catch (Exception e) {
	    	 e.printStackTrace();
	    	 return null;
	     }
	     return dataset;*/
		return null;
	}
	
	static public Dataset getDatasetFromResources(String pathToResource) {
		final ImageJ ij = new ImageJ();
        final ServiceHelper sh = new ServiceHelper(ij.getContext());
        final IOService io = sh.loadService(DefaultIOService.class);
		return getDatasetFromResources(io, pathToResource);
	}
	
    static public void tic() {
        startTime = System.nanoTime();
    }
    
    static long startTime,endTime,duration; 
    
    static public long toc(String message) {
        endTime = System.nanoTime();
        duration = (endTime - startTime);  
        System.out.println(message+":\t"+(duration/1000)+"\t us");  
        return duration;
    }
    
    static public long toc() {
        endTime = System.nanoTime();
        duration = (endTime - startTime);  
        return duration;
    }
    
}
