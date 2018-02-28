package eu.kiaru.limeseg.commands;

import java.util.ArrayList;

import eu.kiaru.limeseg.LimeSeg;
import eu.kiaru.limeseg.struct.CellT;
import ij.measure.ResultsTable;
import ij.plugin.filter.Analyzer;

public class CommandHelper {
	
	static void displaySegmentationOutput(ArrayList<CellT> currentlyOptimizedCellTs, float realXYPixelSize, boolean constructMesh) {
		ResultsTable rt = Analyzer.getResultsTable();
      	 
   		if (rt == null) {
   		        rt = new ResultsTable();
   		        Analyzer.setResultsTable(rt);
   		}
   		for (CellT ct : currentlyOptimizedCellTs) {
   		    rt.incrementCounter();
   		    int i = rt.getLastColumn()+1;
   			ct.updateCenter();
   		    rt.addValue("Cell Name", ct.c.id_Cell);
   			rt.addValue("Number of Surfels", ct.dots.size());
   			rt.addValue("Center X", ct.center.x);
   			rt.addValue("Center Y", ct.center.y);
   			rt.addValue("Center Z", (ct.center.z/LimeSeg.opt.getOptParam("ZScale"))+1); // IJ1 style numerotation
   			rt.addValue("Frame", ct.frame);
   			rt.addValue("Channel", ct.c.cellChannel);       			
   			rt.addValue("Mesh ?", (constructMesh)?"YES":"NO");
   			if (constructMesh) {
   				rt.addValue("Euler characteristic", ct.dots.size()-3.0/2.0*ct.triangles.size()+ct.triangles.size());
   				rt.addValue("Free edges", ct.freeEdges);
   				rt.addValue("Surface", ct.getSurface());
   				rt.addValue("Volume", ct.getVolume());
   				rt.addValue("Real Surface", ct.getSurface()*(realXYPixelSize*realXYPixelSize));
   				rt.addValue("Real Volume", ct.getVolume()*(realXYPixelSize*realXYPixelSize*realXYPixelSize));
   			}
   		}       		
   		rt.show("Results");
	}
}
