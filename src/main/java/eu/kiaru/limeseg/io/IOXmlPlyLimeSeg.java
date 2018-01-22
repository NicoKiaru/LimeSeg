package eu.kiaru.limeseg.io;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.xml.parsers.*;
import javax.xml.transform.*;
import javax.xml.transform.dom.*;
import javax.xml.transform.stream.*;
import org.smurn.jply.ElementReader;
import org.smurn.jply.ElementType;
import org.smurn.jply.PlyReader;
import org.smurn.jply.PlyReaderFile;
import org.w3c.dom.*;

import eu.kiaru.limeseg.*;
import eu.kiaru.limeseg.gui.JOGL3DCellRenderer;
import eu.kiaru.limeseg.opt.Optimizer;
import eu.kiaru.limeseg.struct.Cell;
import eu.kiaru.limeseg.struct.CellT;
import eu.kiaru.limeseg.struct.DotN;
import eu.kiaru.limeseg.struct.TriangleN;
/**
 *
 * @author Nico
 */
public class IOXmlPlyLimeSeg {
	
	   static Element getXmlOptimizerState(Optimizer opt, Document dom) {
	       Element optParam = dom.createElement("Optimizer");
	       for (String st:Optimizer.paramList) {
		       Element e = dom.createElement(st);
		       e.appendChild(dom.createTextNode(Double.toString(opt.getOptParam(st))));
		       optParam.appendChild(e);
	       }   
		   return optParam;
	   }
	   
	   static void setOptimizerStateFromXml(Optimizer opt, NodeList node) {
		   NodeList params = node.item(0).getChildNodes();
		   for (int i=0;i<params.getLength();i++) {
			   if(params.item(i).getNodeType() == Node.ELEMENT_NODE) {
				   String paramName = (((Element)(params.item(i))).getNodeName());
				   Float paramValue = Float.valueOf(params.item(i).getTextContent());				
				   opt.setOptParam(paramName, paramValue);
			   }
		   }
	   }
	   
	   static Element getXml3DViewerState(JOGL3DCellRenderer jcr, Document dom) {
		   System.out.println("getXml3DViewerState needs to be implemented!");
		   return null;
	   }
	   
	   public static byte[] FloatArray2ByteArray(float[] values){
	        ByteBuffer buffer = ByteBuffer.allocate(4 * values.length);
	        for (float value : values){
	            buffer.putFloat(value);
	        }
	        return buffer.array();
	   }
	   
	   public static byte[] IntArray2ByteArray(int[] values){
	        ByteBuffer buffer = ByteBuffer.allocate(4 * values.length);
	        for (int value : values){
	            buffer.putInt(value);
	        }
	        return buffer.array();
	   }
	   
	   
	   static public void saveCellTAsPly(CellT ct, String filename) {
		   boolean isBinary = true;
		   FileOutputStream fos;
		   try {
			   fos = new FileOutputStream(filename);
			   BufferedOutputStream buffOut=new BufferedOutputStream(fos);
			   
			   buffOut.write("ply\n".getBytes("UTF-8"));
			   buffOut.write("format ".getBytes("UTF-8"));
			   String s = isBinary ? "binary_big_endian" : "ascii";//"ply\n";
			   buffOut.write(s.getBytes("UTF-8"));
			   buffOut.write(" 1.0\n".getBytes("UTF-8"));
			   buffOut.write(("element vertex "+ct.dots.size()+"\n").getBytes("UTF-8"));
			   buffOut.write("property float x\n".getBytes("UTF-8"));
			   buffOut.write("property float y\n".getBytes("UTF-8"));
			   buffOut.write("property float z\n".getBytes("UTF-8"));
			   buffOut.write("property float nx\n".getBytes("UTF-8"));
			   buffOut.write("property float ny\n".getBytes("UTF-8"));
			   buffOut.write("property float nz\n".getBytes("UTF-8"));
		        boolean writesMesh=(ct.tesselated==true)&&(ct.triangles!=null);
		        if (writesMesh) {
		        	buffOut.write(("element face "+ct.triangles.size()+"\n").getBytes("UTF-8"));
		        	buffOut.write(("property list int int vertex_index\n").getBytes("UTF-8"));
		        }
		        buffOut.write("end_header\n".getBytes("UTF-8"));
		        
		        try {
		        	int chunkSize=10000;
		        		int numberOfChunks = ct.dots.size()/chunkSize;
		        		int chunkNumber=0;
		        		int dotCount=0;
		        		float[] chunkArray = new float[6*chunkSize];
		        		// complete Chunks
		        		while (chunkNumber<numberOfChunks) {
		        			for (int indexDot=0;indexDot<chunkSize;indexDot++) {
		        				DotN d = ct.dots.get(dotCount+indexDot);
		        				chunkArray[6*indexDot+0] = d.pos.x;
		        				chunkArray[6*indexDot+1] = d.pos.y;
		        				chunkArray[6*indexDot+2] = d.pos.z;
		        				chunkArray[6*indexDot+3] = d.Norm.x;
		        				chunkArray[6*indexDot+4] = d.Norm.y;
		        				chunkArray[6*indexDot+5] = d.Norm.z;		        				
		        			}
		        			buffOut.write(FloatArray2ByteArray(chunkArray));
		        			dotCount+=chunkSize;
		        			chunkNumber++;
		        		}
		        		// Last Chunk - incomplete
		        		int nRemainingDots=ct.dots.size()-dotCount;
		        		chunkArray = new float[6*nRemainingDots];
		        		for (int indexDot=0;indexDot<nRemainingDots;indexDot++) {
	        				DotN d = ct.dots.get(dotCount+indexDot);
	        				chunkArray[6*indexDot+0] = d.pos.x;
	        				chunkArray[6*indexDot+1] = d.pos.y;
	        				chunkArray[6*indexDot+2] = d.pos.z;
	        				chunkArray[6*indexDot+3] = d.Norm.x;
	        				chunkArray[6*indexDot+4] = d.Norm.y;
	        				chunkArray[6*indexDot+5] = d.Norm.z;		        				
	        			}
	        			buffOut.write(FloatArray2ByteArray(chunkArray));

			        if (writesMesh) {
			        	// Needs improvement!
				        for (TriangleN tr:ct.triangles) {
				        	buffOut.write(IntArray2ByteArray(new int[] {3,tr.id1,tr.id2,tr.id3}));
				        }
			        }
			        buffOut.close();
		        } catch (IOException e) {
					e.printStackTrace();
				}	
		        buffOut.flush();
		        fos.close();
		   } catch (Exception e) {
			   e.printStackTrace();
		   }  
	   }
	   
	  
	   
	   static public void loadCellTFromPly(CellT ct, String filename) {
		   try {		   
			   PlyReader plyreader = new PlyReaderFile(filename);
			   ElementReader reader = plyreader.nextElementReader();
		       while (reader != null) {
		            ElementType type = reader.getElementType();
		            // In PLY files vertices always have a type named "vertex".
		            if (type.getName().equals("vertex")) {
		                ArrayList<DotN> dots = new ArrayList<DotN>(reader.getCount());
		                // Read the elements. They all share the same type.
		            	org.smurn.jply.Element element = reader.readElement();
		                while (element != null) {
		                	 DotN dn = new DotN();
		                	 dn.pos.x=(float) element.getDouble("x");
		                	 dn.pos.y=(float) element.getDouble("y");
		                	 dn.pos.z=(float) element.getDouble("z");

		                	 dn.Norm.x=(float) element.getDouble("nx");
		                	 dn.Norm.y=(float) element.getDouble("ny");
		                	 dn.Norm.z=(float) element.getDouble("nz");
		                	 dn.ct=ct;
		                     element = reader.readElement();
		                     dots.add(dn);
		                }
		                ct.dots=dots;
		            }
		            if (type.getName().equals("face")) {
		            	ArrayList<TriangleN> tris = new ArrayList<TriangleN>(reader.getCount());
			            org.smurn.jply.Element triangle = reader.readElement();
		                while (triangle != null) {
			            	int[] indices = triangle.getIntList("vertex_index");
			            	TriangleN tri = new TriangleN();
			            	tri.id1=indices[0];
			            	tri.id2=indices[1];
			            	tri.id3=indices[2];
			            	triangle = reader.readElement();
		                    tris.add(tri);
		                    
		                }
		            	ct.triangles=tris;
		            	ct.tesselated=true;
		            }	            		            
		            // Close the reader for the current type before getting the next one.
		            reader.close();
		            reader = plyreader.nextElementReader();
		       }
		       plyreader.close();
		   } catch (IOException e) {
				e.printStackTrace();
		   }
	   }
	   
	   static public void hydrateCellT(Cell c, String path) {
		   // path should contain the folder with the ply files for each timepoint
		   c.cellTs.clear();
		   Pattern pattern = Pattern.compile("[0-9]+"); 
		   File dir = new File(path);
		   if (!dir.isDirectory()) {
	    		System.out.println("Error, folder for cell "+c.id_Cell+" not found.");
	    		return;
		   }		   
		   File[] files = dir.listFiles(new FilenameFilter() {
		        @Override
		        public boolean accept(File dir, String name) {
		            return name.matches("T_[0-9]+.ply");
		        }
		   });
		   for (File f:files) {
			   String fileName = f.getAbsolutePath();
			   System.out.println("Found cellT file : "+fileName);
			   Matcher matcher = pattern.matcher(f.getName());
			   matcher.find(); 			   
			   String match = matcher.group(); // Get the matching string
			   int tp = Integer.valueOf(match);
			   System.out.println("TP="+tp);			   		   
			   CellT ct = new CellT(c,tp);
			   loadCellTFromPly(ct,f.getAbsolutePath());
			   ct.frame=tp;
			   c.cellTs.add(ct);
		   }
	   }
	   
	   static public void loadState(LimeSeg lms, String path) {
		   if (!path.endsWith(File.separator)) {path=path+File.separator;}
		   File fXmlFile = new File(path+"LimeSegParams.xml");
			DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
			DocumentBuilder dBuilder;
			try {
				dBuilder = dbFactory.newDocumentBuilder();			
				Document doc = dBuilder.parse(fXmlFile);
				//optional, but recommended
				//read this - http://stackoverflow.com/questions/13786607/normalization-in-dom-parsing-with-java-how-does-it-work
				doc.getDocumentElement().normalize();
				if (doc.getDocumentElement().getNodeName().equals("LimeSegParameters")) {
					// Looks ok
					Node version = doc.getElementsByTagName("Version").item(0);
					if (version.getTextContent().equals("0.1")) {
						// Ok, version 0.1 is provided
						setOptimizerStateFromXml(lms.opt,doc.getElementsByTagName("Optimizer"));
						lms.allCells.clear();
						// Get cells node
						Node cellsNode = doc.getElementsByTagName("Cells").item(0);
						if (cellsNode==null) return;
						NodeList cellList = cellsNode.getChildNodes();
						for (int i=0;i<cellList.getLength();i++) {
							Node currentCell = cellList.item(i);
							if (currentCell.getNodeType() == Node.ELEMENT_NODE) {
								   Element e = (Element)(currentCell);
								   String cellName = (e.getNodeName());
								   System.out.println("CellName = "+cellName);
								   Cell c = new Cell();
								   lms.allCells.add(c);
								   c.id_Cell=cellName;
								   c.color[0]=Float.valueOf(e.getElementsByTagName("R").item(0).getTextContent());
								   c.color[1]=Float.valueOf(e.getElementsByTagName("G").item(0).getTextContent());
								   c.color[2]=Float.valueOf(e.getElementsByTagName("B").item(0).getTextContent());
								   c.color[3]=Float.valueOf(e.getElementsByTagName("A").item(0).getTextContent());
								   c.cellChannel = Integer.valueOf(e.getElementsByTagName("channel").item(0).getTextContent());
								   String pathToCell = path+cellName;
								   hydrateCellT(c,pathToCell);
							}							 							
						}						
					} else {
						if (version.getTextContent().equals("0.2")) {
							// Ok, version 0.2 is provided
							setOptimizerStateFromXml(lms.opt,doc.getElementsByTagName("Optimizer"));
							lms.allCells.clear();
							File[] directories = new File(path).listFiles(File::isDirectory);
							for (File currentPath:directories) {
								// Is there a CellParams.xml file ?
								File paramsCell = new File(currentPath+File.separator+"CellParams.xml");
								if (paramsCell.exists()) {
									   Cell c = new Cell();
									   lms.allCells.add(c);
									   c.id_Cell=paramsCell.getParentFile().getName();
									   Document docCell = dBuilder.parse(paramsCell);
									   docCell.getDocumentElement().normalize();
									   if (docCell.getDocumentElement().getNodeName().equals("CellParameters")) {
											c.color[0]=Float.valueOf(docCell.getElementsByTagName("R").item(0).getTextContent());
											c.color[1]=Float.valueOf(docCell.getElementsByTagName("G").item(0).getTextContent());
											c.color[2]=Float.valueOf(docCell.getElementsByTagName("B").item(0).getTextContent());
											c.color[3]=Float.valueOf(docCell.getElementsByTagName("A").item(0).getTextContent());
											c.cellChannel = Integer.valueOf(docCell.getElementsByTagName("channel").item(0).getTextContent());
											hydrateCellT(c,currentPath.getAbsolutePath());											
									   }									   
								}
							}
						}
					}
					
				} else {
					System.out.println("Cannot open file. Root node is not LimeSegParameters as expected");
				}				
			} catch (Exception e) {
				e.printStackTrace();
			}
	   }
	   
	   static public void saveStatev0p1(LimeSeg lms, String path_in) {
		   assert !lms.optimizerIsRunning;
		    if (!path_in.endsWith(File.separator)) {path_in=path_in+File.separator;}
		    String path = path_in;
		    Document dom;
		    // instance of a DocumentBuilderFactory
		    DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
		    try {
		    	File dir = new File(path);
		    	if (!dir.isDirectory()) {
		    		System.out.println("Erreur, given path is not a directory");
		    		return;
		    	}
		    	// By default removes all files in the folder
		    	// But ask for confirmation if the folder is not empty...
		    	if (dir.listFiles().length!=0) {
		    		System.out.println("Saving will remove the content of the folder "+path+" that contains "+dir.listFiles().length+" files and folders.");
		    		if (!askYesNo("Are you sure you want to proceed [Y/N] ?")) {return;}
		    	}
		    	purgeDirectory(dir,1);
		        DocumentBuilder db = dbf.newDocumentBuilder();
		        // create instance of DOM
		        dom = db.newDocument();
		        // create the optimizer parameter element
		        Element lmsParams = dom.createElement("LimeSegParameters");
		        Element versionInfo = dom.createElement("Version");
		        lmsParams.appendChild(versionInfo);
			    versionInfo.appendChild(dom.createTextNode("0.1"));
		        lmsParams.appendChild(getXmlOptimizerState(lms.opt, dom));
		        System.out.println("getXml3DViewerState needs to be implemented!");
		        Element imageInfos = dom.createElement("ImageParameters");
		        lmsParams.appendChild(imageInfos);		        
		        //lmsParams.appendChild(getXml3DViewerState(lms.jcr, dom));
		        Element cells = dom.createElement("Cells");
		        lms.allCells.forEach(c->{
		        	Element currCell = dom.createElement(c.id_Cell);
		        	// Cell Channel
				    Element channel = dom.createElement("channel");
				    channel.appendChild(dom.createTextNode(Integer.toString(c.cellChannel)));
				    currCell.appendChild(channel);
		        	// Cell color
		        	Element color = dom.createElement("color");		        	
				       Element r = dom.createElement("R");
				       r.appendChild(dom.createTextNode(Float.toString(c.color[0])));
				       color.appendChild(r);

				       Element g = dom.createElement("G");
				       g.appendChild(dom.createTextNode(Float.toString(c.color[1])));
				       color.appendChild(g);

				       Element b = dom.createElement("B");
				       b.appendChild(dom.createTextNode(Float.toString(c.color[2])));
				       color.appendChild(b);

				       Element a = dom.createElement("A");
				       a.appendChild(dom.createTextNode(Float.toString(c.color[3])));
				       color.appendChild(a);
				    currCell.appendChild(color);
		        	cells.appendChild(currCell);
		        	// Now writes all ply files for CellT object
		        	c.cellTs.forEach(ct -> {
		        		String pathCell=path+File.separator+c.id_Cell+File.separator;
		        	    File dirCell = new File(pathCell);		        	    
		        	    // attempt to create the directory here
		        	    if (dirCell.mkdir()) {
		        	    	IOXmlPlyLimeSeg.saveCellTAsPly(ct,pathCell+"T_"+ct.frame+".ply");
		        	    } else {
		        	    	if (dirCell.exists()) {
		        	    		IOXmlPlyLimeSeg.saveCellTAsPly(ct,pathCell+"T_"+ct.frame+".ply");
		        	    	}
		        	    }
		        	});
		        });
		        lmsParams.appendChild(cells);
		        dom.appendChild(lmsParams);
		        
		        try {
		            Transformer tr = TransformerFactory.newInstance().newTransformer();
		            tr.setOutputProperty(OutputKeys.INDENT, "yes");
		            tr.setOutputProperty(OutputKeys.METHOD, "xml");
		            tr.setOutputProperty(OutputKeys.ENCODING, "UTF-8");
		            tr.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");
		            // send DOM to file
		            
		            FileOutputStream fos = new FileOutputStream(path+"LimeSegParams.xml"); 
		            tr.transform(new DOMSource(dom), new StreamResult(fos));
		            fos.close();

		        } catch (TransformerException te) {
		            System.out.println(te.getMessage());
		        } catch (IOException ioe) {
		            System.out.println(ioe.getMessage());
		        }
		    } catch (ParserConfigurationException pce) {
		        System.out.println("Save State: Error trying to instantiate DocumentBuilder " + pce);
		    }
	   }
	   
	   static void saveXmlFile(String path, Document dom) {
		   try {
	            Transformer tr = TransformerFactory.newInstance().newTransformer();
	            tr.setOutputProperty(OutputKeys.INDENT, "yes");
	            tr.setOutputProperty(OutputKeys.METHOD, "xml");
	            tr.setOutputProperty(OutputKeys.ENCODING, "UTF-8");
	            tr.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");
	            // send DOM to file
	            
	            FileOutputStream fos = new FileOutputStream(path); 
	            tr.transform(new DOMSource(dom), new StreamResult(fos));
	            fos.close();
	        } catch (TransformerException te) {
	            System.out.println(te.getMessage());
	        } catch (IOException ioe) {
	            System.out.println(ioe.getMessage());
	        }
	   }
	   
	   static public void saveStatev0p2(LimeSeg lms, String path_in) {
		   assert !lms.optimizerIsRunning;
		    if (!path_in.endsWith(File.separator)) {path_in=path_in+File.separator;}
		    String path = path_in;
		    // instance of a DocumentBuilderFactory
		    DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
		    try {
		    	
		    	File dir = new File(path);
		    	if (!dir.isDirectory()) {
		    		System.out.println("Erreur, given path is not a directory");
		    		return;
		    	}
		    	// By default removes all files in the folder
		    	// But ask for confirmation if the folder is not empty...
		    	if (dir.listFiles().length!=0) {
		    		System.out.println("Saving will remove the content of the folder "+path+" that contains "+dir.listFiles().length+" files and folders.");
		    		if (!askYesNo("Are you sure you want to proceed [Y/N] ?")) {return;}
		    	}
		    	purgeDirectory(dir,1);
		        DocumentBuilder db = dbf.newDocumentBuilder();
		        // create instance of DOM
		        Document dom = db.newDocument();
		        // create the optimizer parameter element
		        Element lmsParams = dom.createElement("LimeSegParameters");
		        Element versionInfo = dom.createElement("Version");
		        lmsParams.appendChild(versionInfo);
			    versionInfo.appendChild(dom.createTextNode("0.2"));
		        lmsParams.appendChild(getXmlOptimizerState(lms.opt, dom));
		        // TODO System.out.println("getXml3DViewerState needs to be implemented!");
		        // TODO Element imageInfos = dom.createElement("ImageParameters");
		        // lmsParams.appendChild(imageInfos);		        
		        dom.appendChild(lmsParams);
		        saveXmlFile(path+"LimeSegParams.xml",dom);
		        
		        
		        //lmsParams.appendChild(getXml3DViewerState(lms.jcr, dom));
		        //Element cells = dom.createElement("Cells");
		        lms.allCells.forEach(c->{
		        	// Element currCell = dom.createElement(c.id_Cell);
		        	// Cell Channel
			        Document domCell = db.newDocument();
			        Element cellParams = domCell.createElement("CellParameters");
				    Element channel = domCell.createElement("channel");
				    channel.appendChild(domCell.createTextNode(Integer.toString(c.cellChannel)));
				    cellParams.appendChild(channel);
		        	// Cell color
		        	Element color = domCell.createElement("color");		        	
				       Element r = domCell.createElement("R");
				       r.appendChild(domCell.createTextNode(Float.toString(c.color[0])));
				       color.appendChild(r);

				       Element g = domCell.createElement("G");
				       g.appendChild(domCell.createTextNode(Float.toString(c.color[1])));
				       color.appendChild(g);

				       Element b = domCell.createElement("B");
				       b.appendChild(domCell.createTextNode(Float.toString(c.color[2])));
				       color.appendChild(b);

				       Element a = domCell.createElement("A");
				       a.appendChild(domCell.createTextNode(Float.toString(c.color[3])));
				       color.appendChild(a);
				    cellParams.appendChild(color);
		        	//cells.appendChild(currCell);
		        	// Now writes all ply files for CellT object
				    String pathCell=path+File.separator+c.id_Cell+File.separator;			 		
	        	    File dirCell = new File(pathCell);
	        	    dirCell.mkdir(); // attempt to create the directory here
	        	    domCell.appendChild(cellParams);
			        saveXmlFile(pathCell+"CellParams.xml",domCell);
	        	    if (dirCell.exists()) {
			        	c.cellTs.forEach(ct -> { 
			        	   IOXmlPlyLimeSeg.saveCellTAsPly(ct,pathCell+"T_"+ct.frame+".ply");
			        	});
		        	}
		        });		        
		    } catch (ParserConfigurationException pce) {
		        System.out.println("Save State: Error trying to instantiate DocumentBuilder " + pce);
		    }
	   }
	   
	   static public void saveState(LimeSeg lms, String version, String path_in) {		   
		    assert !lms.optimizerIsRunning;
		    if (version=="0.1") {saveStatev0p1(lms,path_in);}
		    if (version=="0.2") {saveStatev0p2(lms,path_in);}		    
	   }
	   
	   static public boolean askYesNo(String question) {
	        return askYesNo(question, "[Y]", "[N]");
	   }

	   static public boolean askYesNo(String question, String positive, String negative) {
	        Scanner input = new Scanner(System.in);
	        // Convert everything to upper case for simplicity...
	        positive = positive.toUpperCase();
	        negative = negative.toUpperCase();
	        String answer;
	        do {
	            System.out.print(question);
	            answer = input.next().trim().toUpperCase();
	        } while (!answer.matches(positive) && !answer.matches(negative));
	        // Assess if we match a positive response
	        input.close();
	        return answer.matches(positive);
	   }
	    
	   static void purgeDirectory(File dir, int height) {
		   // no need to clean below level 
		   if (height>=0) {
		    for (File file: dir.listFiles()) {
		        if (file.isDirectory()) purgeDirectory(file, height-1);
		        file.delete();
		    }
		   }
		}

}




