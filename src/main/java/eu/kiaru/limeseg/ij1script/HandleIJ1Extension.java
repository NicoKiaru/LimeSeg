package eu.kiaru.limeseg.ij1script;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;

import eu.kiaru.limeseg.LimeSeg;
import ij.macro.ExtensionDescriptor;
import ij.macro.MacroExtension;
import ij.plugin.frame.Recorder;
/**
 * Handle IJ1 macro extension function via their annotations {@link IJ1ScriptableMethod}
 * @author Nicolas Chiaruttini
 *
 */
public class HandleIJ1Extension implements MacroExtension {
	static ArrayList<Class> classesToRegister = new ArrayList<>();
	
	static public void addAClass(Class myClass) {
		boolean notAlreadyHere=true;
		for (Class c:classesToRegister) {
			if (c.equals(myClass)) {
				notAlreadyHere=false;
			}
		}
		if (notAlreadyHere) classesToRegister.add(myClass);
	}
	
	@Override
	public ExtensionDescriptor[] getExtensionFunctions() {
    	ArrayList<ExtensionDescriptor> edlist = new ArrayList<>();
    	int maxNParams=4;
    	int[] paramType = new int[4];
    	for (Class c:classesToRegister)
    	for (Method m:c.getMethods()) {
        	if (m.isAnnotationPresent(IJ1ScriptableMethod.class)) {
        		System.out.println(m.getName()+" is Scriptable.");
        		int nParameters = m.getParameterTypes().length;
        		if ((nParameters>maxNParams)&&(Modifier.isStatic(m.getModifiers()))) {
        			//IJ.log(m.getName()+" cannot be made scriptable in IJ1 since it has too many parameters or is non static.");
        		} else {        			
        			for (int i=0;i<nParameters;i++) {
        				Class p = m.getParameterTypes()[i];
        				//System.out.println(p.getName());
        				if   ((p.getName().equals("float"))
        					||(p.getName().equals("double"))
        					||(p.getName().equals("int"))
        					||(p.getName().equals("java.lang.Float"))
        					||(p.getName().equals("java.lang.Double"))
            				||(p.getName().equals("java.lang.Integer"))){
        					paramType[i]=ARG_NUMBER;
        				}
        				if  ((p.getName().equals("[Ljava.lang.Double;"))){
        					paramType[i]=ARG_NUMBER+ARG_OUTPUT;
            			}
        				if (p.getName().equals("java.lang.String")) {
        					paramType[i]=ARG_STRING;//+ARG_OUTPUT; 
        					// cannot differentiate on the basis of the class if it is an input or an output
        					// pretty annoying if we force ARG_OUTPUT since it requires the input argument to be a variable
        				}
        				//System.out.println("---- param["+i+"]="+paramType[i]);
        			}
        			switch (nParameters) {
        				case 0:
        					edlist.add(ExtensionDescriptor.newDescriptor(m.getName(), this));
        					break;
        				case 1:
        					edlist.add(ExtensionDescriptor.newDescriptor(m.getName(), this, paramType[0]));
                    		break;
        				case 2:
        					edlist.add(ExtensionDescriptor.newDescriptor(m.getName(), this, paramType[0], paramType[1]));
                    		break;
        				case 3:
        					edlist.add(ExtensionDescriptor.newDescriptor(m.getName(), this, paramType[0], paramType[1], paramType[2]));
                    		break;
        				case 4:
        					edlist.add(ExtensionDescriptor.newDescriptor(m.getName(), this, paramType[0], paramType[1], paramType[2], paramType[3]));
                    		break;
        			}        			
        		}
        	}
        }
    	ExtensionDescriptor[] edarray = new ExtensionDescriptor[edlist.size()];
    	return edlist.toArray(edarray);
	}
	
	@Override
	public String handleExtension(String name, Object[] args) {
		staticHandleExtension(name, args);
		return null;
	}

	static public String staticHandleExtension(String name, Object[] args) {
		boolean hasBeenFound=false;
        for (Method m:LimeSeg.class.getMethods()) {
        	if (m.isAnnotationPresent(IJ1ScriptableMethod.class)) {
	        	if (m.getName().equals(name)&&(!hasBeenFound)) {
	        		//System.out.println("m.getName()="+m.getName());
	        		hasBeenFound=true; 
	        		try {
	        			// Argument conversion for IJ1 extension handling
	        			if (args!=null) {
	        				int count=0;
		        			for (Object o:args) {
		        				Class classSrc = o.getClass();		        				
		        				Class classDst = m.getParameterTypes()[count];
		        				try {
		        					classDst.cast(o);
		        					// No exception -> no problem
		        				} catch (ClassCastException e) {
		        					// Class cast exception -> conversion required
	           						if  (
	           							((classDst.equals( Float.class))||(classDst.equals( float.class)))
	           							&&
	           							((classSrc.equals(Double.class))||(classSrc.equals(double.class)))
	           							) {
	           							args[count]=new Float(((Double)o).floatValue());	           							
	           						}
	           						if  (
		           							((classDst.equals(Integer.class))||(classDst.equals(int.class)))
		           							&&
		           							((classSrc.equals(Double.class))||(classSrc.equals(double.class)))
		           							) {
		           							args[count]=new Integer(((Double)o).intValue());	           							
		           					}
		        				}
		        				count++;
	                    	}	   
	        			}
	        			//--------------------------------------------
	        			if (m.getAnnotation(IJ1ScriptableMethod.class).newThread()) {
	        				Thread p = new Thread(new Runnable() {				
	                        		@Override
	                        		public void run() {
	                        			try {
	                        			m.invoke(null, args); //static method
	                        			} catch (IllegalAccessException | IllegalArgumentException | InvocationTargetException | SecurityException e) {
	                	        			e.printStackTrace();
	                	        		}
	                        		}
            					}
                        	);
                        	p.start();      
	        			} else {
	        				m.invoke(null, args); //static method
	        			}
	        			//--------------------------------------------
	        			// Recorder handling
                    	if ((Recorder.record)) {
                    		String strArgs="";
                    		if (args!=null) {
	                    		for (Object o:args) {
	                    			if (o.getClass()==String.class) {
	                    				strArgs+="'"+o+"',";	
	                    			} else {
		                    			strArgs+=o+",";	
	                    			}
	                    		}
	                    		if (args.length>0) {
	                    			strArgs=strArgs.substring(0,strArgs.length()-1);
	                    		}
                    		}
                    		String cmd="Ext."+m.getName()+"("+strArgs+");\n";
                    		Recorder.recordString(cmd);                    		
                    	}	        	
                    	//-------------------------------------------
	        		} catch (IllegalAccessException | IllegalArgumentException | InvocationTargetException | SecurityException e) {
	        			e.printStackTrace();
	        		}
	        	}
        	}
        }
        return null;
	}

}
