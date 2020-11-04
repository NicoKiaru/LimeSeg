package eu.kiaru.limeseg.gui;

import java.util.concurrent.ExecutionException;

import org.scijava.AbstractGateway;

//import ij.IJ;
import net.imagej.ImageJ;
/**
 * Hacky script to load JOGL Natives for 3D Viewer
 * Script needs to be launched in a separate Thread
 * See this thread:
 * - http://forum.imagej.net/t/fiji-command-unable-to-find-jogl-library-minimal-example/6484/29
 * @author Nicolas Chiaruttini
 *
 */
public class JOGLLoader {
	
	private static boolean NativesHaveBeenLoaded=false;	

	private static final String groovyScriptLoadJOGL_GL2 = 
		"import com.jogamp.newt.NewtFactory \n"+
		"import com.jogamp.newt.opengl.GLWindow \n"+
		"import com.jogamp.opengl.GL \n"+
		"import com.jogamp.opengl.GL2 \n"+
		"import com.jogamp.opengl.GLAutoDrawable \n"+
		"import com.jogamp.opengl.GLCapabilities \n"+
		"import com.jogamp.opengl.GLProfile \n"+
		"glProfile = GLProfile.get(GLProfile.GL2) \n"+
		"glCapabilities = new GLCapabilities(glProfile) \n";
	
	public static void LoadNatives_GL2() {
		if (!NativesHaveBeenLoaded) {
			//Natives have not been loaded...
			// This needs to be executed from a separate Thread...
			//Groovy script hack!
			// -> this execution manages to load correctly JOGL natives
			//try {    		
				//staticContext
				//org.scijava.Context ijInstance = ImageJ.getContext(0);//IJ.getInstance();//(ImageJ) (Context.getCurrentContext());//ij.IJ.getInstance();//new ImageJ();
				//net.imagej.ImageJ.
				//Context.getCurrentContext();
		        //ijInstance.script().run("foo.groovy", groovyScriptLoadJOGL_GL2, true).get();
		        NativesHaveBeenLoaded=true;
			//} catch (InterruptedException | ExecutionException e) {
				// TODO Auto-generated catch block
			//	e.printStackTrace();
			//}
			// End of script execution, natives have been loaded
		} else {
			// Natives have already been loaded
		}
	}
}
