package eu.kiaru.limeseg.gui;

import com.jogamp.newt.Display;
import com.jogamp.newt.NewtFactory;
import com.jogamp.newt.Screen;
import com.jogamp.opengl.GLAutoDrawable;
import com.jogamp.opengl.GLEventListener;
import com.jogamp.opengl.GLProfile;

import com.jogamp.newt.Window;
import com.jogamp.newt.event.KeyEvent;
import com.jogamp.newt.event.KeyListener;
import com.jogamp.newt.event.MouseEvent;
import com.jogamp.newt.event.MouseListener;
import com.jogamp.newt.opengl.GLWindow;
import com.jogamp.opengl.GL;
import static com.jogamp.opengl.GL.GL_ARRAY_BUFFER;
import static com.jogamp.opengl.GL.GL_COLOR_BUFFER_BIT;
import static com.jogamp.opengl.GL.GL_DEPTH_BUFFER_BIT;
import static com.jogamp.opengl.GL.GL_ELEMENT_ARRAY_BUFFER;
import static com.jogamp.opengl.GL.GL_FLOAT;
import static com.jogamp.opengl.GL.GL_INVALID_ENUM;
import static com.jogamp.opengl.GL.GL_INVALID_FRAMEBUFFER_OPERATION;
import static com.jogamp.opengl.GL.GL_INVALID_OPERATION;
import static com.jogamp.opengl.GL.GL_INVALID_VALUE;
import static com.jogamp.opengl.GL.GL_NO_ERROR;
import static com.jogamp.opengl.GL.GL_OUT_OF_MEMORY;
import static com.jogamp.opengl.GL.GL_POINTS;
import static com.jogamp.opengl.GL.GL_RENDERER;
import static com.jogamp.opengl.GL.GL_STATIC_DRAW;
import static com.jogamp.opengl.GL.GL_TRIANGLES;
import static com.jogamp.opengl.GL.GL_UNSIGNED_INT;
import static com.jogamp.opengl.GL.GL_VENDOR;
import static com.jogamp.opengl.GL.GL_VERSION;
import com.jogamp.opengl.GL2;
import static com.jogamp.opengl.GL2.GL_FRAGMENT_SHADER;
import static com.jogamp.opengl.GL2.GL_VERTEX_SHADER;
import com.jogamp.opengl.GLCapabilities;
import com.jogamp.opengl.util.FPSAnimator;
import com.jogamp.opengl.util.GLBuffers;
import com.jogamp.opengl.util.glsl.ShaderCode;
import com.jogamp.opengl.util.glsl.ShaderProgram;

import eu.kiaru.limeseg.LimeSeg;
import eu.kiaru.limeseg.struct.Cell;
import eu.kiaru.limeseg.struct.CellT;
import eu.kiaru.limeseg.struct.DotN;
import eu.kiaru.limeseg.struct.TriangleN;
import eu.kiaru.limeseg.struct.Vector3D;
import glm.mat._3.Mat3;
import glm.mat._4.Mat4;
import glm.vec._2.i.Vec2i;
import glm.vec._3.Vec3;
//import java.awt.image.BufferedImage;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;


import org.scijava.java3d.Canvas3D;
import org.scijava.java3d.utils.universe.SimpleUniverse;
/**
 * Custom 3D Viewer while waiting for SciView...
 * @author Nicolas Chiaruttini
 *
 */
public class JOGL3DCellRenderer implements GLEventListener, KeyListener, MouseListener {

    DotNColorSupplier colorSupplier = new DefaultDotNColorSupplier();
	/**
	 * "Zoom" of 3D Viewer
	 */
    public float RatioGlobal=0.001f;
    /**
     * Distortion of ZAxis in 3D viewer only 
     */
    public float ZScaleViewer=1f;
    /**
     * Rotation of the 3D Viewer viewpoint
     */
    public float view_rotx = 0.0f;
    public float view_roty = 0.0f;
    public float view_rotz = 0.0f;  
    int FPS=20;
    /**
     * Continuous rendering of 3D View {@link #FPS}
     */
    public FPSAnimator animator;
    /**
     * Points looked at by 3D Viewer
     */
    public Vector3D lookAt; 
    // View mode : no cut / cut above / cut below / cut above and below
    // With respect to the current z slice of the working image opened (IJ1)
    private boolean cutAbove=false;   
    private boolean cutBelow=false; 
    private boolean showActiveDots=false;
    public int cutAboveMask=0b00000001;
    public int cutBelowMask=0b00000010;
    public int showActiveDotsMask=0b00001000;
    
    private final int screenIdx = 0;
    // For user interaction
    private int prevMouseX, prevMouseY;
    // Cells being currently registered in the 3D Viewer
    private ArrayList<Cell> cellsToDisplay;
    /**
     * Window used by the 3D Viewer
     */
    public GLWindow glWindow;
    // Keep track of user position in working image of ImageJ1
    int CurrSlice, NSlices, NChannel, CurrFrame, CurrZSlice;
    // Window size
    protected Vec2i windowSize;
    // view Matrix
    private Mat4 viewMat4 = new Mat4();
    // Shaders
        private static final String SHADERS_SOURCE_PC = "flat-color";
        private static final String SHADERS_SOURCE_TR = "flat-color-tr";
        private static final String SHADERS_ROOT = "shader";
        private static final int BufferMAX = 3;
        private IntBuffer bufferName = GLBuffers.newDirectIntBuffer(BufferMAX);
        private int programName_PC;
        private int programName_TR;
    // For Dots representation
        // Shader communication
        protected int programHandle_PC;
        // Uniforms
        private int uniformMvp_PC;
        private int uniformNormalMatrix_PC;
        private int uniformDiffuse_PC;
        private int uniformPtThickness_PC;
        private int uniformLightDir_PC;
        private int uniformZSliceMin_PC, uniformZSliceMax_PC;
        private int uniformZMin_PC, uniformZMax_PC;
        // Per vertex
        int mNormalHandle_PC, mPositionHandle_PC, mColorHandle_PC;
        // Buffers handling
        // GPU
        private int numberOfFloatPerVertex_PC;
        private static final int idBufferVERTEX_PC = 0;        
        private int vertexBuffer_PC_Size;
        int numberOfDotsDisplayed_PC;
        // CPU
        float[] dataVertex_PC;
        int numberOfDotsInDataVertex_PC;
    // For triangles representation
        // Shader communication
        protected int programHandle_TR;
        // Uniforms
        private int uniformMvp_TR;
        private int uniformDiffuse_TR;
        private int uniformLightDir_TR;
        private int uniformZSliceMin_TR, uniformZSliceMax_TR;
        private int uniformZMin_TR, uniformZMax_TR;
        // Per vertex
        int mNormalHandle_TR, mPositionHandle_TR, mColorHandle_TR;
        // Buffers handling
        // GPU
        // vertex
        private int numberOfFloatPerVertex_TR;
        private static final int idBufferVERTEX_TR = 1;        
        private int vertexBuffer_TR_Size;
        int numberOfDotsDisplayed_TR;
        // triangles
        private int numberOfIntPerTriangle_TR;
        private static final int idBufferTRIANGLES_TR = 2;        
        private int trianglesBuffer_TR_Size;
        int numberOfTrianglesDisplayed_TR;
        // CPU
        float[] dataVertex_TR;
        int numberOfDotsInDataVertex_TR;
        int[] dataTriangles_TR;
        int numberOfTrianglesInDataTriangles_TR;
        // Display mode
        final static public int DOT_MODE=0;
        final static public int TRIANGLE_MODE_FLAT=1;    
        final static public int TRIANGLE_MODE_SMOOTH=2;
        final static public int DOT_MODE_LINEAR=3;
    
    //================================================================
  
    protected boolean createAndFillVertexBuffer( GL2 gl2) {      
        gl2.glDeleteBuffers(BufferMAX, bufferName);
        gl2.glGenBuffers(BufferMAX, bufferName);
        FloatBuffer tempFloatBuffer;
        IntBuffer tempIntBuffer;
        // Point cloud display
        vertexBuffer_PC_Size = numberOfDotsInDataVertex_PC * Float.BYTES * numberOfFloatPerVertex_PC; 
        if (dataVertex_PC==null) {dataVertex_PC = new float[0];}
        tempFloatBuffer = GLBuffers.newDirectFloatBuffer(dataVertex_PC);
        
        gl2.glBindBuffer(GL_ARRAY_BUFFER, bufferName.get(idBufferVERTEX_PC));
        gl2.glBufferData(GL_ARRAY_BUFFER, vertexBuffer_PC_Size, tempFloatBuffer, GL_STATIC_DRAW);
        gl2.glBindBuffer(GL_ARRAY_BUFFER, 0);
        tempFloatBuffer.clear();
        //BufferUtils.destroyDirectBuffer(tempFloatBuffer);
        numberOfDotsDisplayed_PC = numberOfDotsInDataVertex_PC;
        // Triangle display
        // Vertex buffer
        if ((numberOfDotsInDataVertex_TR>0)&&(numberOfTrianglesInDataTriangles_TR>0)) {            
            vertexBuffer_TR_Size = numberOfDotsInDataVertex_TR * Float.BYTES * numberOfFloatPerVertex_TR;
            tempFloatBuffer = GLBuffers.newDirectFloatBuffer(dataVertex_TR);
            gl2.glBindBuffer(GL_ARRAY_BUFFER, bufferName.get(idBufferVERTEX_TR));
            gl2.glBufferData(GL_ARRAY_BUFFER, vertexBuffer_TR_Size, tempFloatBuffer, GL_STATIC_DRAW);
            gl2.glBindBuffer(GL_ARRAY_BUFFER, 0);
            tempFloatBuffer.clear();
            //BufferUtils.destroyDirectBuffer(tempFloatBuffer);
            // Triangle buffer
            trianglesBuffer_TR_Size = numberOfTrianglesInDataTriangles_TR * Integer.BYTES * numberOfIntPerTriangle_TR;
            tempIntBuffer = GLBuffers.newDirectIntBuffer(dataTriangles_TR);
            gl2.glBindBuffer(GL_ARRAY_BUFFER, bufferName.get(idBufferTRIANGLES_TR));
            gl2.glBufferData(GL_ARRAY_BUFFER, trianglesBuffer_TR_Size, tempIntBuffer, GL_STATIC_DRAW);
            gl2.glBindBuffer(GL_ARRAY_BUFFER, 0);
            tempIntBuffer.clear();
            //BufferUtils.destroyDirectBuffer(tempIntBuffer);
            numberOfTrianglesDisplayed_TR=numberOfTrianglesInDataTriangles_TR;            
        } else {
            numberOfTrianglesDisplayed_TR=0;
        }
        return checkError(gl2, "initBuffer");        
    }

    public void launchAnim() { 
        Display display = NewtFactory.createDisplay(null);
        Screen screen = NewtFactory.createScreen(display, screenIdx);
        GLProfile glProfile = GLProfile.get(GLProfile.GL2);
        
        GLCapabilities glCapabilities = new GLCapabilities(glProfile);
        glWindow = GLWindow.create(screen, glCapabilities);
        assert glWindow != null;
        glWindow.setUndecorated(false);
        glWindow.setAlwaysOnTop(false);
        glWindow.setFullscreen(false);
        glWindow.setPointerVisible(true);
        glWindow.confinePointer(false);
        glWindow.setTitle("3D Display");
        windowSize = new Vec2i(800,800);
        glWindow.setSize(windowSize.x, windowSize.y);
        glWindow.setVisible(true);
        glWindow.addGLEventListener(this);
        glWindow.addMouseListener(this);
        glWindow.addKeyListener(this);
        animator = new FPSAnimator(FPS);
        animator.add(glWindow);
        animator.setExclusiveContext(true);
        animator.start();
    }

    public JOGL3DCellRenderer(){
    	Canvas3D dummyCanvas = new Canvas3D(SimpleUniverse.getPreferredConfiguration());
    	// How to load correclty the natives ?
    	// Steal what's done in Java 3D Viewer, which uses 
    	//final Image3DUniverse univ = new Image3DUniverse();
    	JOGLLoader.LoadNatives_GL2();// updating sci java to 19.0.0 solved this issue -> I thought so, but apparently not!
    	cellsToDisplay=new ArrayList<>();
    	if (lookAt==null) {lookAt=new Vector3D(0,0,0);}    	
    }

    @Override
    public final void init(GLAutoDrawable drawable) {
        GL gl = drawable.getGL().getGL2();
        begin(gl);
    }

    synchronized public final void addCellToDisplay(Cell c) {
        if (!this.cellsToDisplay.contains(c)) {
            this.cellsToDisplay.add(c);
        }
    }
    
    @Override
    public void keyPressed(KeyEvent ke) {
       int kc = ke.getKeyCode();
       float du,dv,dw;
       du=0;dv=0;dw=0;
        switch (kc) {
            case KeyEvent.VK_LEFT:
                du -= 1;
                break;
            case KeyEvent.VK_RIGHT:
                du += 1;
                break;
            case KeyEvent.VK_UP:
                dv -= 1;
                break;
            case KeyEvent.VK_DOWN:
                dv += 1;
                break;
            case KeyEvent.VK_P:
                dw += 1;
                break;
            case KeyEvent.VK_O:
                dw -= 1;
                break;
            default:
            break;
        }    
        float amplitude = 0.01f/RatioGlobal;
        Mat3 tr = new Mat3().identity()
                .rotate(-view_rotz, 0.f, 0.f, 1.f)                
                .rotate(-view_roty, 0.f, 1.f, 0.f)
                .rotate(-view_rotx, 1.f, 0.f, 0.f);
        Vec3 shift = tr.mul(new Vec3(du*amplitude,dv*amplitude,dw*amplitude));
        lookAt.x+=shift.x;
        lookAt.y+=shift.y;
        lookAt.z+=shift.z;
    }

    @Override
    public void keyReleased(KeyEvent ke) {
       // throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }  

    protected boolean end(GL gl) {
        GL2 gl2 = (GL2) gl;
        gl2.glDeleteBuffers(BufferMAX, bufferName);
        gl2.glDeleteProgram(programName_PC);
        gl2.glDeleteProgram(programName_TR);
        bufferName.clear();
        //BufferUtils.destroyDirectBuffer(bufferName);
        return true;
    }
    
    @Override
    public void reshape(GLAutoDrawable drawable, int x, int y, int width, int height) {
        GL gl = drawable.getGL().getGL2();
        windowSize.x=width;
        windowSize.y=height;
        gl.glViewport(0, 0, windowSize.x, windowSize.y);
    }

    @Override
    public final void dispose(GLAutoDrawable drawable) {
        GL gl = drawable.getGL().getGL2();
        assert end(gl);
    }
  
    public int getViewMode() {
        int returnValue = ((cutAbove==true)?1:0)*cutAboveMask+
                          ((cutBelow==true)?1:0)*cutBelowMask+
                          ((showActiveDots==true)?1:0)*showActiveDotsMask;
        return returnValue;
    }
    /**
     * Set 3D View mode:
     * 0 : Normal
     * 1 : Cut below current ZSlice
     * 2 : Cut above current ZSlice
     * 3 : Show only current ZSlice
     * 8 : As 0 but with dots being optimized (red = non converged, green = converged)
     * 9 : As 1 but with dots being optimized (red = non converged, green = converged)
     * 10 : As 2 but with dots being optimized (red = non converged, green = converged)
     * 11 : As 3 but with dots being optimized (red = non converged, green = converged)
     * @param value
     */
    public void setViewMode(int value) {
      cutAbove=((value&cutAboveMask)==cutAboveMask);
      cutBelow=((value&cutBelowMask)==cutBelowMask);
      showActiveDots=((value&showActiveDotsMask)==showActiveDotsMask);
      LimeSeg.notifyCellRendererCellsModif=true;
    }  
  
    protected boolean begin(GL gl) {
        GL2 gl2 = (GL2) gl;
        boolean validated = true;
        /*System.out.println("Vendor " + gl2.glGetString(GL_VENDOR));
        System.out.println("Renderer " + gl2.glGetString(GL_RENDERER));
        System.out.println("Version " + gl2.glGetString(GL_VERSION));
        System.out.println("Extensions " + gl2.glGetString(GL_EXTENSIONS));*/
        if (validated) {
            validated = initProgram(gl2);
        }
        if (validated) {
            validated = initBuffer(gl2);
        }
        return validated;
    }
  
    private boolean initProgram(GL2 gl2) {
        gl2.glEnable(GL2.GL_VERTEX_PROGRAM_POINT_SIZE);
        gl2.glEnable(GL.GL_DEPTH_TEST);
        gl2.glEnable(GL.GL_ALPHA);
        gl2.glEnable(GL2.GL_POINT_SPRITE);
        gl2.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);
        boolean validated = true;  
        // Program dots
        // Create program
        if (validated) {        	
            ShaderCode vertShader = ShaderCode.create(gl2, GL_VERTEX_SHADER, JOGL3DCellRenderer.class, "shader", "shader/bin",
            		SHADERS_SOURCE_PC, true);
            
            ShaderCode fragShader = ShaderCode.create(gl2, GL_FRAGMENT_SHADER, JOGL3DCellRenderer.class, "shader", "shader/bin",
            		SHADERS_SOURCE_PC, true);
            
            vertShader.defaultShaderCustomization(gl2, true, false);
            fragShader.defaultShaderCustomization(gl2, true, false);           
            ShaderProgram program = new ShaderProgram();
            program.add(vertShader);
            program.add(fragShader);
            program.init(gl2);            
            programName_PC = program.program();
            program.link(gl2, System.out);
            mPositionHandle_PC = gl2.glGetAttribLocation(programName_PC, "position");
            mNormalHandle_PC = gl2.glGetAttribLocation(programName_PC, "normal"); 
            mColorHandle_PC = gl2.glGetAttribLocation(programName_PC, "color");
        }
        // Get variables locations
        if (validated) {
            uniformMvp_PC = gl2.glGetUniformLocation(programName_PC, "mvp");
            uniformNormalMatrix_PC = gl2.glGetUniformLocation(programName_PC, "matrix_inverse_transpose");
            uniformDiffuse_PC = gl2.glGetUniformLocation(programName_PC, "diffuse");
            uniformPtThickness_PC = gl2.glGetUniformLocation(programName_PC, "ptThickness");
            uniformLightDir_PC = gl2.glGetUniformLocation(programName_PC, "lightDir");
            uniformZSliceMin_PC = gl2.glGetUniformLocation(programName_PC, "zSliceMin");
            uniformZSliceMax_PC = gl2.glGetUniformLocation(programName_PC, "zSliceMax");
            uniformZMin_PC = gl2.glGetUniformLocation(programName_PC, "zMin");
            uniformZMax_PC = gl2.glGetUniformLocation(programName_PC, "zMax");
        }
        // Set some variables 
        if (validated) {
            // Bind the program for use
            gl2.glUseProgram(programName_PC);
            // Set uniform value
            gl2.glUniform4fv(uniformDiffuse_PC, 1, new float[]{1f, .5f, .5f, 1f}, 0);
            Vector3D lightDir = new Vector3D(0.2f,0.2f,-1f);
            lightDir.normalize();
            gl2.glUniform3fv(uniformLightDir_PC, 1, new float[]{lightDir.x, lightDir.y, lightDir.z}, 0);
            // Unbind the program
            gl2.glUseProgram(0);
        }
        // Program triangles
        if (validated) {
            ShaderCode vertShader = ShaderCode.create(gl2, GL_VERTEX_SHADER, JOGL3DCellRenderer.class, "shader", "shader/bin",
            		SHADERS_SOURCE_TR, true);
            
            ShaderCode fragShader = ShaderCode.create(gl2, GL_FRAGMENT_SHADER, JOGL3DCellRenderer.class, "shader", "shader/bin",
            		SHADERS_SOURCE_TR, true);
            
            vertShader.defaultShaderCustomization(gl2, true, false);
            fragShader.defaultShaderCustomization(gl2, true, false);           
            ShaderProgram program = new ShaderProgram();
            program.add(vertShader);
            program.add(fragShader);
            program.init(gl2);            
            programName_TR = program.program();
            program.link(gl2, System.out);
            mPositionHandle_TR = gl2.glGetAttribLocation(programName_TR, "position");
            mNormalHandle_TR = gl2.glGetAttribLocation(programName_TR, "normal"); 
            mColorHandle_TR = gl2.glGetAttribLocation(programName_TR, "color");
        }
        // Get variables locations
        if (validated) {
            uniformMvp_TR = gl2.glGetUniformLocation(programName_TR, "mvp");
            uniformDiffuse_TR = gl2.glGetUniformLocation(programName_TR, "diffuse");
            uniformLightDir_TR = gl2.glGetUniformLocation(programName_TR, "lightDir");
            uniformZSliceMin_TR = gl2.glGetUniformLocation(programName_TR, "zSliceMin");
            uniformZSliceMax_TR = gl2.glGetUniformLocation(programName_TR, "zSliceMax");
            uniformZMin_TR = gl2.glGetUniformLocation(programName_TR, "zMin");
            uniformZMax_TR = gl2.glGetUniformLocation(programName_TR, "zMax");
        }
        // Set some variables 
        if (validated) {
            // Bind the program for use
            gl2.glUseProgram(programName_TR);
            // Set uniform value
            gl2.glUniform4fv(uniformDiffuse_TR, 1, new float[]{1f, .5f, .5f, 1f}, 0);
            Vector3D lightDir = new Vector3D(0.2f,0.2f,-1f);
            lightDir.normalize();
            gl2.glUniform3fv(uniformLightDir_TR, 1, new float[]{lightDir.x, lightDir.y, lightDir.z}, 0);
            // Unbind the program
            gl2.glUseProgram(0);
        }
        return validated & checkError(gl2, "initProgram");
    }
    
    protected boolean checkError(GL gl, String title) {
        int error = gl.glGetError();
        if (error != GL_NO_ERROR) {
            String errorString;
            switch (error) {
                case GL_INVALID_ENUM:
                    errorString = "GL_INVALID_ENUM";
                    break;
                case GL_INVALID_VALUE:
                    errorString = "GL_INVALID_VALUE";
                    break;
                case GL_INVALID_OPERATION:
                    errorString = "GL_INVALID_OPERATION";
                    break;
                case GL_INVALID_FRAMEBUFFER_OPERATION:
                    errorString = "GL_INVALID_FRAMEBUFFER_OPERATION";
                    break;
                case GL_OUT_OF_MEMORY:
                    errorString = "GL_OUT_OF_MEMORY";
                    break;
                default:
                    errorString = "UNKNOWN";
                    break;
            }
            System.out.println("OpenGL Error(" + errorString + "): " + title);
        }
        return error == GL_NO_ERROR;
    }

    private boolean initBuffer(GL2 gl2) {
        gl2.glGenBuffers(BufferMAX, bufferName);
        LimeSeg.notifyCellRendererCellsModif=true;
        createAndFillVertexBuffer(gl2);
        return checkError(gl2, "initBuffer");
    }
    
    synchronized public void fillBufferCellRenderer_TR() {
        // first thing : get the number of dots to be displayed
        int nDotsToBeDisplayed=0;
        int nTriToBeDisplayed=0;
        for (Cell c : this.cellsToDisplay) { // concurrent modif error
            CellT ct=c.getCellTAt(CurrFrame);
            if ((ct!=null)&&(ct.tesselated==true)&&(c.display_mode>0)) {
                nDotsToBeDisplayed+=ct.dots.size();
                nTriToBeDisplayed+=ct.triangles.size();
            }
        }
        int marginUp=2;
        int marginDown=6;
        numberOfFloatPerVertex_TR=2*3+4;// pos (3) and normal (3) + color (4)
        // dataVertex_TR
        if (dataVertex_TR==null) {
            // first allocation
            dataVertex_TR = new float[nDotsToBeDisplayed*marginUp*numberOfFloatPerVertex_TR];
        }
        if (dataVertex_TR.length>nDotsToBeDisplayed*numberOfFloatPerVertex_TR) {
            // no need to reallocate memory unless the difference is crazy (>MarginDown)
            if (dataVertex_TR.length>marginDown*nDotsToBeDisplayed*numberOfFloatPerVertex_TR) {
                dataVertex_TR = new float [nDotsToBeDisplayed*marginUp*numberOfFloatPerVertex_TR];
            }
        } else {
            // needs reallocation
            dataVertex_TR = new float [nDotsToBeDisplayed*marginUp*numberOfFloatPerVertex_TR];
        }
        numberOfDotsInDataVertex_TR = nDotsToBeDisplayed;
        // 
        numberOfIntPerTriangle_TR=3;
        if (dataTriangles_TR==null) {
            // first allocation
            dataTriangles_TR = new int[nTriToBeDisplayed*marginUp*this.numberOfIntPerTriangle_TR];
        }
        if (dataTriangles_TR.length>nTriToBeDisplayed*numberOfFloatPerVertex_TR) {
            // no need to reallocate memory unless the difference is crazy (>MarginDown)
            if (dataTriangles_TR.length>marginDown*nDotsToBeDisplayed*numberOfIntPerTriangle_TR) {
                dataTriangles_TR = new int [nTriToBeDisplayed*marginUp*numberOfIntPerTriangle_TR];
            }
        } else {
            // needs reallocation
            dataTriangles_TR = new int [nTriToBeDisplayed*marginUp*numberOfIntPerTriangle_TR];
        }
        this.numberOfTrianglesInDataTriangles_TR = nTriToBeDisplayed;
        int indexPt=0;
        int indexTri=0;
        float rnp,gnp,bnp,anp; // RGBA plane
        int indexPtCellStart=0;
        for (Cell c : this.cellsToDisplay) { // concurrent modification error!
          float[] currentcolor;
          CellT ct=c.getCellTAt(CurrFrame);
          float nx, ny, nz;
          float px, py, pz;
          if ((ct!=null)&&(ct.tesselated==true)&&(c.display_mode>0)) {
              indexPtCellStart=indexPt/this.numberOfFloatPerVertex_TR;
              for (DotN dot : ct.dots) {
                  currentcolor = this.colorSupplier.getColor(dot);
                  nx=dot.Norm.x;ny=dot.Norm.y;nz=dot.Norm.z;
                  px=dot.pos.x;py=dot.pos.y;pz=dot.pos.z;
                  //Caused by: java.lang.ArrayIndexOutOfBoundsException: 
                  //at eu.kiaru.limeseg.gui.JOGL3DCellRenderer.fillBufferCellRenderer_TR(JOGL3DCellRenderer.java:548)
                  dataVertex_TR[indexPt++]=px;
                  dataVertex_TR[indexPt++]=py;
                  dataVertex_TR[indexPt++]=pz;   
                  dataVertex_TR[indexPt++]=nx;
                  dataVertex_TR[indexPt++]=ny;
                  dataVertex_TR[indexPt++]=nz;
                  dataVertex_TR[indexPt++]=currentcolor[0];//(float)((java.lang.Math.cos((px*java.lang.Math.cos(py/25.))/25.)+1.)/2.);//java.lang.Math.random();//rnp;
                  dataVertex_TR[indexPt++]=currentcolor[1];//(float)((java.lang.Math.cos((px*pz)/2500.)+1.)/2.);//(float)java.lang.Math.random();//gnp;
                  dataVertex_TR[indexPt++]=currentcolor[2];//(float)((java.lang.Math.cos((py+pz)/25.)+1.)/2.);//(float)java.lang.Math.random();//bnp;
                  dataVertex_TR[indexPt++]=currentcolor[3];
              }
              for (TriangleN tr : ct.triangles) {
                  dataTriangles_TR[indexTri++]=tr.id1+indexPtCellStart;
                  dataTriangles_TR[indexTri++]=tr.id2+indexPtCellStart;
                  dataTriangles_TR[indexTri++]=tr.id3+indexPtCellStart;
              }
          }
      }  
      LimeSeg.setBuffFilled(true);
    }

    // Called during optimization
    synchronized public void fillBufferCellRenderer_PC(ArrayList<DotN> aList) {
        int nDotsToBeDisplayed=aList.size();        
        int marginUp=2;
        int marginDown=6;
        numberOfFloatPerVertex_PC=2*3+4;// pos (3) and normal (3) + color (4)
        if (dataVertex_PC==null) {
            // first allocation
            dataVertex_PC = new float[nDotsToBeDisplayed*marginUp*numberOfFloatPerVertex_PC];
        }
        if (dataVertex_PC.length>nDotsToBeDisplayed*numberOfFloatPerVertex_PC) {
            // no need to reallocate memory unless the difference is crazy (>MarginDown)
            if (dataVertex_PC.length>marginDown*nDotsToBeDisplayed*numberOfFloatPerVertex_PC) {
                dataVertex_PC = new float [nDotsToBeDisplayed*marginUp*numberOfFloatPerVertex_PC];
            }
        } else {
            // needs reallocation
            dataVertex_PC = new float [nDotsToBeDisplayed*marginUp*numberOfFloatPerVertex_PC];
        }
        numberOfDotsInDataVertex_PC= nDotsToBeDisplayed;
        int index=0;
        float rnp,gnp,bnp,anp; // RGBA plane        
        rnp=0.7f;                    
        gnp=0.2f;                    
        bnp=0.1f;                    
        anp=1f;
        float nx, ny, nz;
        float px, py, pz; 
        for (DotN dot : aList) {              
                  nx=dot.Norm.x;ny=dot.Norm.y;nz=dot.Norm.z;
                  px=dot.pos.x;py=dot.pos.y;pz=dot.pos.z;
                  dataVertex_PC[index++]=px;
                  dataVertex_PC[index++]=py;
                  dataVertex_PC[index++]=pz;   
                  dataVertex_PC[index++]=nx;
                  dataVertex_PC[index++]=ny;
                  dataVertex_PC[index++]=nz;
                  if (dot.allNeighborsHaveConvergedPreviously) {
                    dataVertex_PC[index++]=rnp;
                    dataVertex_PC[index++]=1f;//gnp;
                    dataVertex_PC[index++]=bnp;
                    dataVertex_PC[index++]=anp;
                  } else {
                    dataVertex_PC[index++]=rnp;
                    dataVertex_PC[index++]=gnp;
                    dataVertex_PC[index++]=bnp;
                    dataVertex_PC[index++]=anp;  
                  }
      }  
      LimeSeg.setBuffFilled(true);
    }
    
    synchronized public void fillBufferCellRenderer_PC() {
        // first thing : get the number of dots to be displayed
        int nDotsToBeDisplayed=0;
        for (Cell c : this.cellsToDisplay) { // concurrent modif error
            CellT ct=c.getCellTAt(CurrFrame);
            if ((ct!=null)&&((ct.tesselated==false)||(c.display_mode==0))) {
                nDotsToBeDisplayed+=ct.dots.size();
            }
        }
        int marginUp=2;
        int marginDown=6;
        numberOfFloatPerVertex_PC=2*3+4;// pos (3) and normal (3) + color (4)
        if (dataVertex_PC==null) {
            // first allocation
            dataVertex_PC = new float[nDotsToBeDisplayed*marginUp*numberOfFloatPerVertex_PC];
        }
        if (dataVertex_PC.length>nDotsToBeDisplayed*numberOfFloatPerVertex_PC) {
            // no need to reallocate memory unless the difference is crazy (>MarginDown)
            if (dataVertex_PC.length>marginDown*nDotsToBeDisplayed*numberOfFloatPerVertex_PC) {
                dataVertex_PC = new float [nDotsToBeDisplayed*marginUp*numberOfFloatPerVertex_PC];
            }
        } else {
            // needs reallocation
            dataVertex_PC = new float [nDotsToBeDisplayed*marginUp*numberOfFloatPerVertex_PC];
        }
        numberOfDotsInDataVertex_PC = nDotsToBeDisplayed;
        int index=0;
        //float rnp,gnp,bnp,anp; // RGBA plane
        for (Cell c : this.cellsToDisplay) {
          float[] currentcolor;
          CellT ct=c.getCellTAt(CurrFrame);
          float nx, ny, nz;
          float px, py, pz;
          if ((ct!=null)&&((ct.tesselated==false)||(c.display_mode==0))) {
              for (DotN dot : ct.dots) {
                  currentcolor = this.colorSupplier.getColor(dot);
                  nx=dot.Norm.x;ny=dot.Norm.y;nz=dot.Norm.z;
                  px=dot.pos.x;py=dot.pos.y;pz=dot.pos.z;
                  dataVertex_PC[index++]=px; // java.lang.ArrayIndexOutOfBoundsException: 61660
                  dataVertex_PC[index++]=py;
                  dataVertex_PC[index++]=pz;   
                  dataVertex_PC[index++]=nx;
                  dataVertex_PC[index++]=ny;
                  dataVertex_PC[index++]=nz;
                  dataVertex_PC[index++]=currentcolor[0];//rnp;
                  dataVertex_PC[index++]=currentcolor[1];//gnp;
                  dataVertex_PC[index++]=currentcolor[2];//bnp;
                  dataVertex_PC[index++]=currentcolor[3];//anp;
              }
          }
      }
      LimeSeg.setBuffFilled(true);
    }
  
    synchronized public void clearDisplayedCells() {
        if (this.cellsToDisplay!=null) {
            this.cellsToDisplay.clear();
        }   
    }
    
    synchronized public void removeDisplayedCell(Cell c) {
    	if (this.cellsToDisplay!=null) {
            this.cellsToDisplay.remove(c);
        }   
    }
    
    @Override  
    public void display(GLAutoDrawable drawable) {     
        // Fetch points data if changed
        boolean frameHasChanged;
        if (LimeSeg.workingImP!=null) {
            CurrSlice=LimeSeg.workingImP.getCurrentSlice();
            NSlices=LimeSeg.workingImP.getNSlices();
            NChannel=LimeSeg.workingImP.getNChannels();
            frameHasChanged = (CurrFrame!=((int)(CurrSlice-1)/(int)(NSlices*NChannel))+1);
            CurrFrame=((int)(CurrSlice-1)/(int)(NSlices*NChannel))+1;
            CurrZSlice=LimeSeg.workingImP.getSlice();
        } else {
            CurrSlice=1;NSlices=1;NChannel=1;CurrFrame=1;CurrZSlice=1;
            frameHasChanged=false;
        }
          
        if ((LimeSeg.notifyCellRendererCellsModif || frameHasChanged)&&(!LimeSeg.getBuffFilled())) {
        	LimeSeg.requestFillBufferCellRenderer();
        }
       
        float sizeDisk=RatioGlobal*LimeSeg.opt.d_0*(windowSize.x+windowSize.y)*0.7f/1.5f;
          
        // Get the GL corresponding to the drawable we are animating
        GL2 gl2 = drawable.getGL().getGL2();
          
        if (LimeSeg.getBuffFilled()) {            
            createAndFillVertexBuffer(gl2); // Puts the buffer up to date in the display
            LimeSeg.setBuffFilled(false); 		// Fills vertex since the buffer has changed
        } 

        // Compute the MVP (Model View Projection matrix)
        Mat4 projection = glm.Glm.perspective_((float) Math.PI * 0.25f, (float)windowSize.x / (float)windowSize.y, 0.05f, 200f);
       
        Mat4 model = new Mat4().identity().scale(-RatioGlobal, -RatioGlobal, RatioGlobal) // translation*rotation*scale
                .rotateX(view_rotx)
                .rotateY(view_roty)
                .rotateZ(view_rotz)
                .translate(-lookAt.x, -lookAt.y, -lookAt.z)
                .scale(1f, 1f, ZScaleViewer);
        

        viewMat4 = Mat4.lookAt(new Vec3(0,0,-1), new Vec3(0,0,0), new Vec3(0,1,0), viewMat4);

        Mat4 mv=viewMat4.mul(model);
        
        Mat4 normalMatrix = new Mat4();
        inverse(mv,normalMatrix);
        normalMatrix.transpose();
        
        Mat4 mvp = projection.mul(mv);
        
        gl2.glViewport(0, 0, windowSize.x, windowSize.y);
        // Clear color buffer with white
        gl2.glClearColor(1f, 1f, 1f, 1f);
        gl2.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // Bind program
        gl2.glUseProgram(programName_PC);
        // Set the value of MVP uniform.
        gl2.glUniformMatrix4fv(uniformMvp_PC, 1, false, mvp.toFa_(), 0);
        gl2.glUniformMatrix4fv(this.uniformNormalMatrix_PC, 1, false, normalMatrix.toFa_(), 0);
        // Set the value of Uniforms
        gl2.glUniform1f(uniformPtThickness_PC, sizeDisk);//uniformPtThickness
        gl2.glUniform1f(uniformZSliceMin_PC, (float) ((CurrZSlice-1)*LimeSeg.opt.getOptParam("ZScale")));
        gl2.glUniform1f(uniformZSliceMax_PC, (float) ((CurrZSlice)*LimeSeg.opt.getOptParam("ZScale")));
        if (cutAbove) {
            gl2.glUniform1f(uniformZMin_PC, (float) ((CurrZSlice-1)*LimeSeg.opt.getOptParam("ZScale")));
        } else {
            gl2.glUniform1f(uniformZMin_PC, (float) (-Float.MAX_VALUE));
        }
        if (cutBelow) {
            gl2.glUniform1f(uniformZMax_PC, (float) ((CurrZSlice)*LimeSeg.opt.getOptParam("ZScale")));
        } else {
            gl2.glUniform1f(uniformZMax_PC, (float) (Float.MAX_VALUE));
        }
        // Bind buffer of positions
        gl2.glBindBuffer(GL_ARRAY_BUFFER, bufferName.get(idBufferVERTEX_PC));
        gl2.glVertexAttribPointer(mPositionHandle_PC, 3, GL_FLOAT, false, numberOfFloatPerVertex_PC*Float.BYTES, 0);
        gl2.glVertexAttribPointer(mNormalHandle_PC, 3, GL_FLOAT, false, numberOfFloatPerVertex_PC*Float.BYTES, 3*Float.BYTES); // offset by position
        gl2.glVertexAttribPointer(mColorHandle_PC, 4, GL_FLOAT, false, numberOfFloatPerVertex_PC*Float.BYTES, 6*Float.BYTES); // offset by position
        gl2.glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        gl2.glEnableVertexAttribArray(mPositionHandle_PC);
        gl2.glEnableVertexAttribArray(mNormalHandle_PC);
        gl2.glEnableVertexAttribArray(mColorHandle_PC);
        gl2.glDrawArrays(GL_POINTS, 0, this.numberOfDotsDisplayed_PC);
        gl2.glDisableVertexAttribArray(mPositionHandle_PC);
        gl2.glDisableVertexAttribArray(mNormalHandle_PC);
        gl2.glDisableVertexAttribArray(mColorHandle_PC);
        gl2.glUseProgram(0);
        
        if ((getViewMode()<8)&&(!LimeSeg.optimizerIsRunning)) {
	        gl2.glUseProgram(programName_TR);
	        // Set the value of MVP uniform.
	        gl2.glUniformMatrix4fv(uniformMvp_TR, 1, false, mvp.toFa_(), 0);
	        // Set the value of Uniforms
	        gl2.glUniform1f(uniformZSliceMin_TR, (float) ((CurrZSlice-1)*LimeSeg.opt.getOptParam("ZScale")));
	        gl2.glUniform1f(uniformZSliceMax_TR, (float) ((CurrZSlice)*LimeSeg.opt.getOptParam("ZScale")));
	        if (cutAbove) {
	            gl2.glUniform1f(uniformZMin_TR, (float) ((CurrZSlice-1)*LimeSeg.opt.getOptParam("ZScale")));
	        } else {
	            gl2.glUniform1f(uniformZMin_TR, (float) (-Float.MAX_VALUE));
	        }
	        if (cutBelow) {
	            gl2.glUniform1f(uniformZMax_TR, (float) ((CurrZSlice)*LimeSeg.opt.getOptParam("ZScale")));
	        } else {
	            gl2.glUniform1f(uniformZMax_TR, (float) (Float.MAX_VALUE));
	        }
	        // Bind buffer of positions
	        gl2.glBindBuffer(GL_ARRAY_BUFFER, bufferName.get(idBufferVERTEX_TR));
	        gl2.glVertexAttribPointer(mPositionHandle_TR, 3, GL_FLOAT, false, numberOfFloatPerVertex_TR*Float.BYTES, 0);
	        gl2.glVertexAttribPointer(mNormalHandle_TR, 3, GL_FLOAT, false, numberOfFloatPerVertex_TR*Float.BYTES, 3*Float.BYTES); // offset by position
	        gl2.glVertexAttribPointer(mColorHandle_TR, 4, GL_FLOAT, false, numberOfFloatPerVertex_TR*Float.BYTES, 6*Float.BYTES); // offset by position
	        gl2.glBindBuffer(GL_ARRAY_BUFFER, 0);
	        gl2.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferName.get(idBufferTRIANGLES_TR));        
	        gl2.glEnableVertexAttribArray(mPositionHandle_TR);
	        gl2.glEnableVertexAttribArray(mNormalHandle_TR);
	        gl2.glEnableVertexAttribArray(mColorHandle_TR);
	        gl2.glDrawElements(GL_TRIANGLES, 3*numberOfTrianglesDisplayed_TR, GL_UNSIGNED_INT, 0);
	        gl2.glDisableVertexAttribArray(mPositionHandle_TR);
	        gl2.glDisableVertexAttribArray(mNormalHandle_TR);
	        gl2.glDisableVertexAttribArray(mColorHandle_TR);
	        gl2.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        }
        
        gl2.glUseProgram(0);
    }

    @Override
    public void mouseWheelMoved(MouseEvent me) {
        RatioGlobal*=(8f+me.getRotation()[1])/8f;
    }
  

    @Override
    public void mousePressed(MouseEvent e) {
        prevMouseX = e.getX();
        prevMouseY = e.getY();
    }

        
    @Override
	public void mouseReleased(MouseEvent e) {
    }

    @Override
	public void mouseDragged(MouseEvent e) {
        final int x = e.getX();
        final int y = e.getY();
        int width=0, height=0;
        Object source = e.getSource();
        if(source instanceof Window) {
            Window window = (Window) source;
            width=window.getSurfaceWidth();
            height=window.getSurfaceHeight();
        } else if(source instanceof GLAutoDrawable) {
        	GLAutoDrawable glad = (GLAutoDrawable) source;
            width=glad.getSurfaceWidth();
            height=glad.getSurfaceHeight();
        } else if (GLProfile.isAWTAvailable() && source instanceof java.awt.Component) {
            java.awt.Component comp = (java.awt.Component) source;
            width=comp.getWidth();
            height=comp.getHeight();
        } else {
            throw new RuntimeException("Event source neither Window nor Component: "+source);
        }
        float thetaY = 6.14159f*( (float)(x-prevMouseX)/(float)width);
        float thetaX = 6.14159f*( (float)(prevMouseY-y)/(float)height);

        prevMouseX = x;
        prevMouseY = y;

        view_rotx -= thetaX;
        view_roty -= thetaY;
      }

      @Override
      public void mouseClicked(MouseEvent me) {
           // throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
      }

      @Override
      public void mouseEntered(MouseEvent me) {
           // throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
      }

      @Override
      public void mouseExited(MouseEvent me) {
         // throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
      }

      @Override
      public void mouseMoved(MouseEvent me) {
            // throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
      }
      
      public static Mat4 inverse(Mat4 src, Mat4 dest) {

    	    float m00=src.m00, m10=src.m10, m20=src.m20, m30=src.m30;
    	    float m01=src.m01, m11=src.m11, m21=src.m21, m31=src.m31;
    	    float m02=src.m02, m12=src.m12, m22=src.m22, m32=src.m32;
    	    float m03=src.m03, m13=src.m13, m23=src.m23, m33=src.m33;
          float a = m00 * m11 - m01 * m10;
          float b = m00 * m12 - m02 * m10;
          float c = m00 * m13 - m03 * m10;
          float d = m01 * m12 - m02 * m11;
          float e = m01 * m13 - m03 * m11;
          float f = m02 * m13 - m03 * m12;
          float g = m20 * m31 - m21 * m30;
          float h = m20 * m32 - m22 * m30;
          float i = m20 * m33 - m23 * m30;
          float j = m21 * m32 - m22 * m31;
          float k = m21 * m33 - m23 * m31;
          float l = m22 * m33 - m23 * m32;
          float det = a * l - b * k + c * j + d * i - e * h + f * g;
          det = 1.0f / det;
          dest.set(
                  (+m11 * l - m12 * k + m13 * j) * det,
                  (-m01 * l + m02 * k - m03 * j) * det,
                  (+m31 * f - m32 * e + m33 * d) * det,
                  (-m21 * f + m22 * e - m23 * d) * det,
                  (-m10 * l + m12 * i - m13 * h) * det,
                  (+m00 * l - m02 * i + m03 * h) * det,
                  (-m30 * f + m32 * c - m33 * b) * det,
                  (+m20 * f - m22 * c + m23 * b) * det,
                  (+m10 * k - m11 * i + m13 * g) * det,
                  (-m00 * k + m01 * i - m03 * g) * det,
                  (+m30 * e - m31 * c + m33 * a) * det,
                  (-m20 * e + m21 * c - m23 * a) * det,
                  (-m10 * j + m11 * h - m12 * g) * det,
                  (+m00 * j - m01 * h + m02 * g) * det,
                  (-m30 * d + m31 * b - m32 * a) * det,
                  (+m20 * d - m21 * b + m22 * a) * det);
          return dest;
      } 
      
 
}