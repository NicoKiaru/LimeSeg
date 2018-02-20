package eu.kiaru.limeseg.struct;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import eu.kiaru.limeseg.LimeSeg;
import eu.kiaru.limeseg.opt.Optimizer;

/** 
 * Represents a Cell Timepoint. This means it basically represents a single 3D entity
 * @author Nicolas Chiaruttini
 * @see Cell
 */
public class CellT implements Cloneable{
    /**
     * Set of dots representing the cell timepoint
     * @see DotN
     */
    public ArrayList<DotN> dots = new ArrayList<>();
    /**
     * Center of mass of the cell. Needs to be updated with the updateCenter() method
     */
    public Vector3D center = new Vector3D(0,0,0);
    /**
     * Static (unique?) identifier number for the cell timepoint
     * used to check if interactions between dots are repulsive or attractive
     */                          
    public int idInt;
    /**
     * Counter for static identifier of the cell timepoint
     */
    static public int idIntStatic;
    /**
     * Parent cell to which this cell timepoint belongs
     */
    public Cell c;
    /**
     * frame identifier of this cell timepoint
     */
    public int frame;    
    /**
     * 
     */
    public int flag_stored=-1;
    /**
     * Notifies a modification of the cell timepoint due to optimization
     * Notifies to ... who ?
     */
    public boolean modified = true;
    /**
     * list of triangles if the surface has been tesselated
     */
    public ArrayList<TriangleN> triangles = new ArrayList<>();
    /**
     * Number of freeEdges found after tesselation
     * Should be equal to zero if the surface is closed
     */
    public int freeEdges;
    /**
     * tells is this cell timepoint has been tesselated
     */    
    public boolean tesselated = false;
    /**
     * Number of identified independent volumes after tesselation 
     */
    public int nVolume=-1;
    /**
     * Constructor of a cell timepoint
     * @param c_in Cell to which this cell timepoint is attached
     * @param frame_in Frame number (IJ1 convention) to which this cell timpeoint is attached
     */
    public CellT(Cell c_in, int frame_in) {
        c=c_in;
        frame=frame_in;
        c.modified=true;
        idIntStatic++;
        idInt=idIntStatic;
    }
    /**
     * If the cell timepoint has been tesselated
     * @return the volume of this cell timepoint
     */
    public double getVolume() {
        double VolumeTot=-1;
        if (this.tesselated) {
            this.updateCenter();
            VolumeTot=0;
            // volume of cone is approximated by r2h            
            // 1/3!*|A.(BxC)|
            for (int i=0;i<triangles.size();i++){
                TriangleN tri = triangles.get(i);
                Vector3D v1=this.dots.get(tri.id1).pos;                
                Vector3D v2=this.dots.get(tri.id2).pos;//tri.d2.pos;                
                Vector3D v3=this.dots.get(tri.id3).pos;//tri.d3.pos;
                double ABx=v2.x-v1.x;
                double ABy=v2.y-v1.y;
                double ABz=v2.z-v1.z;
                double ACx=v3.x-v1.x;
                double ACy=v3.y-v1.y;
                double ACz=v3.z-v1.z;
                double ADx=center.x-v1.x;                
                double ADy=center.y-v1.y;                
                double ADz=center.z-v1.z;
                double pVx=ACy*ABz-ACz*ABy;
                double pVy=ACz*ABx-ACx*ABz;
                double pVz=ACx*ABy-ACy*ABx;
                VolumeTot+=(pVx*ADx+pVy*ADy+pVz*ADz)/6.0;                
            }            
        }
        return (float)(VolumeTot);
    }
    /**
     * If the cell timepoint has been tesselated
     * @return the surface of the Cell Timepoint
     */
    public double getSurface() {
        double SurfaceTot=-1;
        if (this.tesselated) {
            SurfaceTot=0;
            for (int i=0;i<triangles.size();i++){
                TriangleN tri = triangles.get(i);
                Vector3D v1=this.dots.get(tri.id1).pos;//tri.d1.pos;                
                Vector3D v2=this.dots.get(tri.id2).pos;//tri.d2.pos;                
                Vector3D v3=this.dots.get(tri.id3).pos;//tri.d3.pos;
                Vector3D AB=new Vector3D(v2.x-v1.x,v2.y-v1.y,v2.z-v1.z);                
                Vector3D AC=new Vector3D(v3.x-v1.x,v3.y-v1.y,v3.z-v1.z);
                SurfaceTot+=1.0/2.0*(Vector3D.prodVect(AB,AC).norme());
            }            
        }
        return (float)(SurfaceTot);
    }    
    /**
     * Update the center property of the Cell T according to its current list of points (dots)
     */
    public void updateCenter() {
        center = new Vector3D(0,0,0);
        for (int i=0;i<dots.size();i++){
            DotN nd = dots.get(i);
            center.x+=nd.pos.x;            
            center.y+=nd.pos.y;            
            center.z+=nd.pos.z;
        }
        // We assume that all point occupies the same area
        center.x/=dots.size();
        center.y/=dots.size();
        center.z/=dots.size();
        modified=true;            
        c.modified=true;
    }
    
    public int constructMesh() {
    	MeshConstructor meshConstructor = new MeshConstructor(dots);
    	triangles = meshConstructor.constructMesh(LimeSeg.opt.d_0);
    	this.tesselated=true;
    	this.flag_stored=meshConstructor.numberOfEdges;
    	//System.out.println("Number of edges = "+meshConstructor.numberOfEdges);
    	if (meshConstructor.numberOfEdges==0) {
    		int isZeroIfSimpleVolume = 2*dots.size()-triangles.size()-4;
    		//System.out.println("isZeroIfSimpleVolume="+isZeroIfSimpleVolume);
    	}
    	freeEdges =	meshConstructor.numberOfEdges;
    	return freeEdges;
    }
    
    @Override
    public CellT clone() {
        final CellT clone;
        try {
            clone = (CellT) super.clone();
        }
        catch (CloneNotSupportedException ex) {
            throw new RuntimeException("superclass messed up", ex);
        }
        clone.dots = new ArrayList<>(this.dots.size());
        for (DotN item : this.dots) {
        	clone.dots.add(item.clone());
        }
        return clone;
    }
    
}
