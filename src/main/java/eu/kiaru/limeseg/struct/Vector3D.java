package eu.kiaru.limeseg.struct;

/**
 *
 * @author Nicolas Chiaruttini
 */
public class Vector3D implements Cloneable{
    public float x,y,z;
    
    public Vector3D(float x_, float y_, float z_) {
        x=x_;
        y=y_;
        z=z_;
    }
    
    public Vector3D(double x_, double y_, double z_) {
        x=(float)x_;
        y=(float)y_;
        z=(float)z_;
    }
    
    public float normalize() {
        float N = this.norme();
        if (N==0) {
            x=0; y=0;z=0;
        } else {
            float factor = 1f/N;
            x*=factor;y*=factor;z*=factor;
        }
        return N;
    }
    
    static public Vector3D VecteurDir(Vector3D pA, Vector3D pB) {
        return new Vector3D(pB.x-pA.x, pB.y-pA.y, pB.z-pA.z);
    }
    
    public float norme() {
        return (float) java.lang.Math.sqrt(x*x+y*y+z*z);
    }
    
    public float norme2() {
        return (x*x+y*y+z*z);
    }
    
    static public float prodScal(Vector3D v1, Vector3D v2) {
        return v1.x*v2.x+v1.y*v2.y+v1.z*v2.z;
    }
    
    static public float dist2(Vector3D v1, Vector3D v2) {
        return ((v2.x-v1.x)*(v2.x-v1.x)+(v2.y-v1.y)*(v2.y-v1.y)+(v2.z-v1.z)*(v2.z-v1.z));
    }
    
    static public float dist(Vector3D v1, Vector3D v2) {
        return (float)(java.lang.Math.sqrt((v2.x-v1.x)*(v2.x-v1.x)+(v2.y-v1.y)*(v2.y-v1.y)+(v2.z-v1.z)*(v2.z-v1.z)));
    }
    
    static public Vector3D prodVect(Vector3D v1, Vector3D v2) {
        return new Vector3D(
                v1.y*v2.z-v1.z*v2.y,
                v1.z*v2.x-v1.x*v2.z,
                v1.x*v2.y-v1.y*v2.x);
    }
    
    static public void prodVectWithOutput(Vector3D v1, Vector3D v2, Vector3D output) {
        output.x=v1.y*v2.z-v1.z*v2.y;
        output.y=v1.z*v2.x-v1.x*v2.z;
        output.z=v1.x*v2.y-v1.y*v2.x;
    }
    
    @Override
    public String toString() {
        return ("[x="+x+";y="+y+";z="+z+"]");
    }
    
    @Override 
    public Vector3D clone() {
    	try {
			return (Vector3D) (super.clone());
		} catch (CloneNotSupportedException e) {
			return null;
		}
    }
    
}
