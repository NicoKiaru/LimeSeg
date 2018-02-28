package eu.kiaru.limeseg.struct;
import java.util.ArrayList;

/**
 * Allows for open or closed polygons
 * If it is closed dots.lastElement has to be equal to dots.get(0);
 * @author Nicolas Chiaruttini
 */
public class PolygonSkeleton {
    ArrayList<Vector3D> dots;    
    public boolean isClosed;
    float length;
    float[] coords;
    
    public float getLength() {
        return length;
    }
    
    public void setDots(ArrayList<Vector3D> v_in) {
        dots=v_in;
        Vector3D v1 = dots.get(0);
        Vector3D v2 = dots.get(dots.size()-1);
        isClosed = false;
        if ((v1.x==v2.x)&&(v1.y==v2.y)&&(v1.z==v2.z)) {isClosed=true;}
        this.preCompute();
    }
    
    private void preCompute() {        
        length=0;
        coords= new float[dots.size()];
        coords[0]=0f;
        for (int i=0;i<(dots.size()-1);i++) {
            Vector3D pA=dots.get(i);
            Vector3D pB=dots.get(i+1);
            coords[i+1]=coords[i]+Vector3D.dist(pA, pB);
        }          
        
        length=coords[dots.size()-1];
        for (int i=0;i<dots.size();i++) {
            coords[i]/=length;
        }
        
    }
    
    void getPosAt(float pathCoord, Vector3D out) {
            if (pathCoord<0) {pathCoord=pathCoord-(int)(pathCoord)+1;}        
            if (pathCoord>1) {pathCoord=pathCoord-(int)(pathCoord);}
            if (dots.size()==1) {            
                out.x=dots.get(0).x;
                out.y=dots.get(0).y;
                out.z=dots.get(0).z;
            } else if (pathCoord==0) {
                out.x=dots.get(0).x;
                out.y=dots.get(0).y;
                out.z=dots.get(0).z;
            } else {
                // Get bounding points
                int i=0;
                while (coords[i]<pathCoord) {i++;}
                // it is between i-1 and i
                Vector3D pA=dots.get(i-1);            
                Vector3D pB=dots.get(i);
                float c=(pathCoord-coords[i-1])/(coords[i]-coords[i-1]);            
                out.x=pA.x+c*(pB.x-pA.x);
                out.y=pA.y+c*(pB.y-pA.y);
                out.z=pA.z+c*(pB.z-pA.z);
            }
    }  
    
    public static float getMaxDist(PolygonSkeleton p1, PolygonSkeleton p2)  {
        float maxDist=0;
        for (int i=0;i<p1.dots.size();i++) {
            for (int j=0;j<p2.dots.size();j++) {
                Vector3D v1=p1.dots.get(i);                
                Vector3D v2=p2.dots.get(j);                
                if (Vector3D.dist(v1, v2)>maxDist) {
                    maxDist=Vector3D.dist(v1, v2);
                }               
            }
        }
        return maxDist;
    }       
    
}
