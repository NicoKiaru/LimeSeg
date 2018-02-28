package eu.kiaru.limeseg.struct;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.function.Predicate;

/**
 *
 * @author Nicolas Chiaruttini
 */
public class Skeleton2D {
	
    public ArrayList<PolygonSkeleton> pols;
    
    public Skeleton2D() {
        pols= new ArrayList<>();
    }
    
    public ArrayList<DotN> getSurface(float r0) {
        ArrayList<DotN> ans = new ArrayList<>();
        LinkedList<DotN> previousDots = new LinkedList<>();        
        int ip1=0;
        int ip2=1;
        while (ip2<pols.size()) {
            PolygonSkeleton pol1=pols.get(ip1);        
            PolygonSkeleton pol2=pols.get(ip2);
            int maxListSize=0;
            if (2*(pol1.getLength()/r0)>maxListSize) {maxListSize=(int) (2*(pol1.getLength()/r0));}
            if (2*(pol2.getLength()/r0)>maxListSize) {maxListSize=(int) (2*(pol2.getLength()/r0));}
            float dx;
            float dy = 0.5f /(PolygonSkeleton.getMaxDist(pol1,pol2)/r0);
            float y=0;
            Vector3D curPos = new Vector3D(0,0,0);
            Vector3D curPosPDX = new Vector3D(0,0,0);            
            Vector3D vDX = new Vector3D(0,0,0);
            Vector3D vDY = new Vector3D(0,0,0);
            Vector3D curPosPol1 = new Vector3D(0,0,0);        
            Vector3D curPosPol2 = new Vector3D(0,0,0);
            float x=0;
            while (y<1) {
                dx = 0.5f/((pol1.length+y*(pol2.length-pol1.length))/r0);
                while (x<1) {
                    if (pol1.length+y*(pol2.length-pol1.length)!=0) {
                        pol1.getPosAt(x, curPosPol1);
                        pol2.getPosAt(x, curPosPol2);

                        curPos.x=curPosPol1.x+(y+x*dy)*(curPosPol2.x-curPosPol1.x);                
                        curPos.y=curPosPol1.y+(y+x*dy)*(curPosPol2.y-curPosPol1.y);                
                        curPos.z=curPosPol1.z+(y+x*dy)*(curPosPol2.z-curPosPol1.z);

                        vDY.x=dy*(curPosPol2.x-curPosPol1.x);                
                        vDY.y=dy*(curPosPol2.y-curPosPol1.y);                
                        vDY.z=dy*(curPosPol2.z-curPosPol1.z);

                        if (x+dx<1) {
                            pol1.getPosAt(x+dx, curPosPol1);
                            pol2.getPosAt(x+dx, curPosPol2);
                        } else {
                            pol1.getPosAt(1, curPosPol1);
                            pol2.getPosAt(1, curPosPol2);
                        }

                        curPosPDX.x=curPosPol1.x+(y+x*dy)*(curPosPol2.x-curPosPol1.x);                
                        curPosPDX.y=curPosPol1.y+(y+x*dy)*(curPosPol2.y-curPosPol1.y);                
                        curPosPDX.z=curPosPol1.z+(y+x*dy)*(curPosPol2.z-curPosPol1.z);

                        vDX.x=curPosPDX.x-curPos.x;
                        vDX.y=curPosPDX.y-curPos.y;
                        vDX.z=curPosPDX.z-curPos.z;

                        Vector3D n = Vector3D.prodVect(vDX, vDY);
                        // Interacting way
                        Vector3D pNew = new Vector3D(curPos.x,curPos.y,curPos.z);
                        float alpha=1.4f;
                        Predicate<DotN> p = dn -> Vector3D.dist2(pNew, dn.pos) > alpha*r0*r0;
                        if (previousDots.stream().allMatch(p)) {
                            n.normalize();
                            DotN dn = new DotN(pNew,n);
                            ans.add(dn);  
                            previousDots.add(dn);
                            while (previousDots.size()>maxListSize) {
                                previousDots.removeFirst();
                            }         
                        }
                        
                    } else {
                        dx=1f;
                    }
                    x=x+dx;                    
                }
                x=x-(int)(x);
                y=y+dy;
            }   
            ip1++;
            ip2++;
        }
        return ans;       
    }   
}

