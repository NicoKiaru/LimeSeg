package eu.kiaru.limeseg.struct;

import java.util.ArrayList;

import eu.kiaru.limeseg.opt.Optimizer;
import net.imglib2.RealRandomAccess;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/** 
 * Represents a surface element / Surfel / lipid particle
 * Has a position and a unit normal vector
 * Can also be a superdot containing other dots 
 * -------------- Parameters for convergence check
 * The basic idea is that if a particle hasn't moved by more than convDist within convTimesteps along its normal vector, it has converged
 * Things to check:
 *  - set hasConverged to false upon dot creation
 *  - decide when a parameter is changed, whether dot convergence flag has to be reset
 * also a global static value will say whether a convergence check has to be performed
 * @author Nicolas Chiaruttini
 * @see Optimizer
 */
public class DotN implements Cloneable {    
    /**
     * Index of dot within ArrayList - a bit ugly
     */
    public int dotIndex;    
    // ---------------- Variables for superDot implementation
    /**
     * Flags if the current DotN is a superDot
     */
    public boolean isSuperDot=false;
    /**
     * if isSuperDot : radius and squared radius of superDot 
     */
    public float superDotRadius,superDotRadiusSquared=0;
    /**
     * if isSuperDot : list of DotN contained within the superDot
     */
    public ArrayList<DotN> dotsContained=null;
    /**
     * sets whether the Surfel is locked by the Optimizer - necessary to create a superdot
     */
    public boolean isOptimized=true;
    
    // ---------------- Variables defined by the user to customize dot behavior
    /**
     * sets whether the surfel position can be updated by the Optimizer
     */
    public boolean userMovable=true; 
    /**
     * sets whether the surfel normal can be updated by the Optimizer
     */
    public boolean userRotatable=true;
    /**
     * sets whether the Optimizer can generate new surfel from this surfel
     */
    public boolean userGenerate=true;
    /**
     * sets whether the surfel can be removed by the Optimizer
     */
    public boolean userDestroyable=true;
    /**
     * to flag the surfel with an IJ1 script
     */
    public boolean userDefinedFlag=false;    
    /*
     * variables to diminish the size to store DotN boolean properties 
     */
    static private int userMovableMask=0b00000001;
    static private int userRotatableMask=0b00000010;
    static private int userDestroyableMask=0b00000100;
    static private int userGenerateMask=0b00001000;
    //----------------------- Variables for convergence
    /**
     * flags if a surfel has Converged
     */
    public boolean hasConverged=false;
    /**
     * Previous state (position) which is updated every convTimesteps (Optimizer parameter)
     */
    private Vector3D previousPos;
    /**
     * Previous state (normal) which is updated every convTimesteps (Optimizer parameter)
     */
    private Vector3D previousNorm;
    /*
     * flags if allNeighborsHaveConverged
     */
    public boolean allNeighborsHaveConverged=false;
    /*
     * Flags if allNeighborsHaveConverged at the previous timestep = buffers previous state for Optimizer
     */
    public boolean allNeighborsHaveConvergedPreviously=false;
    /*
     *  For relaxed mode set to 1 when the dot found the correct position -> normalForce to zero
     */
    public float relaxed;
    // ------------- Mechanical parameters
    /**
     * Position of the surfel
     */
    public Vector3D pos;
    /**
     * Normal of the surfel
     */
    public Vector3D Norm;
    /**
     * force exerted on the surfel 
     */
    public Vector3D force;
    /**
     * torque exerted on the surfel
     */
    public Vector3D moment;
    /**
     * Sum of the repulsive forces exerted on the surfel - used to set the position of the potentially generated new Surfel
     */
    public Vector3D repForce;
    /**
     * Number of neighboring surfels
     */
    public int N_Neighbor;
    /**
     * number of Optimization steps performed since the surfel generation
     */
    public long age=0;             // probably never be able to reach the max value... it's good like that
    
    // ------------- Relation to image Data    

    /**
     * The cellT to which this superdots belongs to. Is null if this surfel is a superDot.
     */
    public CellT ct=null;
    /**
     * For cell tesselation, this variable keeps a list of the neighboring surfels of the same CellT
     */
    public ArrayList<Integer> Voisins=null;
    /**
     * Can be computed with a @see CurvaturesComputer
     */
    public float gaussianCurvature = Float.NaN;
    /**
     * Can be computed with a @see CurvaturesComputer
     */
    public float meanCurvature = Float.NaN;
    /**
     * Value within image during last optimization step
     */
    public float floatImageValue = Float.NaN;
    
    @Override
    public String toString() {
        String ans="";
        ans+="Movable:";
        if (userMovable) {ans+="yes\n";} else {ans+="no\n";}
        ans+="Destroyable:";
        if (userDestroyable) {ans+="yes\n";} else {ans+="no\n";}
        ans+="Pos="+pos+"\n";
        ans+="Norm="+Norm+"\n";
        ans+="N_Neighbor="+N_Neighbor+"\n";
        ans+="age="+age+"\n";
        if(ct!=null) {
             ans+="Cell = "+ct.c.id_Cell+", Frame="+ct.frame+"\n";
        } else {
            ans+="Cell = null\n";
        }
        return ans;
    }
    
    /**
     * Function used to save state of the surfel
     * @return dots boolean properties as an int (see userGenerate, userMovable etc.)
     */
    public int getDotProps() {
    	int returnValue = 
    			((userMovable==true)?1:0)*userMovableMask+
    			((userRotatable==true)?1:0)*userRotatableMask+
    			((userGenerate==true)?1:0)*userGenerateMask+
    			((userDestroyable==true)?1:0)*userDestroyableMask;
    	return returnValue;
    }
    
    /**
     * Used to restore the boolean properties of the surfel (userMovable, etc.)
     * @param props int which contains boolean surfel properties according to the Mask
     */
    public void setDotProps(int props) {
    	userMovable=((props&userMovableMask)==userMovableMask);
    	userRotatable=((props&userRotatableMask)==userRotatableMask);
    	userGenerate=((props&userGenerateMask)==userGenerateMask);
    	userDestroyable=((props&userDestroyableMask)==userDestroyableMask);
    }
    
    /**
     * Standard surfel constructor
     */
    public DotN() {   
    	this(new Vector3D(0,0,0), new Vector3D(0,0,0));
    } 
    
    /**
     * Constructs surfel with initial Normal and Position
     * @param pos_in po
     * @param N_in
     */
    public DotN(Vector3D pos_in, Vector3D N_in) {
        pos=pos_in;
        Norm=N_in;
        Norm.normalize();
        force = new Vector3D(0,0,0);
        moment = new Vector3D(0,0,0);
        repForce = new Vector3D(0,0,0);
        N_Neighbor=6;        
        // convergence related
        previousPos = new Vector3D(0,0,0);
        previousNorm = new Vector3D(0,0,0);
        age=0;
    }
    
    /**
     * Reinitializes surfel properties for next integration step
     * Reset force, moment, number of neighbors, normalizes normal vector
     * Resets neighbors convergence flag
     */
    public void reInit() {
        force.x=0;force.y=0;force.z=0;
        repForce.x=0;repForce.y=0;repForce.z=0;
        moment.x=0;moment.y=0;moment.z=0;
        N_Neighbor=0;    
        Norm.normalize();
        allNeighborsHaveConverged=true;
    }
    
    /**
     * Returns the data linkage term with the 3D image
     * the force is added to the vector force of the surfel, along the normal direction
     * the force is either 0 , +k_grad, or -k_grad
     * @param rA random accessible corresponding to the 3D image
     * @param k_grad amplitude of the force exerted
     * @param radiusRelaxed in pixel units if the image maximum is in the range [-radiusRelaxed;+radiusRelaxed], the force is zero
     * @param radiusRes in pixel units, sampling size along the normal vector of the surfel
     * @param radiusDelta offsets to the real position of the surfel. Can be used to maintain a distance to the image maxima
     * @param radiusSearch the maximum in the 3D image is searched, in pixels units within [-radiusSearch;+radiusSearch]
     * @param ZScale equals to voxel size in Z / voxel size in x (or y, as the image are assumed isotropic in xy)
     * @param MinX minimal value in x accessed in rA (in pixel)
     * @param MaxX maximal value in x accessed in rA (in pixel)
     * @param MinY minimal value in y accessed in rA (in pixel)
     * @param MaxY maximal value in y accessed in rA (in pixel)
     * @param MinZ minimal value in z accessed in rA (in pixel)
     * @param MaxZ maximal value in z accessed in rA (in pixel)
     */
    public < T extends RealType< T > & NativeType< T >> void computeGradForce_Max(RealRandomAccess<T> rA, float k_grad, float radiusRelaxed, float radiusRes, float radiusDelta, float radiusSearch, float ZScale, float MinX, float MaxX, float MinY, float MaxY, float MinZ, float MaxZ){
        float fx_=0;float fy_=0;float fz_=0;
        float dx,dy,dz,xp,yp,zp; 
        T val_max;
        val_max = rA.get().createVariable();
        dx=radiusRes*Norm.x;
        dy=radiusRes*Norm.y;
        dz=radiusRes*Norm.z;
        xp=pos.x+radiusDelta*Norm.x;
        yp=pos.y+radiusDelta*Norm.y;
        zp=pos.z+radiusDelta*Norm.z;
        rA.setPosition(xp, 0);
        rA.setPosition(yp, 1);
        rA.setPosition(zp/ZScale, 2);
        val_max.set(rA.get());
        T val_test;
        this.floatImageValue = val_max.getRealFloat();
        float r=0;
        int NSteps=(int) ((radiusSearch/2f)/(radiusRes));
        relaxed=1f;
        boolean allEqual=true;
        int comp;
        for (int i=0;i<NSteps;i++) {
            xp+=dx;yp+=dy;zp+=dz;r=r+radiusRes;
            if ((xp>MinX)&&(xp<MaxX)&&(yp>MinY)&&(yp<MaxY)&&(zp/ZScale>MinZ)&&(zp/ZScale<MaxZ)) {
                rA.setPosition(xp, 0);
                rA.setPosition(yp, 1);
                rA.setPosition(zp/ZScale, 2);
                val_test=rA.get();
                comp=val_test.compareTo(val_max);
                allEqual=(allEqual&&(comp==0));
                if (comp>0) {
                    val_max.set(val_test);
                    if (r<radiusRelaxed)  {relaxed=1f;} else {relaxed=0f;}
                    fx_=1f;fy_=1f;fz_=1f; 
                }
            }
        }
        r=0;
        xp=pos.x+radiusDelta*Norm.x;
        yp=pos.y+radiusDelta*Norm.y;
        zp=pos.z+radiusDelta*Norm.z;
        for (int i=0;i<NSteps;i++) {
            xp-=dx;yp-=dy;zp-=dz;r=r+radiusRes;    
            if ((xp>MinX)&&(xp<MaxX)&&(yp>MinY)&&(yp<MaxY)&&(zp/ZScale>MinZ)&&(zp/ZScale<MaxZ)) {
                rA.setPosition(xp, 0);
                rA.setPosition(yp, 1);
                rA.setPosition(zp/ZScale, 2);
                val_test=rA.get();
                comp=val_test.compareTo(val_max);
                allEqual=(allEqual&&(comp==0));
                if (comp>0) {                
                    val_max.set(val_test);
                    if (r<radiusRelaxed)  {relaxed=1f;} else {relaxed=0f;}
                    fx_=-1f;fy_=-1f;fz_=-1f;  
                }
            }
        }
        if (allEqual) {relaxed=0;}
        force.x+=fx_*k_grad*Norm.x;
        force.y+=fy_*k_grad*Norm.y;
        force.z+=fz_*k_grad*Norm.z;
    }

    /**
     * Updates position and normal of the surfel. Called at the end of each timestep.
     * @param normalForce - pressure or balloon force
     * @param d_0 - equilibrium distance between surfel. Forces are rescaled with respect to d_0.
     * @param maxDisplacementPerStep - maximal displacement for each timestep. In d_0 units.
     * @param MinX - minimal position of the surfel in x (pixel unit) 
     * @param MaxX - maximal position of the surfel in x (pixel unit) 
     * @param MinY - minimal position of the surfel in y (pixel unit) 
     * @param MaxY - maximal position of the surfel in y (pixel unit) 
     * @param MinZ - minimal position of the surfel in z (pixel unit) 
     * @param MaxZ - maximal position of the surfel in z (pixel unit) 
     * @param ZScale - ZScale equals to voxel size in Z / voxel size in x (or y, as the image are assumed isotropic in xy)
     */
    public void updatePosition(float normalForce, float d_0, float maxDisplacementPerStep, float MinX, float MaxX, float MinY, float MaxY, float MinZ, float MaxZ, float ZScale) {
        // Computes normal force
        force.x+=normalForce*Norm.x*(1.0f-relaxed);         
        force.y+=normalForce*Norm.y*(1.0f-relaxed);        
        force.z+=normalForce*Norm.z*(1.0f-relaxed);        
        // Basic explicit euler integration
        // Updates position        
        force.x*=d_0;
        force.y*=d_0;
        force.z*=d_0;
        float N=force.norme();
        float maxForce=maxDisplacementPerStep*d_0;
        if (N>maxForce) {
            force.x*=maxForce/N;
            force.y*=maxForce/N;
            force.z*=maxForce/N;
        }
        if ((isOptimized)&&(userMovable)) {
            pos.x+=force.x;
            pos.y+=force.y;
            pos.z+=force.z;   
        }        
        if (pos.x<=MinX) {
            pos.x=MinX;
            relaxed=1.0f;
        }
        if (pos.x>=MaxX) {
            pos.x=MaxX;
            relaxed=1.0f;
        }
        if (pos.y<=MinY) {
            pos.y=MinY;
            relaxed=1.0f;
        }
        if (pos.y>=MaxY) {
            pos.y=MaxY;
            relaxed=1.0f;
        }
        if (pos.z<=MinZ*ZScale) {
            pos.z=MinZ*ZScale;
            relaxed=1.0f;
        }        
        if (pos.z>=MaxZ*ZScale) {
            pos.z=MaxZ*ZScale;
            relaxed=1.0f;
        }       
        if ((isOptimized)&&(userRotatable)) {
            // Updates normal
            Norm.x+=moment.x;
            Norm.y+=moment.y;
            Norm.z+=moment.z;
        }
        
        age++;
    }
    
    /**
     * Checks whether this surfel has converged or not (can work both ways)
     * @param d_0
     * @param convergenceDistTresholdSquared
     * @param convergenceTimestepSampling
     * @param convergenceNormTresholdSquared
     */
    public void checkConvergence(float d_0, float convergenceDistTresholdSquared, float convergenceTimestepSampling, float convergenceNormTresholdSquared) {
        float maxDistConv=d_0*d_0*convergenceDistTresholdSquared; // should be removed from this loop        
        if ((age % convergenceTimestepSampling)==0) {
        	if (age==0) {
        		// Initial step -> no convergence possible
            } else {
                // Has it converged ?
                // Distance measured along the normal vector
                float dTest=(previousPos.x-pos.x)*Norm.x+
                            (previousPos.y-pos.y)*Norm.y+
                            (previousPos.z-pos.z)*Norm.z;
                hasConverged = (dTest*dTest<maxDistConv)&&(Vector3D.dist2(previousNorm, Norm)<convergenceNormTresholdSquared);
            }
            // Update previous pos
            previousPos.x=pos.x;
            previousPos.y=pos.y;
            previousPos.z=pos.z;
            previousNorm.x=Norm.x;
            previousNorm.y=Norm.y;
            previousNorm.z=Norm.z;
        }        
        if ((this.allNeighborsHaveConverged)) {
        	isOptimized=false;            
            hasConverged=true;
            previousPos.x=pos.x;
            previousPos.y=pos.y;
            previousPos.z=pos.z;
            previousNorm.x=Norm.x;
            previousNorm.y=Norm.y;
            previousNorm.z=Norm.z;            
        } else {
        	isOptimized=true;
        }
    }
    
    /**
     * Generates a new surfel from this surfel, according to its current repulsive force
     * @param d_0 - equilibrium distance between surfels
     * @return a new surfel in the direction of the maximal repulsive force, times half of d_0
     */
    public DotN generate(float d_0) {
    	Vector3D p_new = new Vector3D(0,0,0);                
        if ((repForce.x==0)&&(repForce.y==0)&&(repForce.z==0)) {                
        	p_new.x=this.pos.x-(d_0/2f)*this.Norm.x;
            p_new.y=this.pos.y-(d_0/2f)*this.Norm.y;
            p_new.z=this.pos.z-(d_0/2f)*this.Norm.z;
            this.pos.x+=(d_0/4f)*this.Norm.x;
            this.pos.y+=(d_0/4f)*this.Norm.y;
            this.pos.z+=(d_0/4f)*this.Norm.z;
        } else {
        	repForce.normalize();
            float corPlan = repForce.x*Norm.x+repForce.y*Norm.y+repForce.z*Norm.z;
            repForce.x-=corPlan*Norm.x;
            repForce.y-=corPlan*Norm.y;
            repForce.z-=corPlan*Norm.z;
            repForce.normalize();
                    
            p_new.x=this.pos.x+0.5f*(d_0)*repForce.x;
            p_new.y=this.pos.y+0.5f*(d_0)*repForce.y;
            p_new.z=this.pos.z+0.5f*(d_0)*repForce.z;
        }
        Vector3D n_new = new Vector3D(this.Norm.x,this.Norm.y,this.Norm.z);
        DotN nd_new = new DotN(p_new,n_new);
        nd_new.ct=this.ct;
        // this had wrongly been deleted! Do not delete RognTudju
        this.age=0;
        this.hasConverged=false;
        this.allNeighborsHaveConverged=false;
        this.ct.dots.add(nd_new);
        return nd_new;
    }
    
    @Override
    public DotN clone() {
    	
    	 final DotN clone;
         try {
             clone = (DotN) super.clone();
         }
         catch (CloneNotSupportedException ex) {
             throw new RuntimeException("DotN superclass messed up", ex);
         }
         
         clone.pos = pos.clone();
         clone.Norm = Norm.clone();
         clone.ct = null;
         clone.dotsContained = null;
         clone.force = force.clone();
         clone.moment = moment.clone();
         clone.previousNorm = previousNorm.clone();
         clone.previousPos = previousPos.clone();
         clone.repForce = repForce.clone();
         clone.Voisins = null;
         
         
         return clone;
    }
    
    public void stallDot() {
    	userDestroyable=false;
    	userGenerate=false;
    	userMovable=false;
    	userRotatable=false;
    	this.allNeighborsHaveConverged=true;
    	this.allNeighborsHaveConvergedPreviously=true;
    	this.isOptimized=false;
    	this.hasConverged=true;
		if (isSuperDot) {
			this.dotsContained.forEach(dn ->{
				dn.stallDot();
			});
		}
    }
    
    public void freeDot() {
       	userDestroyable=true;
    	userGenerate=true;
    	userMovable=true;
    	userRotatable=true;
    	allNeighborsHaveConverged=false;
    	allNeighborsHaveConvergedPreviously=false;
    	isOptimized=true;
    	hasConverged=false;
    	if (isSuperDot) {
			this.dotsContained.forEach(dn ->{
				dn.freeDot();
			});
		}    	
    }
    
}
