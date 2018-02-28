package eu.kiaru.limeseg.opt;

import java.util.ArrayList;

import eu.kiaru.limeseg.struct.DotN;
import eu.kiaru.limeseg.struct.Vector3D;

/**
 * Block of dot
 * - Structure used for ternary tree space partitioning
 * @author Nicolas Chiaruttini
 */
public class BlockOfDots {
    public ArrayList<DotN> dotsInBlock0;
    public ArrayList<DotN> dotsInBlock1;
    public int blockLevel;
    public boolean hasCompletelyConverged, isSmallEnough;
    public boolean centerComputed;
    public Vector3D center, dirProj;
    float dmax;
    ArrayList<BlockOfDots> childrenBlocks;
    
    public static int BLOCK_SPLIT = 0;
    public static int BLOCK_KEEP = 1;
    public static int BLOCK_DISCARD = 2;
    public static int BLOCK_TRASH = 3;
    
    int blockKeyIndex;
    
    public BlockOfDots(int  bLevel, int sizeGuessed) {
        blockLevel=bLevel;
        centerComputed=false;
        if (blockLevel==0)  {
            dotsInBlock0 = new ArrayList<>(sizeGuessed);
        } else {
            dotsInBlock0 = new ArrayList<>(sizeGuessed);
            dotsInBlock1 = new ArrayList<>(sizeGuessed);
        }
        hasCompletelyConverged=false;
        isSmallEnough=false;
        center = new Vector3D(0,0,0);
        dirProj = new Vector3D(0,0,0);
    }
    
    public BlockOfDots(int bLevel, int sizeBl0, int sizeBl1) {
        blockLevel=bLevel;
        centerComputed=false;
        if (blockLevel==0)  {
            dotsInBlock0 = new ArrayList<>(sizeBl0);
        } else {
            dotsInBlock0 = new ArrayList<>(sizeBl0);
            dotsInBlock1 = new ArrayList<>(sizeBl1);
        }
        center = new Vector3D(0,0,0);
    }
    
    public int numberOfDotsInBlock() {
        if (blockLevel==0) {
            return dotsInBlock0.size();
        } else {
            return dotsInBlock1.size()+dotsInBlock0.size();
        }
    }
    
    public float numberOfInteractions() {
        if (blockLevel==0) {
            float n = (float) dotsInBlock0.size();
            return n*(n-1)/2;
        } else {
            return (float)(dotsInBlock1.size())*(float)(dotsInBlock0.size());
        }
    }
    int NumberDotInBlock;
    
    public void computeCenterAndConvergence() {
        hasCompletelyConverged=true;
        for (DotN dn:dotsInBlock0) {
            center.x+=dn.pos.x;
            center.y+=dn.pos.y;
            center.z+=dn.pos.z;
            hasCompletelyConverged = hasCompletelyConverged && dn.allNeighborsHaveConverged;
        }
        if (blockLevel>0) {
            for (DotN dn:dotsInBlock1) {
                center.x+=dn.pos.x;
                center.y+=dn.pos.y;
                center.z+=dn.pos.z;
                hasCompletelyConverged = hasCompletelyConverged && dn.allNeighborsHaveConverged;
            }
        }
        NumberDotInBlock = numberOfDotsInBlock();
        center.x/=NumberDotInBlock;
        center.y/=NumberDotInBlock;
        center.z/=NumberDotInBlock;
        centerComputed=true;
    }
    
    public boolean blockComplete(int minNumberOfInteractionsPerBlock) {
        return ((numberOfInteractions()<minNumberOfInteractionsPerBlock)||(isSmallEnough));
    }
    
    public void findDirOfFurthestPoint() {
        dmax=-1;
        DotN dotFar = null;
        for (DotN dn:dotsInBlock0) {
            float dtest=Vector3D.dist2(dn.pos, center);
            if (dtest>dmax) {
                dotFar=dn;
                dmax=dtest;
            }
        }                    
        if (blockLevel>0) {
            for (DotN dn:dotsInBlock1) {
                float dtest=Vector3D.dist2(dn.pos, center);
                if (dtest>dmax) {
                    dotFar=dn;
                    dmax=dtest;
                }
            }
        }
        if (dotFar==null) {
            System.out.println("bah ca alors!");
            System.out.println(this.blockLevel);

            System.out.println("il y a "+dotsInBlock0.size()+" dans le block0.");
            System.out.println("Le centre est à "+center);
            for (DotN dn:dotsInBlock0) {
            	System.out.println("O:"+dn);
            }
            if (dotsInBlock1!=null)
            for (DotN dn:dotsInBlock1) {
                System.out.println("1:"+dn);
            }
            
        }
        // Furthest point "+indFar+" located at  "+((float)(java.lang.Math.sqrt(dmax)))+" pix far away.
        dirProj.x = dotFar.pos.x-center.x;
        dirProj.y = dotFar.pos.y-center.y;
        dirProj.z = dotFar.pos.z-center.z;
        dirProj.normalize();
    }
    
    public int splitBlock(float sqLimitInteractAttract, int minNumberOfInteractions) {        
        computeCenterAndConvergence();   
        childrenBlocks=null;
        if (hasCompletelyConverged) {
            // No need to split : it has fully converged
            // System.out.println("discard : it has fully converged");
            ArrayList<BlockOfDots> ans =  new ArrayList<>(1);
            ans.add(this);
            childrenBlocks = ans;
            if (blockLevel==0) {
                return BLOCK_DISCARD;
            } else {
                return BLOCK_TRASH;
            }
        }
        findDirOfFurthestPoint();
        // Tries to split
        // First, fetch the farthest point        
        if (dmax<sqLimitInteractAttract*sqLimitInteractAttract) {
            // All interactions need to be computed > no spliting required
            //System.out.println(" All interactions need to be computed > no spliting required");
            isSmallEnough=true;
            ArrayList<BlockOfDots> ans =  new ArrayList<>(1);
            ans.add(this);
            childrenBlocks = ans;
            return BLOCK_KEEP;
        }
        // Ok, here we really need to split
        // First create 3 new blocks
        BlockOfDots blPos = new BlockOfDots(blockLevel, NumberDotInBlock*3/2);
        BlockOfDots blNeg = new BlockOfDots(blockLevel, NumberDotInBlock*3/2);
        BlockOfDots blMid0 = new BlockOfDots(blockLevel+1, NumberDotInBlock/2);
        BlockOfDots blMid1 = null;
        if (blockLevel>0) {
            blMid1 = new BlockOfDots(blockLevel+1, NumberDotInBlock/2);
        }
        float dx, dy, dz;
        // Now begins the real deal
        if (blockLevel==0) {
            for (DotN dn:dotsInBlock0) {
                dx=dn.pos.x-center.x;
                dy=dn.pos.y-center.y;
                dz=dn.pos.z-center.z;
                float value=(dx*dirProj.x+dy*dirProj.y+dz*dirProj.z)/sqLimitInteractAttract;
                
                if (value>=0) {
                    blPos.dotsInBlock0.add(dn);
                    if (dn.isSuperDot) {
                        if ((value-dn.superDotRadius/sqLimitInteractAttract)<0) {
                            blNeg.dotsInBlock0.add(dn);
                        }
                    } else if (value<=1) {
                       blMid0.dotsInBlock0.add(dn);
                    }
                } else {
                    blNeg.dotsInBlock0.add(dn);
                    if (dn.isSuperDot) {
                        if ((value+dn.superDotRadius/sqLimitInteractAttract)>=0) {
                            blPos.dotsInBlock0.add(dn);
                        }
                    } else  if (value>-1) {
                       blMid0.dotsInBlock1.add(dn);
                    }
                }
            }
        } else {
            for (DotN dn:dotsInBlock0) {
                dx=dn.pos.x-center.x;
                dy=dn.pos.y-center.y;
                dz=dn.pos.z-center.z;
                float value=(dx*dirProj.x+dy*dirProj.y+dz*dirProj.z)/sqLimitInteractAttract;
                if (value>=0) {
                    blPos.dotsInBlock0.add(dn);
                    if (value<1) {
                       blMid1.dotsInBlock0.add(dn);
                    }
                } else {
                    blNeg.dotsInBlock0.add(dn);
                    if (value>-1) {
                      blMid0.dotsInBlock0.add(dn);
                    }
                }
            }
            for (DotN dn:dotsInBlock1) {
                dx=dn.pos.x-center.x;
                dy=dn.pos.y-center.y;
                dz=dn.pos.z-center.z;
                float value=(dx*dirProj.x+dy*dirProj.y+dz*dirProj.z)/sqLimitInteractAttract;
                if (value>0) {
                    blPos.dotsInBlock1.add(dn);
                    if (value<1) {
                       blMid0.dotsInBlock1.add(dn);
                    }
                } else {
                    blNeg.dotsInBlock1.add(dn);
                    if (value>-1) {
                      blMid1.dotsInBlock1.add(dn);
                    }
                }
            }
        }
        float numberOfInteractions = blNeg.numberOfInteractions()
                                    +blPos.numberOfInteractions()
                                    +blMid0.numberOfInteractions();
        if (blockLevel>0) {
            numberOfInteractions+=blMid1.numberOfInteractions();
        }
        if ((numberOfInteractions<this.numberOfInteractions())&&(numberOfInteractions>minNumberOfInteractions)) {
            ArrayList<BlockOfDots> ans =  new ArrayList<>(4);    
            if (blNeg.numberOfInteractions()>0) {
                ans.add(blNeg);
            }
            if (blPos.numberOfInteractions()>0) {
                ans.add(blPos);
            }
            if (blMid0.numberOfInteractions()>0) {
                ans.add(blMid0);
            }
            if (blockLevel>0) {
                if (blMid1.numberOfInteractions()>0) {
                    ans.add(blMid1);
                }
            }
            childrenBlocks = ans;
            return BLOCK_SPLIT;  
        } else {
            ArrayList<BlockOfDots> ans =  new ArrayList<>(1);
            ans.add(this);
            childrenBlocks = ans;
            return BLOCK_KEEP;
        }
    }
    
}
