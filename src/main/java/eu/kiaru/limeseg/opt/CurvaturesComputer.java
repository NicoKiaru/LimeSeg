package eu.kiaru.limeseg.opt;

import eu.kiaru.limeseg.struct.DotN;
import eu.kiaru.limeseg.struct.LocalCurvature;
import eu.kiaru.limeseg.struct.Vector3D;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

import org.orangepalantir.leastsquares.Fitter;
import org.orangepalantir.leastsquares.Function;
import org.orangepalantir.leastsquares.fitters.LinearFitter;

public class CurvaturesComputer {

    ArrayList<DotN> dots;

    HashMap<DotN,ArrayList<DotN>> neighbors;

    HashMap<DotN, LocalCurvature> curvature;


    float d_0, range;

    public CurvaturesComputer(ArrayList<DotN> dots_in,float d_0, float range) {
        dots = dots_in;
        for (int i=0;i<dots.size();i++) {
            DotN dn;
            dn = dots.get(i);
            dn.dotIndex=i;
        }
        this.d_0 = d_0;
        this.range = range;
        System.out.print("--- Finding and storing dot neighbors >> ");
        ComputeNeighbors(); // stores all neighbors within distance = d_0*range of this point
        System.out.println("DONE.");

        System.out.print("--- Computing curvatures >> ");
        curvature = new HashMap<>();

        dots.parallelStream().forEach(d ->
        {
                ArrayList<DotN> nd = neighbors.get(d);
                // d is the dot
                // nd are the neighboring dots
                // Just need to compute a local curvature matrix now!
                // https://computergraphics.stackexchange.com/questions/1718/what-is-the-simplest-way-to-compute-principal-curvature-for-a-mesh-triangle
                // No because it's ok only for mean curvature
                // Let's go for a full least square fit of the second fundamental form, assuming a plane tangent to d.Norm
                // 1 - Let's find a local 3D coordinate system: d.Norm = n, u and v

                Vector3D n = d.Norm.clone();
                n.x*=-1; n.y*=-1; n.z*=-1;
                Vector3D u;
                if (d.Norm.x<0.1f) {
                    u = new Vector3D(0f,n.z,-n.y);
                } else {
                    u = new Vector3D(n.y,-n.x,0);
                }
                u.normalize();
                Vector3D v = Vector3D.prodVect(n,u); // Norm = 1 by construction

                LocalCurvature lc = new LocalCurvature();

                lc.n = n;
                lc.u = u;
                lc.v = v;

                // Computes points coordinates in the new local system

                List<double[]> localPos = nd.stream().map(dotn -> {
                    float xp = dotn.pos.x-d.pos.x;
                    float yp = dotn.pos.y-d.pos.y;
                    float zp = dotn.pos.z-d.pos.z;

                    return new double[] {
                            xp*u.x+yp*u.y+zp*u.z,
                            xp*v.x+yp*v.y+zp*v.z,
                            xp*n.x+yp*n.y+zp*n.z};

                }).collect(Collectors.toList());

                // Least square fit of curvature matrix to data points
                //https://github.com/odinsbane/least-squares-in-java
                double[][] dataCoords = new double [localPos.size()][2];
                double[] dataZPos = new double[localPos.size()];

                for (int i=0;i<localPos.size();i++) {
                    dataCoords[i][0]= localPos.get(i)[0];
                    dataCoords[i][1]= localPos.get(i)[1];
                    dataZPos[i] = localPos.get(i)[2];
                }

                //http://mathworld.wolfram.com/SecondFundamentalForm.html
                Function fun = new Function(){
                    @Override
                    public double evaluate(double[] p, double[] lc) {
                        return p[0]*p[0]*lc[0]/2
                                +p[0]*p[1]*lc[1]
                                +p[1]*p[1]*lc[2]/2; // Quadratic expression of the local surface height, see https://en.wikipedia.org/wiki/Second_fundamental_form
                    }
                    @Override
                    public int getNParameters() {
                        return 3;
                    }

                    @Override
                    public int getNInputs() {
                        return 2;
                    }
                };

                Fitter fit = new LinearFitter(fun);
                fit.setData(dataCoords, dataZPos);
                fit.setParameters(lc.cM);
                try {
                    fit.fitData();
                    lc.cM=fit.getParameters();
                    d.meanCurvature=(float) lc.getMeanCurvature();
                    d.gaussianCurvature=(float) lc.getGaussianCurvature();
                    curvature.put(d,lc);
                } catch (RuntimeException e) {
                    // Could not fit point
                }

        }
        );
        System.out.println("DONE.");
    }

    public void ComputeNeighbors() {
        // Build tree to find all neighbors within distance range
        Optimizer opt = new Optimizer();
        dots.forEach(d->{
            d.reInit();
            d.allNeighborsHaveConverged=false;
            d.allNeighborsHaveConvergedPreviously=false;
        }); // avoids moving dots when optimizing

        // push the only relevant parameter for mesh construction
        opt.dots=dots;
        opt.setOptParam("d_0", d_0);
        opt.setOptParam("radiusTresholdInteract", range);
        ArrayList<BlockOfDots> iniBlocks = new ArrayList<>();
        BlockOfDots firstBlock=new BlockOfDots(0,dots.size());
        firstBlock.dotsInBlock0=dots;

        // Gives index for GPU purposes
        iniBlocks.add(firstBlock);
        if (opt.CUDAEnabled) {
            opt.CUDAEnabled=false;
            opt.buildTreeForNeighborsSearch(iniBlocks,1500);
            opt.CUDAEnabled=true;
        } else {
            opt.buildTreeForNeighborsSearch(iniBlocks,1500);
        }
        float d_02 = (d_0*range*d_0*range);
        opt.limitInteractAttract = d_02;
        // Now hydrates the list of neighbors
        int nBlocks = opt.allBlocks.size();
        // let's do this block by block
        // We need a two level hashmap to store information about the edges
        // edges indexes will be A and B, with A>B
        //System.out.print("Storing edges...");
        int nBlockTotal =  opt.allBlocks.size();
        //nBlockDone = 0;

        neighbors = new HashMap<>();

        opt.allBlocks.forEach(block ->
                {
                    if (block.blockLevel != 0) {
                        for (DotN dn1 : block.dotsInBlock0) {
                            for (DotN dn2 : block.dotsInBlock1) {
                                if (Vector3D.dist2(dn1.pos, dn2.pos) < d_02) {
                                    storeEdge(dn1,dn2);
                                }
                            }
                        }
                    } else {
                        int blSize = block.dotsInBlock0.size();
                        for (int i = 0; i < blSize - 1; i++) {
                            DotN dn1 = block.dotsInBlock0.get(i);
                            for (int j = i + 1; j < blSize; j++) {
                                DotN dn2 = block.dotsInBlock0.get(j);
                                if (Vector3D.dist2(dn1.pos, dn2.pos) < d_02) {
                                    storeEdge(dn1,dn2);
                                }
                            }
                        }
                    }
                }
        );
    }

    void storeEdge(DotN dn1, DotN dn2) {
        if (neighbors.containsKey(dn1)) {
            neighbors.get(dn1).add(dn2);
        } else {
            ArrayList<DotN> ini = new ArrayList<>();
            ini.add(dn2);
            neighbors.put(dn1,ini);
        }
        if (neighbors.containsKey(dn2)) {
            neighbors.get(dn2).add(dn1);
        } else {
            ArrayList<DotN> ini = new ArrayList<>();
            ini.add(dn1);
            neighbors.put(dn2,ini);
        }
    }

}
