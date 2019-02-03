package eu.kiaru.limeseg.struct;

public class LocalCurvature {

    public Vector3D n; // normal vector
    public Vector3D u;
    public Vector3D v;

    public double[] cM = new double[] {0f,0f,0f}; // opposed triangular curvature Matrix

    public double getGaussianCurvature() { // Matrix determinant
        return cM[0]*cM[2]+cM[1]*cM[1];
    }

    public double getMeanCurvature() { // Matrix trace
        return 0.5f*(cM[0]+cM[2]);
    }

    public Vector3D[] getPrincipalCurvatures() {
        // Unsupported yet
        return null;
    }

}
