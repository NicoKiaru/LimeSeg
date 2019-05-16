package eu.kiaru.limeseg.gui;

import eu.kiaru.limeseg.LimeSeg;
import eu.kiaru.limeseg.struct.DotN;

public class CurrentCellColorLUT extends DotNColorSupplier {

    public float[] getColor(DotN dn) {
        if (dn.ct!=null) {
            if (dn.ct.c!=null) {
                if (dn.ct.c== LimeSeg.currentCell) {
                    return new float[]{0.5f,0.8f,0.5f,1f};
                }
            }
        }
        return new float[]{0.8f,0.5f,0.5f,1f};
    }

}