package eu.kiaru.limeseg.gui;

import eu.kiaru.limeseg.struct.DotN;

public class DefaultDotNColorSupplier extends DotNColorSupplier {
    @Override
    public float[] getColor(DotN dn) {
        return new float[] {dn.ct.c.color[0],dn.ct.c.color[1],dn.ct.c.color[2],dn.ct.c.color[3]};
    }
}
