package eu.kiaru.limeseg.struct;

/**
 *
 * @author Nicolas Chiaruttini
 */
public class TriangleN implements Cloneable{
    public int id1, id2, id3;
    
    public TriangleN() {
    }
    
    public TriangleN(int i1, int i2, int i3) {
        id1=i1;
        id2=i2;
        id3=i3;        
    }
    
    @Override
    public TriangleN clone() {
    	return (TriangleN) (this.clone());
    }
    
}
