package eu.kiaru.limeseg.commands;

import eu.kiaru.limeseg.LimeSeg;
import eu.kiaru.limeseg.opt.CurvaturesComputer;
import eu.kiaru.limeseg.struct.DotN;
import net.imagej.ImageJ;
import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import java.util.ArrayList;

@Plugin(type = Command.class, menuPath = "Plugins>LimeSeg>Test Curvature")
public class TestCurvature implements Command {
    @Override
    public void run() {
        System.out.println("Test Curvature begin!");

        ArrayList<DotN> sheet;
        //sheet = LimeSeg.makeXYSheet(2f,-10,-10,10,10,0);
        sheet = LimeSeg.makeSphere(1f,0,0,0,10);
        //sheet = LimeSeg.makeCylinder(1.5f,0,0,0,10,2);
        CurvaturesComputer cc = new CurvaturesComputer(sheet,1f,2.5f);

        System.out.println("Test Curvature end!");
    }

    public static void main(String... args) {
        ImageJ ij = new ImageJ();
        ij.ui().showUI();
        ij.command().run(TestCurvature.class,true);
    }
}
