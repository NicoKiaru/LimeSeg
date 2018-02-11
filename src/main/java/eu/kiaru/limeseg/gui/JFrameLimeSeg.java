package eu.kiaru.limeseg.gui;

import java.awt.BorderLayout;
import java.awt.Dimension;
import javax.swing.JFrame;

import eu.kiaru.limeseg.LimeSeg;


/**
 * Awt window for LimeSeg GUI display
 * Container of JPanelLimeSeg
 * @author Nicolas Chiaruttini
 */
public class JFrameLimeSeg extends JFrame{
    public JFrameLimeSeg(LimeSeg lms) {
        initComponents(lms);
    }
    
    public void initComponents(LimeSeg lms) {
        setDefaultCloseOperation(javax.swing.WindowConstants.DISPOSE_ON_CLOSE);
        this.setPreferredSize(new Dimension(480,600));
        this.setLayout(new BorderLayout());        
        JPanelLimeSeg optExplorer = new JPanelLimeSeg(lms);
        this.add(optExplorer,BorderLayout.CENTER);
        this.getContentPane().validate();
        this.getContentPane().repaint();
        this.setTitle("Lipid Membrane Segmentation");
        pack();
    }
}
