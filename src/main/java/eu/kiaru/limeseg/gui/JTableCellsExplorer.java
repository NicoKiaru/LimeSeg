package eu.kiaru.limeseg.gui;


import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Point;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.SwingUtilities;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import javax.swing.event.TableModelEvent;
import javax.swing.event.TableModelListener;
import javax.swing.table.AbstractTableModel;
import javax.swing.table.DefaultTableCellRenderer;

import eu.kiaru.limeseg.LimeSeg;
import eu.kiaru.limeseg.struct.Cell;
import eu.kiaru.limeseg.struct.CellT;


/**
 * Tab containing infos about cells contained in LimeSeg
 *  -> Not completely stable
 * @author Nicolas Chiaruttini
 */
public class JTableCellsExplorer extends javax.swing.JFrame {

    JScrollPane scrollPane;
    
    int curFrame;
    Cell curCell;
    static public boolean needsRedraw;
    static public boolean needsReset;
    public JTableCellsExplorer(){
        initComponents();
        needsRedraw=false;
        needsReset=true;
        this.setTitle("Cells explorer");        
        new Thread(new Runnable() {
              @Override
		public void run() {
                    boolean exit=false;
                    int curFrameActive=0;
                    while (exit==false) {                             
                        int CurrSlice=0;
                        int curFrameTest=0;                       
                        if (LimeSeg.workingImP!=null) {
                            NSlices=LimeSeg.workingImP.getNSlices();
                            NChannel=LimeSeg.workingImP.getNChannels();
                            CurrSlice=LimeSeg.workingImP.getCurrentSlice();
                            curFrameTest=((int)(CurrSlice-1)/(int)(NSlices*NChannel))+1;
                        }
                        if (curFrame!=curFrameTest) {needsRedraw=true;}
                        if (curCell!=LimeSeg.currentCell) {needsRedraw=true;}
                        if (LimeSeg.currentFrame!=curFrameActive) {needsRedraw=true;}                        
                        if (LimeSeg.notifyCellExplorerCellsModif) {needsReset=true;}

                        if (needsReset) {                            
                        	LimeSeg.notifyCellExplorerCellsModif=false;
                            curFrame=curFrameTest;
                            curFrameActive=LimeSeg.currentFrame;
                            curCell=LimeSeg.currentCell;
                            needsRedraw=false;
                            needsReset=false;           
                            makeNewTable();
                        }
                        
                        if (needsRedraw) {
                            curFrame=curFrameTest;
                            curFrameActive=LimeSeg.currentFrame;
                            curCell=LimeSeg.currentCell;
                            needsRedraw=false;
                            // On redraw
                            updateTable();
                            LimeSeg.notifyCellExplorerCellsModif=false;                            
                        }
                        try {
                            Thread.sleep(330);
                        } catch (InterruptedException ex) {
                            Logger.getLogger(JTableCellsExplorer.class.getName()).log(Level.SEVERE, null, ex);
                        }                       
                    }                    
                }
            }).start();
        this.makeNewTable();
        
    }
    
    
    
    JTable table;
    MyTableModel myTableModel;
    MyRenderer myRenderer;
    public void makeNewTable() {
            String[] columnNames;
            Object[][] data;
            if (LimeSeg.allCells!=null) {
                columnNames = new String[LimeSeg.allCells.size()+1];
                for (int i=0;i<LimeSeg.allCells.size();i++) {
                    Cell c = LimeSeg.allCells.get(i);
                    columnNames[i+1]=c.id_Cell;
                }
            } else {
                columnNames = new String[1];
            }
            columnNames[0]="Frame";   
            int NFrames;
            if ((LimeSeg.workingImP!=null)||(LimeSeg.allCells!=null)) {
                NFrames = LimeSeg.getMaxFrames();
                data = new Object[NFrames][LimeSeg.allCells.size()+1];
                for (int f=0;f<NFrames;f++) {
                    data[f][0]=f+1;
                }
                for (int i=0;i<LimeSeg.allCells.size();i++) {    
                    Cell c = LimeSeg.allCells.get(i);
                    for (int f=0;f<NFrames;f++) {
                        CellT ct = c.getCellTAt(f+1);                    
                        if (ct!=null) {         
                            String st="";
                            if (ct.tesselated) {
                                st="T ("+ct.dots.size()+";"+ct.triangles.size()+")";
                            } else {
                                st="C ("+ct.dots.size()+")";
                            }
                                data[f][i+1]=st;
                            } else {                        
                                data[f][i+1]=new String("-");
                            }
                    }
                    c.modified=false;
                }          
            } else {
                data = new Object[1][1];
                data[0][0]=1;
            }      
            JTableCellsExplorer obj=this;
            SwingUtilities.invokeLater(new Runnable() {
                public void run() {
                    obj.getContentPane().removeAll();
                    myTableModel = new MyTableModel(data, columnNames);            
                    table = new JTable(myTableModel);
                    myRenderer = new MyRenderer();  
                    table.setDefaultRenderer(Object.class, myRenderer);        
                    table.getModel().addTableModelListener(myRenderer);        
                    table.setSelectionMode(javax.swing.ListSelectionModel.SINGLE_SELECTION);
                    table.setRowSelectionAllowed(false);
                    table.setColumnSelectionAllowed(false);
                    scrollPane = new JScrollPane(table, JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED, JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
                    table.setFillsViewportHeight(true);
                    table.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
                    table.getSelectionModel().addListSelectionListener(new SelectCellListener());
                    obj.setLayout(new BorderLayout());
                    obj.add(scrollPane,BorderLayout.CENTER);
                    obj.getContentPane().validate();
                    
                    ((JTable)table).addMouseListener(new MouseAdapter() {
                        public void mousePressed(MouseEvent me) {
                            JTable table =(JTable) me.getSource();
                            Point p = me.getPoint();
                            int col = table.columnAtPoint(p);
                            if (me.getClickCount() == 2) {
                            	// Double click! -> select cell
                            	if (col!=0) {
                            		//System.out.println("you double clicked at "+col);
                            		//System.out.println("columnNames[col]="+columnNames[col]);
                            		LimeSeg.handleExtension("selectCellById",new Object[]{columnNames[col]}); 
                            	}
                            }
                        }
                    });
                }
            });
    }
    int NSlices, NChannel;
    public void updateTable() {
        if ((LimeSeg.allCells!=null)||(LimeSeg.workingImP!=null)) {
            int NFrames = LimeSeg.getMaxFrames();
            for (int i=0;i<LimeSeg.allCells.size();i++) {            
                Cell c = LimeSeg.allCells.get(i);
                if (c.modified==true) {
                    for (int f=0;f<NFrames;f++) {
                        CellT ct = c.getCellTAt(f+1);     
                                             
                        //System.out.println("frame="+f);
                        if ((ct!=null)&&(ct.modified==true)) {                            
                            //System.out.println(ct.c.id_Cell+"f="+f);
                            String st="";
                            if (ct.tesselated) {
                                st="T ("+ct.dots.size()+";"+ct.triangles.size()+")";
                            } else {
                                st="C ("+ct.dots.size()+")";
                            }
                            table.getModel().setValueAt(st, f, i+1);
                            ct.modified=false;
                        }
                        if (ct==null) {
                            table.getModel().setValueAt("-", f, i+1);
                        }
                    }
                    c.modified=false;
                }
            }
        }
        this.getContentPane().validate();
        this.repaint();       
    }
    
    public void update() {
        boolean modif=false;
        int NC=0;
        if (LimeSeg.allCells!=null) {
           NC=LimeSeg.allCells.size();
           for (int i=0;i<LimeSeg.allCells.size();i++) {
               Cell c = LimeSeg.allCells.get(i);
               if (c.modified==true) {
                   modif=true;
               }
           }
        }

        if ((modif==true)||(LimeSeg.notifyCellExplorerCellsModif)) {
                NCells=NC;
                this.makeNewTable();              
        }
        
    }
    
    int NCells=0;

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    private void initComponents() {

        setDefaultCloseOperation(javax.swing.WindowConstants.DISPOSE_ON_CLOSE);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 400, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 300, Short.MAX_VALUE)
        );

        pack();
    }




class SelectCellListener implements ListSelectionListener {
    @Override
    public void valueChanged(ListSelectionEvent e) {
           
    }                
}

class MyTableModel extends AbstractTableModel {

    public int getColumnCount() {
        return header.length;
    }

    public int getRowCount() {
        return al.size();
    }

    public String getColumnName(int index) {
        if ((index>0)&&(index<header.length)) {
            return header[index];
        } else {
            return "-";
        }
    }

    public Object getValueAt(int row, int col) {
        if (al!=null) {
            if (al.get(row)!=null) {
                if ((col>=0)&&(col<al.get(row).length)) {
                    return al.get(row)[col];
                } else {
                    return null;
                }
            } else {
                return null;
            }
        } else {
            return null;
        }
    }

    public Class getColumnClass(int c) {
        String st="";
        return st.getClass();
    }

    /*
     * Don't need to implement this method unless your table's
     * editable.
     */
    public boolean isCellEditable(int row, int col) {
        return false;
    }

    /*
     * Don't need to implement this method unless your table's
     * data can change.
     */
    public void setValueAt(Object value, int row, int col) {
        if (al.get(row)!=null) {
            if (col<al.get(row).length) {
                (al.get(row))[col]=value;
                fireTableCellUpdated(row, col);
            }
        }
    }
    
    ArrayList<Object[]> al;
    String[] header;
    
    public MyTableModel(Object[][] obj, String[] header) {
            this.header = header;
            al = new ArrayList<Object[]>();
            // copy the rows into the ArrayList
            for(int i = 0; i < obj.length; ++i)
                al.add(obj[i]);
    }

    
    
}

class MyRenderer extends DefaultTableCellRenderer implements TableModelListener  
{ 
    public Component getTableCellRendererComponent(JTable table, Object value, boolean   isSelected, boolean hasFocus, int row, int column) 
    { 
        Component c = super.getTableCellRendererComponent(table, value, isSelected, hasFocus, row, column); 
        int r=255;
        int g=255;
        int b=255;
        if(row == table.getSelectedRow()) {
            r-=55;
        }
        if(column == table.getSelectedColumn()) {
            b-=55;      
        }        
        
        if ((row+1) == curFrame) {
            r-=50;
            g-=50;
            b-=50;
        }
        
        if ((row+1)==LimeSeg.currentFrame) {
            g-=50;
            b-=50;
            if (r<205) r+=50;
        }
        if (LimeSeg.currentCell!=null) {
            if (table.getModel()!=null)
            if (table.getModel().getColumnName(column)!=null)
            if (table.getModel().getColumnName(column).equals(LimeSeg.currentCell.id_Cell)) {
                r-=40;
                g-=20;
                b-=10;
            }
        }
        if (row==LimeSeg.currentFrame) {
            r-=50;
            g-=50;
            b-=50;
        }       
        
        Color col = new java.awt.Color(r,g,b);
        c.setBackground(col);
        
        return c; 
    }
    
    int selectRow=-1;
    int selectColumn=-1;
    
    @Override
    public void tableChanged(TableModelEvent e) {
    	selectRow = e.getFirstRow();
        selectColumn = e.getColumn();
        MyTableModel model = (MyTableModel)e.getSource();
    
    }
} 
}



