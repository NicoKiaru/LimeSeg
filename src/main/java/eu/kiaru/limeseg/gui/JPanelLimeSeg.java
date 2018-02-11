package eu.kiaru.limeseg.gui;

import ij.WindowManager;

import java.awt.Component;
import java.io.File;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;

import eu.kiaru.limeseg.LimeSeg;
import eu.kiaru.limeseg.ij1script.IJ1ScriptableMethod;

/**
 * Panel containing:
 * - LimeSeg GUI with:
 * 		- commands on top (command panel)
 * 		- Indications about LimeSeg status at the bottom (status panel)
 * Loads commands from LimeSeg methods annotations
 * see {@link eu.kiaru.limeseg.gui.JFrameLimeSeg}
 * @author Nicolas Chiaruttini
 */

public class JPanelLimeSeg extends javax.swing.JPanel {
	LimeSeg lms;
	int counter=0;
	/**
	 * Update status panel
	 */
    public void updateAllDisplayedInfos() {
    	myState.setText(getState());
    }
    
    /**
     * Initializes panel
     * @param lms_in
     */
    public JPanelLimeSeg(LimeSeg lms_in) {
    	lms=lms_in;
        initComponents(lms);
        Object obj=this;        
        new Thread(new Runnable() {
              @Override
              public void run() {
            	  boolean exit=false;
                  while (exit==false) {                   
                	  try {
                		  Thread.sleep(1000); // updates each seconds
                      } catch (InterruptedException ex) {
                      // Logger.getLogger(JTableCellsExplorer.class.getName()).log(Level.SEVERE, null, ex);
                  }
                  ((JPanelLimeSeg)(obj)).updateAllDisplayedInfos();
              }
          }              
        }).start();
    }
    
    ArrayList<JPanel> panels;
    
    private JPanel getPanelByName(String name) {
    	if (panels==null) {
    		panels=new ArrayList<>();
    	}
    	JPanel ans = null;
    	for (JPanel panel:panels) {
    		if (panel.getName().equals(name)) {
    			ans = panel;
    		}
    	}
    	if (ans==null) {
    		JPanel newPanel = new JPanel();
    		newPanel.setName(name);
    		panels.add(newPanel);
    		ans=newPanel;
    	}
    	return ans;
    }    
    
    JTextArea myState;
    
    private String getState() {
    	String str="";
        for (Field f : LimeSeg.class.getFields()) {
        	if (f.isAnnotationPresent(DisplayableOutput.class)) {
            	str+= f.getName()+"=";
            	try {
					str+=f.get(null)+"\n";
				} catch (IllegalArgumentException | IllegalAccessException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
        	}
        }
        
        ArrayList<Method> methods = new ArrayList<Method>();
        for (Method m:LimeSeg.class.getDeclaredMethods()) {
            if (m.isAnnotationPresent(DisplayableOutput.class)) {
            	methods.add(m);
            }
        }
        methods.sort(new MethodsPrioritiesComparator());
        
        for (Method m : methods) {
        	if (m.isAnnotationPresent(DisplayableOutput.class)) {
            	try {
            		//str+="<p>";
            		str+= m.invoke(null, null);
            		//str+="</p>";
				} catch (IllegalArgumentException | IllegalAccessException | InvocationTargetException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
        	}
        }        
    	return str;
    }
    
    private String imageChooser() {
		// IJ1		
    	return (String)JOptionPane.showInputDialog(
                new JFrame(),
                "",
                "Choose 3D Image",
                JOptionPane.PLAIN_MESSAGE,
                null,
                WindowManager.getImageTitles(),
                "");
    }
    
    private void initComponents(LimeSeg lms) {
        jTabbedPanelMethods = new JTabbedPane(JTabbedPane.LEFT,JTabbedPane.SCROLL_TAB_LAYOUT);
        setLayout(new java.awt.GridLayout(2, 1));
        ArrayList<Method> methods = new ArrayList<Method>();
        for (Method m:LimeSeg.class.getDeclaredMethods()) {
            if (m.isAnnotationPresent(IJ1ScriptableMethod.class)) {
            	methods.add(m);
            }
        }
        methods.sort(new MethodsPrioritiesComparator());
        
    	for (Method m:methods){
            if (m.isAnnotationPresent(IJ1ScriptableMethod.class)) {
            	IJ1ScriptableMethod scriptableAnnotation = (IJ1ScriptableMethod)m.getAnnotation(IJ1ScriptableMethod.class);
            	if (!scriptableAnnotation.ui().equals("NO")) {
            		int nParameters = m.getParameterTypes().length;
            		String[] targets = scriptableAnnotation.target().split(",");
            		for (String target:targets) {      
                		Object[] getParams = new Object[nParameters]; 
                   		JButton jButton = new javax.swing.JButton();
	                    JPanel paramPanel = new JPanel();
	                    paramPanel.setLayout(new java.awt.GridLayout(1, nParameters));
	                    jButton.setText(m.getName());
	                    JPanel targetPanel = getPanelByName(target);
	                    targetPanel.add(jButton);
	                    targetPanel.add(paramPanel);
	                    jButton.setToolTipText("<html>"+m.getName()+scriptableAnnotation.tt()+"</html>");
	                    for (int i=0;i<nParameters;i++) {
	            			Class p = m.getParameterTypes()[i];
	        				if   ((p.getName()=="float")
	        					||(p.getName()=="double")
	        					||(p.getName()=="int")
	        					||(p.getName()=="java.lang.Float")
	        					||(p.getName()=="java.lang.Integer")
								||(p.getName()=="java.lang.Double")) {
	        					getParams[i] = new JTextField();
	        					((JTextField) getParams[i]).setToolTipText(m.getParameters()[i].getName());
	        					paramPanel.add((JTextField) getParams[i]);	        					
	        				}
	        				if (p.getName().equals("java.lang.String")) {
	        					// Needs to check the type:
	        					//	- Is it a filepath ?
	        					//	- Is it a standard string ?
	        					if (scriptableAnnotation.ui().equals("ImageChooser")) {
		        					getParams[i] = new javax.swing.JButton();
	        						((JButton) getParams[i]).addActionListener(new java.awt.event.ActionListener() {
	        							public void actionPerformed(java.awt.event.ActionEvent evt) {
	    		        					String imgName=imageChooser();
	    		        					((JButton) evt.getSource()).setToolTipText(imgName);
	    	        						((JButton) evt.getSource()).setText("Img:"+imgName);
	        							}
	        						});
	        						((JButton) getParams[i]).setText("Img:");
	        						paramPanel.add((JButton) getParams[i]);
	        					}
	        					if (scriptableAnnotation.ui().equals("FileWriter")) {
	        						getParams[i] = new javax.swing.JButton();
	        						((JButton) getParams[i]).addActionListener(new java.awt.event.ActionListener() {
	        							public void actionPerformed(java.awt.event.ActionEvent evt) {
	        								JFileChooser fc = openFChooser();	        						        
	        						        int returnVal = fc.showOpenDialog((Component) evt.getSource());
	        						        if (returnVal == JFileChooser.APPROVE_OPTION) {
	        						        	((JButton) evt.getSource()).setToolTipText(fc.getSelectedFile().getAbsolutePath());
	        						        	((JButton) evt.getSource()).setText("WriteTo:"+fc.getSelectedFile().getAbsolutePath());
	        						        }
	        						        if (fc.getCurrentDirectory()!=null) {JPanelLimeSeg.currentPath=fc.getCurrentDirectory().getAbsolutePath();}
	        							}
	        						});
	        						((JButton) getParams[i]).setText("WriteTo:");
	        						paramPanel.add((JButton) getParams[i]);
	        					}
	        					if (scriptableAnnotation.ui().equals("FileOpener")) {
	        						getParams[i] = new javax.swing.JButton();
	        						((JButton) getParams[i]).addActionListener(new java.awt.event.ActionListener() {
	        							public void actionPerformed(java.awt.event.ActionEvent evt) {
	        								JFileChooser fc = openFChooser();	        						        
	        						        int returnVal = fc.showOpenDialog((Component) evt.getSource());
	        						        if (returnVal == JFileChooser.APPROVE_OPTION) {
	        						        	((JButton) evt.getSource()).setToolTipText(fc.getSelectedFile().getAbsolutePath());
	        						        	((JButton) evt.getSource()).setText("LoadFrom:"+fc.getSelectedFile().getAbsolutePath());
	        						        }
	        						        if (fc.getCurrentDirectory()!=null) {JPanelLimeSeg.currentPath=fc.getCurrentDirectory().getAbsolutePath();}
	        							}
	        						});
	        						((JButton) getParams[i]).setText("LoadFrom:");
	        						paramPanel.add((JButton) getParams[i]);
	        					}
	        					if (scriptableAnnotation.ui().equals("PathWriter")) {
	        						getParams[i] = new javax.swing.JButton();
	        						((JButton) getParams[i]).addActionListener(new java.awt.event.ActionListener() {
	        							public void actionPerformed(java.awt.event.ActionEvent evt) {
	        								JFileChooser fc = openFChooser(true);	        						        
	        						        int returnVal = fc.showOpenDialog((Component) evt.getSource());
	        						        if (returnVal == JFileChooser.APPROVE_OPTION) {
	        						        	((JButton) evt.getSource()).setToolTipText(fc.getSelectedFile().getAbsolutePath());
	        						        	((JButton) evt.getSource()).setText("WriteTo:"+fc.getSelectedFile().getAbsolutePath());
	        						        }
	        						        if (fc.getCurrentDirectory()!=null) {JPanelLimeSeg.currentPath=fc.getCurrentDirectory().getAbsolutePath();}
	        							}
	        						});
	        						((JButton) getParams[i]).setText("WriteTo:");
	        						paramPanel.add((JButton) getParams[i]);
	        					}
	        					if (scriptableAnnotation.ui().equals("PathOpener")) {
	        						getParams[i] = new javax.swing.JButton();
	        						((JButton) getParams[i]).addActionListener(new java.awt.event.ActionListener() {
	        							public void actionPerformed(java.awt.event.ActionEvent evt) {
	        								JFileChooser fc = openFChooser(true);	        						        
	        						        int returnVal = fc.showOpenDialog((Component) evt.getSource());
	        						        if (returnVal == JFileChooser.APPROVE_OPTION) {
	        						        	((JButton) evt.getSource()).setToolTipText(fc.getSelectedFile().getAbsolutePath());
	        						        	((JButton) evt.getSource()).setText("LoadFrom:"+fc.getSelectedFile().getAbsolutePath());
	        						        }
	        						        if (fc.getCurrentDirectory()!=null) {JPanelLimeSeg.currentPath=fc.getCurrentDirectory().getAbsolutePath();}
	        							}
	        						});
	        						((JButton) getParams[i]).setText("LoadFrom:");
	        						paramPanel.add((JButton) getParams[i]);
	        					}
	        					if (scriptableAnnotation.ui().equals("STD")) {
		        					getParams[i] = new JTextField();
		        					((JTextField) getParams[i]).setToolTipText(m.getParameters()[i].getName());
		        					paramPanel.add((JTextField) getParams[i]);	        						
	        					}
	        				}
	            		}
	                    jButton.addActionListener(new java.awt.event.ActionListener() {
	                        public void actionPerformed(java.awt.event.ActionEvent evt) {
	                        	//System.out.println("Clicked on "+m.getName());
	                        	//System.out.println("This function has "+nParameters+" parameters.");
	                        	Object[] args = new Object[nParameters];
	                        	for (int i=0;i<nParameters;i++) {
	                        		Class p = m.getParameterTypes()[i];
	                        		if ((p.equals(float.class))||(p.equals(Float.class))) {
	                        			System.out.print("arg["+i+"]=float=");
	                        			try {
	                        				args[i] = new Float(((JTextField) getParams[i]).getText());
	                        			} catch (NumberFormatException e) {
	                        				System.out.println("Argument "+i+" is not a number.");
	                        				break;
	                        			}
	                        			System.out.println(args[i].toString());
	                        		}
	                        		if ((p.equals(double.class))||(p.equals(Double.class))) {
	                        			System.out.print("arg["+i+"]=double=");
	                        			try {
	                        				args[i] = new Double(((JTextField) getParams[i]).getText());
	                        			} catch (NumberFormatException e) {
	                        				System.out.println("Argument "+i+" is not a number.");
	                        				break;
	                        			}
	                        			System.out.println(args[i].toString());

	                        		}
	                        		if ((p.equals(int.class))||(p.equals(Integer.class))) {
	                        			System.out.print("arg["+i+"]=int=");
	                        			try {
	                        				((JTextField) getParams[i]).validate();
	                        				args[i] = new Integer(((JTextField) getParams[i]).getText());//.intValue();
	                        			} catch (NumberFormatException e) {
	                        				System.out.println("Argument "+i+" is not a number.");	                        				
	                        				break;
	                        			}
	                        			System.out.println(args[i].toString());
	                        		}
	                        		if (p.equals(String.class)) {
	                        			if ((scriptableAnnotation.ui().equals("STD"))) {
		                        			//System.out.print("arg["+i+"]=String=");
		                        			args[i] = ((JTextField) getParams[i]).getText();
		                        			//System.out.println(args[i]);
	                        			} else {
	                        				System.out.print("arg["+i+"]=StringParticular=");
		                        			args[i] = ((JButton) getParams[i]).getToolTipText();
		                        			System.out.println(args[i]);
	                        			}
	                        		}
	                        	}
	                        	lms.handleExtension(m.getName(),args);
	                        }
	                    });
	            	}
            	}
            }
        }
    	System.out.println(panels.get(0).getName());
    	panels.sort(new PanelPrioritiesComparator());
    	System.out.println(panels.get(0).getName());
    	for (int i=0;i<panels.size();i++) {
    		JPanel panel = panels.get(i);
    		panel.setLayout(new java.awt.GridLayout(panel.getComponentCount()/2, 2));
    		jTabbedPanelMethods.addTab(panel.getName(), panel);
    		jTabbedPanelMethods.setComponentAt(i, panel);
    	}

        add(jTabbedPanelMethods);       
        myState = new JTextArea();
        myState.setText(getState());
        add(myState);       
    }
    
    private static final Map<String, Integer> panelPrioritiesMap;
    static {
        Map<String, Integer> aMap = new HashMap<String, Integer>();
        int cPrior=0;
        aMap.put(LimeSeg.IO,cPrior);cPrior++;
        aMap.put(LimeSeg.STATE,cPrior);cPrior++;
        aMap.put(LimeSeg.CURRENT_CELL,cPrior);cPrior++;
        aMap.put(LimeSeg.CURRENT_CELLT,cPrior);cPrior++;
        aMap.put(LimeSeg.CLIPPED_DOTS,cPrior);cPrior++;
        aMap.put(LimeSeg.OPT,cPrior);cPrior++;
        aMap.put(LimeSeg.VIEW_2D,cPrior);cPrior++;
        aMap.put(LimeSeg.VIEW_3D,cPrior);cPrior++;
        aMap.put(LimeSeg.CURRENT_DOT,cPrior);cPrior++;
        aMap.put(LimeSeg.BENCHMARK,cPrior);cPrior++;
        panelPrioritiesMap = Collections.unmodifiableMap(aMap);
    }   
    
    static class PanelPrioritiesComparator implements Comparator<JPanel>
    {
        public int compare(JPanel j1, JPanel j2)
        {	Integer p1=panelPrioritiesMap.get(j1.getName());
        	Integer p2=panelPrioritiesMap.get(j2.getName());
        	if (p1==null) {p1 = new Integer(-1); System.out.println(j1.getName());}
        	if (p2==null) {p2 = new Integer(-1); System.out.println(j2.getName());}        	
        	return p1.compareTo(p2);
        }
    } 
    
    static class MethodsPrioritiesComparator implements Comparator<Method>
    {
    	public int compare(Method m1,Method m2) {
    		Integer p1=0;
    		Integer p2=0;
    		if (m1.isAnnotationPresent(IJ1ScriptableMethod.class)) {
    			IJ1ScriptableMethod scriptableAnnotation = (IJ1ScriptableMethod)m1.getAnnotation(IJ1ScriptableMethod.class);
            	p1=scriptableAnnotation.pr();
    		}
    		if (m2.isAnnotationPresent(IJ1ScriptableMethod.class)) {
    			IJ1ScriptableMethod scriptableAnnotation = (IJ1ScriptableMethod)m2.getAnnotation(IJ1ScriptableMethod.class);
            	p2=scriptableAnnotation.pr();
    		}
    		if (m1.isAnnotationPresent(DisplayableOutput.class)) {
    			DisplayableOutput scriptableAnnotation = (DisplayableOutput)m1.getAnnotation(DisplayableOutput.class);
            	p1=scriptableAnnotation.pr();
    		}
    		if (m2.isAnnotationPresent(DisplayableOutput.class)) {
    			DisplayableOutput scriptableAnnotation = (DisplayableOutput)m2.getAnnotation(DisplayableOutput.class);
            	p2=scriptableAnnotation.pr();
    		}
    		
    		return p1.compareTo(p2);
    	}
    }
    
    public JFileChooser openFChooser() {
    	return openFChooser(false);
    }

    public JFileChooser openFChooser(boolean ChooseDir) {
        File f=null;
        JFileChooser fc=null;
        if (currentPath!=null) {
            f = new File (currentPath);
            if (f.exists()) {                
                fc = new JFileChooser(currentPath);
            } else {                
                fc = new JFileChooser();
            }
        } else {
            fc = new JFileChooser();
        }
        if (ChooseDir) {fc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);}
        return fc;
    }
    
    public static String currentPath;
    
    private javax.swing.JTabbedPane jTabbedPanelMethods;
}
