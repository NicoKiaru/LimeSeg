package eu.kiaru.limeseg.commands;

import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import eu.kiaru.limeseg.LimeSeg;
/**
 * Clear all the segmented objects
 * @author Nicolas Chiaruttini
 */
@Plugin(type = Command.class, menuPath = "Plugins>LimeSeg>Clear all")
public class ClearAll implements Command{

	@Override
	public void run() {
		LimeSeg.clearAllCells();
	}
	
}
