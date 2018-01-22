package eu.kiaru.limeseg.commands;

import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import eu.kiaru.limeseg.LimeSeg;
/**
 * Triggers a stop optimisation request
 * Segmentation can be resumed with resumesegmentation command
 * @author Nicolas Chiaruttini
 *
 */
@Plugin(type = Command.class, menuPath = "Plugins>LimeSeg>Stop Optimisation")
public class StopOptimisation implements Command{

	@Override
	public void run() {
		LimeSeg.stopOptimisation();
	}

}
