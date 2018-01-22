package eu.kiaru.limeseg.commands;

import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import eu.kiaru.limeseg.LimeSeg;
/**
 * Restores state before last optimization
 * @author Nicolas Chiaruttini
 *
 */
@Plugin(type = Command.class, menuPath = "Plugins>LimeSeg>Cancel Last Optimisation")
public class CancelLastOptimisation implements Command{

	@Override
	public void run() {
		if ((LimeSeg.optimizerIsRunning==false)&&(LimeSeg.opt!=null)) {
			LimeSeg.restoreOptState();
			LimeSeg.clearOverlay();
		}
	}

}
