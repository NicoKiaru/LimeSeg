package eu.kiaru.limeseg.gui;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
/**
 * Annotation for LimeSeg command GUI
 * @author Nicolas Chiaruttini
 *
 */
@Retention(RetentionPolicy.RUNTIME)

public @interface DisplayableOutput {
	/**
	 * Category of command
	 * @return
	 */
	String target() default "Undefined";
	/**
	 * priority of command for ordering the display
	 * @return
	 */
	int pr() default 500;
}

