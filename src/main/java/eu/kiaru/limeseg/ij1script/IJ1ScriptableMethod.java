package eu.kiaru.limeseg.ij1script;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

@Retention(RetentionPolicy.RUNTIME)

public @interface IJ1ScriptableMethod {
	/**
	 * Specify a category for a method
	 * @return
	 */
	String target() default "Undefined";
	/**
	 * Display in GUI ?
	 * @return
	 */
	String ui() default "NO";
	/**
	 * Label text explaining how to use the method
	 * @return
	 */
	String tt() default "";
	/**
	 * Priority to order the display in GUI
	 * @return
	 */
	int pr() default 500;
	/**
	 * Should this method be executed in a separate thread ?
	 * @return
	 */
	boolean newThread() default false;
	
}
