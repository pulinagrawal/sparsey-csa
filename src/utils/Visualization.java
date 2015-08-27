package utils;

import java.awt.EventQueue;

import javax.swing.JFrame;

import structure.MacroColumn;
import experiments.RandomGeneration;

public class Visualization {

	MacroColumn macroColumn;
	private JFrame frame;

	/**
	 * Launch the application.
	 */
	public static void main(String[] args) {
		final RandomGeneration object=new RandomGeneration();
		EventQueue.invokeLater(new Runnable() {
			public void run() {
				try {
					Visualization window = new Visualization();
					window.macroColumn=object.macroColumn;
					window.frame.setVisible(true);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
	}

	/**
	 * Create the application.
	 */
	public Visualization() {
		initialize();
	}

	/**
	 * Initialize the contents of the frame.
	 */
	private void initialize() {
		frame = new JFrame();
		frame.setBounds(100, 100, 450, 300);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

}
