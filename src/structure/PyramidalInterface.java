package structure;

public interface PyramidalInterface {
	
	void recieveInputs();
	double firstIntegration();
	void firstFire();
	void recieveNeuromodulator(double modulation);
	double secondIntegration();
	void secondFire();
}
