/**
 * 
 */
package structure;

/**
 * Try to make an abstract pyramidal cell structure that can be used for 
 * Layer 2/3 and Layer 5
 * @author PulinTablet
 *
 */
public class Pyramidal implements PyramidalInterface{
	double[] synapses;
	double[] input;
	double l=28,f=-5;// it used to be -14 during the first long break i took on this.
	/**
	 * aggregate of input
	 */
	long u; 
	
	/**
	 * normalized aggregate (u)
	 */
	double v;
	
	public void setV(double v) {
		this.v = v;
	}

	/**
	 * relative likelihood of this pyramidal being winner
	 */
	double psi;
	
	/**
	 * probability of getting active
	 */
	double prob;
	
	double n;
	
	boolean isActive;
	
	public boolean isActive() {
		return isActive;
	}

	public void setActive(boolean isActive) {
		this.isActive = isActive;
	}

	Pyramidal(double[] cInput){
		input=cInput;
		recieveInputs();
		synapses=new double[cInput.length];
		initializeSynapses();
	}
	
	void initializeSynapses(){
		for(int i=0;i<synapses.length;i++)
			synapses[i]=0;
	}
	
	void calcV(){
		int countActiveInput=0;
		u=0;
		for(int i=0;i<synapses.length;i++)
			if(synapses[i]>0){
				countActiveInput++;
				u+=input[i];
			}
		v=countActiveInput>0?(double)u/((double)countActiveInput):0;
	}
	
	double getV(){
		return v;
	}
	
	void calcPsi(){
		
		psi= 1+(n/(1+Math.exp(-((l*v)+f))));
	}
	
	double getPsi(){
		return psi;
	}
	
	void learn(){
		for (int i = 0; i < synapses.length; i++) {
			if(input[i]>0){
				synapses[i]=1.0;
			}
		}
	}

	
	@Override
	public void recieveInputs() {
		// TODO Do any changes to input synapse configuration
		
	}

	@Override
	public double firstIntegration() {
		calcV();
		return getV();
	}

	@Override
	public void firstFire() {
			
	}


	@Override
	public void recieveNeuromodulator(double modulation) {
		n=modulation;
	}

	@Override
	public double secondIntegration() {
		calcPsi();
		return getPsi();
	}

	@Override
	public void secondFire() {
		setActive(isActive);
		
	}

	
}

