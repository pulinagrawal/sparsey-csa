/**
 * 
 */
package structure;

import utils.StatUtils;

/**
 * @author PulinTablet
 *
 */
public class MiniColumn implements MiniColumnInterface{
	Pyramidal[] layer2; // Layer 2 of the 6 layers of a cortical column
	
	double vMax;
	double[] p;
	double n;
	
	MiniColumn(double[] input,int l2Pyramidals){
		layer2=new Pyramidal[l2Pyramidals];
		p=new double[l2Pyramidals];
		assignInputToL2Pyramidals(input);
	}
	
	void assignInputToL2Pyramidals(double[] input){
		for (int i = 0; i < layer2.length; i++) {
			layer2[i]=new Pyramidal(input);
		}
	}
	
	int getNoOfPyramidals(){
		return layer2.length;
	}
	
	void calcVMax(){
		vMax=0;
		for (int i = 0; i < layer2.length; i++) {
			double v=layer2[i].firstIntegration();
			layer2[i].firstFire();
			vMax=v>vMax?v:vMax;
		}
	}
	
	void calcP(){
		double sum=0;
		for (int i = 0; i < layer2.length; i++) {
			p[i]=layer2[i].secondIntegration();
			
			sum+=p[i];
		}
		for (int i = 0; i < layer2.length; i++) {
			p[i]=p[i]/sum;
		}
	}
	
	void setN(double cN){
		n=cN;
		calcP();
	}

	double getVMax(){
		return vMax;
	}
	
	void calcActivePyramidal(){
		for (int i = 0; i < layer2.length; i++) {
			layer2[i].setActive(false);
		}
		
		int index=StatUtils.sampleDistribution(p);
		layer2[index].setActive(true);
	}
	
	int getActivePyramidal(){
		for (int i = 0; i < layer2.length; i++) {
			if(layer2[i].isActive)
				return i;
		}
		return -1;
	}
	
	void learn(){
		for (int i = 0; i < layer2.length; i++) {
			if(layer2[i].isActive()){
				layer2[i].learn();
			}
		}
	}
	
	void setupForNextStep(){
		for (int i = 0; i < layer2.length; i++) {
			layer2[i].setActive(false);
		}
	}
	@Override
	public void recieveInputs() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public double firstFirePyramidal() {
		calcVMax();
		return getVMax();
		
	}

	@Override
	public void recieveNeuromodulator(double n) {
		
		for (Pyramidal p : layer2) {
			p.recieveNeuromodulator(n);
		}
		calcP();
	}

	@Override
	public int secondFirePyramidal() {
		calcActivePyramidal();
		return getActivePyramidal();
	}
}

