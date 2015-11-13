/**
 * 
 */
package structure;

import cern.colt.bitvector.BitVector;

/**
 * @author PulinTablet
 *
 */
public class MacroColumn implements MacroColumnInterface{
	
	double K=100,B=800,m=.95,v=60; // parameters used in function calcN()
	
	MiniColumn[] miniColumns;
	
	public double[] representation;
	/**
	 * macrocolumn's familiarity to current input
	 */
	public double g;
	
	/**
	 * eta (greek variable)
	 */
	double n;
	
	public MacroColumn(double[] input,int nMiniColumns, int nPyramidalsPerMiniColumn){
		miniColumns= new MiniColumn[nMiniColumns];
		representation= new double[nMiniColumns*nPyramidalsPerMiniColumn];
		initializeMiniColumns(input,nPyramidalsPerMiniColumn);
	}
	
	void initializeMiniColumns(double[] input, int nPyramidalsPerMiniColumn){
		for (int i = 0; i < miniColumns.length; i++) {
			miniColumns[i]=new MiniColumn(input, nPyramidalsPerMiniColumn);
		}
	}
	
	void calcG(){
		g=0;
		for (MiniColumn miniCol : miniColumns) {
			g+=miniCol.firstFirePyramidal();
		} 
		g/=(double)miniColumns.length;
		System.out.println("G:"+g);
	}
	
	void calcN(double cG){
		
		
		n=K/Math.pow(1+Math.exp((-B)*(g-m)),1.0/v);
	//	System.out.print("n:"+n);
		//n=( e/(a + Math.exp(-((l*g)+f)) ));
	}
	
	void giveN(){
		for (MiniColumn miniCol : miniColumns) {
			miniCol.recieveNeuromodulator(n);
		}
	}
	
	void buildRepresentation(){
		for (int i = 0; i < miniColumns.length; i++) {
			int index=miniColumns[i].secondFirePyramidal();
			representation[i*miniColumns[i].getNoOfPyramidals()+index]=1.0;
		}
	}
	
	public void learn(){
		for (int i = 0; i < miniColumns.length; i++) {
			miniColumns[i].learn();
		}
	}
	
	public void setupForNextStep(){
		for (int i = 0; i < representation.length; i++) {
			representation[i]=0;
		}
		for (int i = 0; i < miniColumns.length; i++) {
			miniColumns[i].setupForNextStep();
		}
	}
	
	
	public void run(){
		calcG();
		calcN(g);
		releaseNeuroModulator();
		buildRepresentation();
	}
	
	

	@Override
	public void recieveInput() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void releaseNeuroModulator() {
		giveN();
		
	}
}
