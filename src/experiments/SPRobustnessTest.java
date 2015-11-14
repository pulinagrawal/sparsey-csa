package experiments;

import structure.MacroColumn;
import utils.InputUtils;
import cern.colt.bitvector.BitVector;

public class SPRobustnessTest {
	public static void main(String[] args){

		boolean longRepitions=false;
		int repitions=Integer.parseInt(args[0]);
		double size=500;/// number of features/dimensions/bits in input
		int representationDimensionality=2048;
		double sparsity=.02;//percentage Double.parseDouble(args[1]);
		double inputActivity=.35;//Double.parseDouble(args[2])/100;
		double noiseArg=0;//Double.parseDouble(args[3]);
		double noise=(double)((noiseArg/100)*size);
		int length=1000;//Integer.parseInt(args[4]); /// number of different categories of objects in dataset
		double card=inputActivity*size;
		double[] input=new double[(int)size];
		double[][] pRep=new double[length][(int)(representationDimensionality)];
		double[][] finalRep=new double[length][(int)(representationDimensionality)];

		int nMiniCol=(int) (representationDimensionality*sparsity);//Integer.parseInt(args[0]);
		int nPyramidalPerMiniCol=(int) (representationDimensionality/nMiniCol);//Integer.parseInt(args[0]);
		MacroColumn macroColumn= new MacroColumn(input, (int)nMiniCol, (int)nPyramidalPerMiniCol);

		//dataset with NxM input patterns with N different categories of object with M instances of each
		BitVector[] dataset1=new BitVector[(int)length];

		// Initialize Dataset Array
		for(int i=0;i<dataset1.length;i++){
			dataset1[i]=new BitVector((int) size);
			dataset1[i].clear();
		}

		// Setup Dataset / Build Datset
		for(int i=0;i<dataset1.length;i++){
			for(int j=0;j<size;j++){
				if(Math.random()<inputActivity)
					dataset1[i].set(j);
			}
		}



		//Initialize for comparision
		for (int i = 0; i < pRep.length; i++) {
			for(int j=0;j<pRep[0].length;j++)
				pRep[i][j]=0;
		}


		//Run system on dataset
		if (longRepitions==false){
			for (int i = 0; i < dataset1.length; i++) {


				InputUtils.setInput(input, dataset1[i]);
				//			System.out.print(i%dupl+",");

				macroColumn.run();	// execute
				pRep[i]=macroColumn.representation.clone();

				for(int j=0;j<repitions-1;j++){
					macroColumn.learn();
					macroColumn.setupForNextStep();
					macroColumn.run();	// execute

				}		
					macroColumn.learn();
					macroColumn.setupForNextStep();

				finalRep[i]=macroColumn.representation.clone();
			}
		}
		else{


			for(int j=0;j<repitions;j++){
				for (int i = 0; i < dataset1.length; i++) {
					InputUtils.setInput(input, dataset1[i]);
					//			System.out.print(i%dupl+",");

					macroColumn.run();	// execute
					if(j==0)
						pRep[i]=macroColumn.representation.clone();


					finalRep[i]=macroColumn.representation.clone();
					macroColumn.learn();
					macroColumn.setupForNextStep();

				}
				//Run system on test set

			}
		}
		int totalError=0;
		for(int i=0;i<dataset1.length;i++){
			for(int j=0;j<finalRep[0].length;j++)
				totalError+=Math.abs(finalRep[i][j]-pRep[i][j]);
		}

		System.out.println("Error is "+ ((double)totalError/dataset1.length));
	}
}
