package experiments;

import structure.MacroColumn;
import utils.InputUtils;
import cern.colt.bitvector.BitVector;

public class RandomGeneration {
public static void main(String[] args){
		
		
		double size=15;/// number of features/dimensions/bits in input
		int nMiniCol=100;//Integer.parseInt(args[0]);
		double sparsity=2;//Double.parseDouble(args[1]);
		double inputActivity=.35;//Double.parseDouble(args[2])/100;
		double noiseArg=0;//Double.parseDouble(args[3]);
		double noise=(double)((noiseArg/100)*size);
		int length=5;//Integer.parseInt(args[4]); /// number of different categories of objects in dataset
		int dupl=3;//Integer.parseInt(args[5]);/// number of different instances of each object in the dataset
		double card=inputActivity*size;
		double nPyramidal=100/sparsity;
		double[] input=new double[(int)size];
		double[][] pRep=new double[length+1][(int)(nMiniCol*nPyramidal)];
		
		MacroColumn macroColumn= new MacroColumn(input, (int)nMiniCol, (int)nPyramidal);
		
		//dataset with NxM input patterns with N different categories of object with M instances of each
		BitVector[][] dataset1=new BitVector[(int)length][(int)dupl];
		
		// Initialize Dataset Array
		for(int i=0;i<dataset1.length;i++){
			for(int j=0;j<dupl;j++){
			dataset1[i][j]=new BitVector((int) size);
			dataset1[i][j].clear();
			}
		}
		
		// Setup Dataset / Build Datset
		for(int i=0;i<dataset1.length;i++){
				for(int j=0;j<card;j++){
					int index=(int) ((double)Math.random()*size); //generate a position to set bit
					// set all the bits at 'index' of same instances of that category of input
					for(int k=0;k<dupl;k++){
						try{
						dataset1[i][k].set(index);
						}catch(ArrayIndexOutOfBoundsException e){
							System.err.println("Array Index Out of Bounds at:"+k);
						}
						
					}
					
				}
				//add noise
				for(int k=1;k<dupl;k++){
					for(int l=0;l<noise;l++){
						int index1=(int)((double)Math.random()*size);
						dataset1[i][k].put(index1, !dataset1[i][k].get(index1));
					}
				}
				
		}

		
		//Initialize for comparision
		for (int i = 0; i < pRep.length; i++) {
			for(int j=0;j<pRep[0].length;j++)
			pRep[i][j]=0;
		}
		
		
		//Run system on dataset
		for (int i = 0; i < dataset1.length; i++) {
			for(int k=0;k<dupl-1;k++){
				int count =0;
				
				
				InputUtils.setInput(input, dataset1[i][k]);
	//			System.out.print(i%dupl+",");
				
				macroColumn.run();	// execute
				int repCard=0;
				for (int j = 0; j < macroColumn.representation.length; j++) {
					//Print space to end microcolumn
	//				if(j%nPyramidal==0){ System.out.print(" ");	}
	//				System.out.print((int)macroColumn.representation[j]);
					
					if(macroColumn.representation[j]==1){
						repCard++;
						if(pRep[i][j]!= macroColumn.representation[j]) count++;
					}
					
				}
	//			System.out.println();
				BitVector noise1=(BitVector) dataset1[i][k].clone();
				if(k>0) noise1.xor(dataset1[i][k-1]);
	//			System.out.println("Noise percentage:"+(double)noise1.cardinality()/noise1.size()+", Representation Difference:"+(double)count/repCard);
				
				pRep[i+1]=macroColumn.representation.clone();//TODO made the change of i to i+1 pRep is earlier compared using the current index
				macroColumn.learn();
				macroColumn.setupForNextStep();
			}
		}
		
		//Run system on test set
		System.out.println("Input,G,Noise,RDiff");
		for (int i = 0; i < dataset1.length; i++) {
			int k=dupl-1;
				int count =0;
				for (int j = 0; j < input.length; j++) {
					input[j]=dataset1[i][k].get(j)?1:0;
				}
				System.out.print(i%dupl+",");
				
				macroColumn.run();	// execute
				int repCard=0;
				for (int j = 0; j < macroColumn.representation.length; j++) {
					//Print space to end microcolumn
		//			if(j%nPyramidal==0){ System.out.print(" ");	}
		//			System.out.print((int)macroColumn.representation[j]);
					if(macroColumn.representation[j]==1){
						repCard++;
						if(pRep[i][j]!= macroColumn.representation[j]) count++;
					}
				}

				System.out.print(macroColumn.g+",");
		//		System.out.println(repCard);
				BitVector noise1=(BitVector) dataset1[i][k].clone();
				if(k>0) noise1.xor(dataset1[i][k-1]);
		//		System.out.print("Noise percentage:");
				System.out.print((double)noise1.cardinality()/noise1.size()+",");
				System.out.println((double)count/repCard+","); //Representation difference
				
		//		pRep[i]=macroColumn.representation.clone();
		//		macroColumn.learn();
				macroColumn.setupForNextStep();
			}
		
	}

}
