package experiments;

import structure.MacroColumn;
import utils.VectorUtils;
import cern.colt.bitvector.BitVector;

public class RandomGeneration {

	double[] input;
	public MacroColumn macroColumn;
	
	public RandomGeneration(){

		double inputDimensionality = 20;// / number of features/dimensions/bits
										// in input
		int nMiniCol = 5;// Integer.parseInt(args[0]);

		double sparsity = 2;// Double.parseDouble(args[1]);
		double inputActivity = .1;// Double.parseDouble(args[2])/100;
		double noiseArg = 0;// Double.parseDouble(args[3]);

		double noise = (double) ((noiseArg / 100) * inputDimensionality);
		int entities = 5;// Integer.parseInt(args[4]); /// number of different
							// categories of objects in dataset
		int copies = 3;// Integer.parseInt(args[5]);/// number of different
						// instances of each object in the dataset
		double cardinality = inputActivity * inputDimensionality;
		double nPyramidal = 5;// 100/sparsity;
		double[] input = new double[(int) inputDimensionality];

		macroColumn = new MacroColumn(input, (int) nMiniCol,
				(int) nPyramidal);

		BitVector[][] dataset1 = setupDataset(inputDimensionality, noise,
				entities, copies, cardinality);

		BitVector output[][] = runDataset(input, macroColumn, dataset1);
			System.out.println("Average Output Vectors");
		for(int i=0;i<output.length;i++){
			BitVector avgOuput= VectorUtils.getAvgVector(output[i]);
			
			System.out.println("Entity "+i+":");
			VectorUtils.printVector(avgOuput);
		}
	
		System.out.println("All Output Vectors");
		for(int i=0;i<output.length;i++){
			System.out.println("Entity "+i+":");
			System.out.println("Input");
			for(int j=0;j<output[0].length;j++){
				VectorUtils.printVector(dataset1[i][j]);
			}
			System.out.println("Output");
			for(int j=0;j<output[0].length;j++){
				VectorUtils.printVector(output[i][j]);
			}
			System.out.println();
			
		}
		
		double avgOutputDiff[]=new double[output.length];
		for(int i=0;i<output.length;i++){
			for(int j=1;j<output[0].length;j++){
				BitVector clone=(BitVector) output[i][0].clone();
				clone.xor(output[i][j]);
				double diff=clone.cardinality();
				avgOutputDiff[i]+=diff;
			}
			avgOutputDiff[i]/=output[0].length-1;
			System.out.println("Average Ouput Difference for "+output[0].length+" outputs of "+i+" is "+avgOutputDiff[i]);
		}
		

	}

	public static void runTest() {

	}

	public static void main(String[] args) 
	{
		new RandomGeneration();
	}

	public static BitVector[][] setupDataset(double size, double noise,
			int length, int dupl, double card) {
		// dataset with NxM input patterns with N different categories of object
		// with M instances of each
		BitVector[][] dataset1 = new BitVector[(int) length][(int) dupl];

		// Initialize Dataset Array
		for (int i = 0; i < dataset1.length; i++) {
			for (int j = 0; j < dupl; j++) {
				dataset1[i][j] = new BitVector((int) size);
				dataset1[i][j].clear();
			}
		}

		// Setup Dataset / Build Datset
		for (int i = 0; i < dataset1.length; i++) {
			for (int j = 0; j < card; j++) {
				int index = (int) ((double) Math.random() * size); // generate a
																	// position
																	// to set
																	// bit
				// set all the bits at 'index' of same instances of that
				// category of input
				for (int k = 0; k < dupl; k++) {
					try {
						dataset1[i][k].set(index);
					} catch (ArrayIndexOutOfBoundsException e) {
						System.err.println("Array Index Out of Bounds at:" + k);
					}

				}

			}
			// add noise
			for (int k = 1; k < dupl; k++) {
				for (int l = 0; l < noise; l++) {
					int index1 = (int) ((double) Math.random() * size);
					dataset1[i][k].put(index1, !dataset1[i][k].get(index1));
				}
			}

		}
		return dataset1;
	}

	public static BitVector[][] runDataset(double[] input,
			MacroColumn macroColumn, BitVector[][] dataset1) {

		BitVector[][] output = new BitVector[dataset1.length][dataset1[0].length];

		// Initialize the output BitVector
		for (int i = 0; i < dataset1.length; i++)
			for (int k = 0; k < dataset1[0].length; k++)
				output[i][k] = new BitVector(macroColumn.representation.length);

		// Run system on dataset

		for (int i = 0; i < dataset1.length; i++) {
			for (int k = 0; k < dataset1[0].length; k++) {

				VectorUtils.setInput(input, dataset1[i][k]);

				macroColumn.run(); // execute

				VectorUtils.setOutput(output[i][k], macroColumn.representation);

				macroColumn.learn();

				macroColumn.setupForNextStep();
			}
		}
		return output;
	}

}
