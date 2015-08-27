package utils;

import cern.colt.bitvector.BitVector;

public class VectorUtils {
	
	public static void setInput(double[] input, BitVector vector){
		for (int j = 0; j < input.length; j++) {
			input[j]=vector.get(j)?1:0;
			System.out.print((int)input[j]+" ");
		}
		System.out.println();
	}

	public static void setOutput(BitVector output, double[] vector){
		output.clear();
		for (int j=0;j<vector.length;j++){
			if(vector[j]==1)
				output.set(j);
		}
	}	

	public static BitVector getAvgVector(BitVector[] vector){
		BitVector output=new BitVector(vector[0].size());
		double[] avgVector=new double[vector[0].size()];
		for(int j=0;j<vector[0].size();j++){
			for(int i=0;i<vector.length;i++)
				avgVector[j]+=vector[i].get(j)?1:0;
			avgVector[j]/=vector.length;
			avgVector[j]=avgVector[j]<0.5?0:1;
		}
		VectorUtils.setOutput(output, avgVector);
		return output;
	}
	
	public static void printVector(BitVector vector){
		for(int i=0;i<vector.size();i++){
			System.out.print(" "+(vector.get(i)?1:0));
		}
		System.out.println();
	}

}
