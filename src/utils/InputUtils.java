package utils;

import cern.colt.bitvector.BitVector;

public class InputUtils {
	
	public static void setInput(double[] input, BitVector vector){
		for (int j = 0; j < input.length; j++) {
			input[j]=vector.get(j)?1:0;
		}
	}

}
