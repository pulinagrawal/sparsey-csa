package utils;

public class StatUtils {
	
	public static int sampleDistribution(double distribution[]){
		double randomNumber=Math.random();
		int sample=0;
		double pSum=0;
		for (int i = 0; i < distribution.length; i++) {
			pSum+=distribution[i];
			if(randomNumber<pSum){
				sample=i;
				break;
			}
		}
		
		return sample;
	}

	public static int sampleDistribution2(double distribution[]){
		double randomNumber=Math.random();
		int sample=0;
		for (int i = 0; i < distribution.length; i++) {
			if(randomNumber<=distribution[i]){
				sample=i;
				break;
			}
			else{
				randomNumber=distribution[i];
			}
		}
		
		return sample;
	}
	
	public static void main(String[] args){
		double distribution[]={0.1,0.4,0.2,.05,.25};
		int samplesFreq[]=new int[5];
		double means[]=new double[5];
		for(int j=0;j<1000;j++){
			for(int k=0;k<means.length;k++){
				samplesFreq[k]=0;
			}
			for(int i=0;i<1000;i++){
				samplesFreq[sampleDistribution2(distribution)]++;
			}
			for(int k=0;k<means.length;k++){
				means[k]+=((double)samplesFreq[k])/1000.0;
			}
		}
		for(int k=0;k<means.length;k++){
			means[k]/=1000.0;
			System.out.println(k+"="+means[k]);
		}

	}

}
