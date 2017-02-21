import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;



public class Bayes {

	/**
	 * Creates a fresh instance of the classifier.
	 * 
	 * @return a classifier
	 */
	private static NaiveBayesClassifier getBayesClassifier() {
		NaiveBayesClassifier nbc = new NaiveBayesClassifier();
		return nbc;
	}
	
	/**
	 * Creates a fresh instance of the classifier.
	 * 
	 * @return	a classifier
	 */
		private static TanClassifier getTanClassifier() {
			TanClassifier nbc = new TanClassifier();
			return nbc;
		}	
	
	
		
	/**
	 * Calculate the accuracy rate of the test data classify result.
	 * 
	 * @param trainingData
	 * @param testData
	 * @param method
	 * @return accuracy rate
	 */
	public static double getAccuracy(Instances trainingData, Instances testData,
			String method) {
		int trueCnt = 0;
		int total = 0;
		if (method.equals("NaiveBayes")) {
			NaiveBayesClassifier bayesClassifier = getBayesClassifier();
			for (int ins = 0; ins < testData.numInstances(); ins++) {
				ClassifyResult output = bayesClassifier.classify(trainingData,
						testData.instance(ins));
				total++;
				if (output.pre_label.equals(testData.instance(ins).stringValue(
						testData.classIndex()))) {
					trueCnt++;
				}
			}
			return (double) trueCnt / (double) total;
		} else {
			TanClassifier tanClassifier = getTanClassifier();
			for (int ins = 0; ins < testData.numInstances(); ins++) {
				ClassifyResult output = tanClassifier.classify(trainingData, 
						testData.instance(ins));
				total++;
				if (output.pre_label.equals(testData.instance(ins).stringValue(
						testData.classIndex()))) {
					trueCnt++;
				}
			}
			return (double) trueCnt / (double) total;
		}
	}
	
	
		
	/**
	 * Main method reads command-line flags and outputs NaiveBayes Classifier
	 * result.
	 * 
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		if(args.length != 3) {
			System.err.println("Invalid arguments");
			return;
		}
		
		
		// read training data
		BufferedReader reader = new BufferedReader(new FileReader(args[0]));
		Instances trainingData = new Instances(reader);
		reader.close();
		// setting class attribute
		trainingData.setClassIndex(trainingData.numAttributes() - 1);
		
		
		
		// read test data
		reader = new BufferedReader(new FileReader(args[1]));
		Instances testData = new Instances(reader);
		reader.close();
		// setting class attribute
		testData.setClassIndex(testData.numAttributes() - 1);
		
		
		if(args[2].equals("n")) {
			// print predicted label and its probability by using naiveBayes
			
			NaiveBayesClassifier bayesClassifier = getBayesClassifier();
			
			for (int i = 0; i < testData.numAttributes()-1; i++){
				System.out.println(testData.attribute(i).toString().
						split(" ")[1] + " " + "class");
			}
			System.out.println();
			int nbTrueCnt = 0;
			for (int ins = 0; ins < testData.numInstances(); ins++){
				ClassifyResult output = bayesClassifier.classify(trainingData,
						testData.instance(ins));
				if (output.pre_label.equals(testData.instance(ins).stringValue
						(testData.classIndex()))){
					nbTrueCnt++;
				}
				System.out.print(output.pre_label + " " + testData.
						instance(ins).stringValue(testData.classIndex()) + " ");
				System.out.println(String.format("%.12f",output.prob_label));
			} 
			System.out.println();
			System.out.println(nbTrueCnt);	
		}
		else {
			// print predicted label and its probability by using TAN
			TanClassifier tanClassifier = getTanClassifier();
			ArrayList<Node> nodes = tanClassifier.train(trainingData);
			for (int i = 0; i < nodes.size(); i++){
				if (nodes.get(i).parentIndex == -1){
					System.out.println(testData.attribute(i).toString().
							split(" ")[1]+ " " + "class");
				} else {
					int parIdx = nodes.get(i).parentIndex;
					String par = testData.attribute(parIdx).toString().
							split(" ")[1];
					System.out.println(testData.attribute(i).toString().
							split(" ")[1]+" "+par+" "+"class");
				}		
			}
			System.out.println();
			int tanTrueCnt = 0;
			for (int ins = 0; ins < testData.numInstances(); ins++){ 
				ClassifyResult output = tanClassifier.classify(trainingData, 
						testData.instance(ins));
				if (output.pre_label.equals(testData.instance(ins).
						stringValue(testData.classIndex()))){
					tanTrueCnt++;
				}
				System.out.print(output.pre_label + " " +
				testData.instance(ins).stringValue(testData.classIndex())+ " ");
				System.out.println(String.format("%.12f",output.prob_label));
			}
			System.out.println();
			System.out.println(tanTrueCnt);
		}
		
		
		
		
/**		
		// plot learning curve
		
		int[] sampleSize = new int[]{25,50,100};
		double [] accuRate = new double[3];
		
		for (int i = 0; i < sampleSize.length; i++){
			for (int k = 0; k < 4; k++){
			accuRate[i] = accuRate[i] + getAccuracy(RandSampling.getSampling(
			trainingData, sampleSize[i]), testData, "NaiveBayes");
			}
			accuRate[i] = accuRate[i]/4;
		}
		LearningCurve learningCurve = new LearningCurve("Learning Curve",
		 "NaieBayes");
		learningCurve.draw(accuRate, sampleSize,"NaiveBayes");
		System.out.println(accuRate[0]);
		System.out.println(accuRate[1]);
		System.out.println(accuRate[2]);
		
		double [] accuRate1 = new double[3];
		for (int i = 0; i < sampleSize.length; i++){
			for (int k = 0; k < 4; k++){
			accuRate1[i] = accuRate1[i] + getAccuracy(RandSampling.getSampling(
			trainingData, sampleSize[i]), testData, "TAN");
			}
			accuRate1[i] = accuRate1[i]/4;
		}
		LearningCurve learningCurve2 = new LearningCurve("Learning Curve", "NaieBayes");
		learningCurve2.draw(accuRate1, sampleSize,"TAN");
		System.out.println(accuRate1[0]);
		System.out.println(accuRate1[1]);
		System.out.println(accuRate1[2]);
	
*/		
	}
}


