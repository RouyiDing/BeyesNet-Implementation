import java.util.HashMap;
import weka.core.Instance;
import weka.core.Instances;


/**
 * Use Naive Bayes method to train data and get classify result of test data.
 * 
 * @author Rouyi Ding
 *
 */
public class NaiveBayesClassifier {	
	
	HashMap<String, Integer> label1Map, label2Map;

	int l1InsCnt, l2InsCnt;		
	
	String label1, label2;
	
	

	/***
	 * Store the original training data set into two hashMaps. Count the number of 
	 * instances with the same label and attribute value.
	 * 
	 * @param trainingData
	 */	
	public void train(Instances trainingData, int attrIndex) {

		// initialize HashMaps
		label1Map = new HashMap<>();
		label2Map = new HashMap<>();
		l1InsCnt = 0;
		l2InsCnt = 0;
		// Index of the last attribute
		int classIndex = trainingData.classIndex();
		// set binary class labels
		setLabel(trainingData);

		for (int i = 0; i < trainingData.numInstances(); i++) {
			Instance curr = trainingData.instance(i); // current instance
			String currKey = curr.stringValue(attrIndex);

			// Instance with class1Label is count in class1Map, else is count in
			// class2Map
			if (curr.stringValue(classIndex).equals(label1)) {				
				l1InsCnt++;				
				if (!label1Map.containsKey(currKey)) {
					label1Map.put(currKey, 1);
				} else {
					int keyVal = label1Map.get(currKey) + 1;
					label1Map.put(currKey, keyVal);
				}

			} else if (curr.stringValue(classIndex).equals(label2)) {				
				l2InsCnt++;				
				if (!label2Map.containsKey(currKey)) {
					label2Map.put(currKey, 1);
				} else {
					int keyVal = label2Map.get(currKey) + 1;
					label2Map.put(currKey, keyVal);
				}
			}
		}
	}

	/**
	 * Returns the prior probability of each label P(label)
	 * 
	 * @param label
	 *        label of instances
	 * @return P(label)
	 */
	public double prior(String label){
		int total = l1InsCnt + l2InsCnt;
		int total_psd = total + 2;
		int l1InsCnt_psd = l1InsCnt + 1;
		int l2InsCnt_psd = l2InsCnt + 1;
		if (label.equals(label1)){
			return (double)l1InsCnt_psd / (double)total_psd;
		} else {			
			return (double)l2InsCnt_psd / (double)total_psd;
		}
		
	}

	/**
	 * Return the conditional probability of the attribute value given the
	 * label, i.e. P(attribute | label) 
	 * 
	 * @param trainingData
	 * @param attrIndex
	 * @param attrVal
	 * @param label
	 * 
	 * @return P(attribute | label)
	 */
	public double prob_given_label(Instances trainingData, int attrIndex, 
			String attrVal, String label) {
		int tmp = trainingData.attribute(attrIndex).numValues();
		
		// get hashMap for attribute[attrIndex]
		train(trainingData, attrIndex);

		if (label.equals(label1)) {
			if (label1Map.containsKey(attrVal)) {
				int numerator = label1Map.get(attrVal) + 1;
				return (double) numerator / ((double) l1InsCnt + (double) tmp);
			} else {
				return (double) 1 / ((double) l1InsCnt + (double) tmp);
			}
		} else {
			if (label2Map.containsKey(attrVal)) {
				int numerator = label2Map.get(attrVal) + 1;
				return (double) numerator / ((double) l2InsCnt + (double) tmp);
			} else {
				return (double) 1 / ((double) l2InsCnt + (double) tmp);
			}
		}
	}

	
	
	/**
	 * Classify an instance as either of label1 or label2
	 * 
	 * @param trainingData
	 * @param testData
	 * 
	 * @return classify result with predicted label and probability
	 */
	public ClassifyResult classify(Instances trainingData, Instance testData) {

		setLabel(trainingData);
		ClassifyResult cr = new ClassifyResult();
		double prior_label1;
		double prior_label2;
		double pos_label1 = 1;
		double pos_label2 = 1;

		// calculate posterior
		for (int i = 0; i < testData.numAttributes() - 1; i++) {
			// attribute value of the test data
			String val = testData.stringValue(testData.attribute(i));
			pos_label1 = pos_label1 * 
					prob_given_label(trainingData, i, val, label1);
			pos_label2 = pos_label2 * 
					prob_given_label(trainingData, i, val, label2);
		}

		prior_label1 = prior(label1);
		prior_label2 = prior(label2);
		pos_label1 = pos_label1 * prior_label1;
		pos_label2 = pos_label2 * prior_label2;

		if (pos_label1 >= pos_label2) {
			cr.pre_label = label1;
			cr.prob_label = pos_label1 / (pos_label1 + pos_label2);
		} else {
			cr.pre_label = label2;
			cr.prob_label = pos_label2 / (pos_label1 + pos_label2);
		}

		return cr;
	}

	
	/**
	 * set class labels from training data
	 * 
	 * @param trainingData
	 */
	public void setLabel(Instances trainingData) {
		label1 = trainingData.attribute(trainingData.classIndex()).value(0);
		label2 = trainingData.attribute(trainingData.classIndex()).value(1);

	}

}
