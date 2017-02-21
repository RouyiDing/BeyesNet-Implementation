import java.util.HashMap;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Edge represents the connection between each pair of attributes.
 * 
 * @author Rouyi Ding
 *
 */
public class Edge implements Comparable<Edge> {

	int attrIndex1;
	int attrIndex2;
	double weight;
	Instances trainingData;
	/**
	 * Stores the Counts of instances having the same value of attribute 1,
	 * attribute 2 and label
	 */
	HashMap<String, HashMap<String, HashMap<String, Integer>>> hMap;

	
	public Edge(int attrIndex1, int attrIndex2, Instances trainingData) {
		this.attrIndex1 = attrIndex1;
		this.attrIndex2 = attrIndex2;
		this.trainingData = trainingData;
	}

	/**
	 * Set the weight of each pair of nodes
	 * 
	 * @param edge
	 */
	public void setWeight(Edge edge) {
		int idx1 = edge.attrIndex1;
		int idx2 = edge.attrIndex2;
		int numLabels = trainingData.numClasses();
		int attr1_numVal = trainingData.attribute(idx1).numValues();
		int attr2_numVal = trainingData.attribute(idx2).numValues();
		this.weight = 0;
		setHashMap(idx1, idx2);
		for (int k = 0; k < numLabels; k++) {
			String label_val = trainingData.attribute(
					trainingData.classIndex()).value(k);
			
			for (int j = 0; j < attr1_numVal; j++) {
				String attr1_val = trainingData.attribute(idx1).value(j);
				
				for (int i = 0; i < attr2_numVal; i++) {
					String attr2_val = trainingData.attribute(idx2).value(i);
					double tmp = attr1and2_prob_given_label(idx1, attr1_val,
							idx2, attr2_val, label_val)/(attr1_prob_given_label(
									idx1, attr1_val, idx2, label_val) *
									attr2_prob_given_label(idx1, idx2, 
											attr2_val, label_val));
					this.weight = this.weight + attr1and2andlabel_prob(idx1, 
							attr1_val, idx2, attr2_val, label_val) * 
							Math.log(tmp) / Math.log(2.0);
				}
			}
		}
	}

	/**
	 * Count the number of instances with the same attribute_1, attribute_2 and
	 * label. Then Store all counts into a three layers hashMap.
	 * 
	 * @param attr1_Index
	 * @param attr2_Index
	 */
	public void setHashMap(int attr1_Index, int attr2_Index) {
		hMap = new HashMap<>();
		int attr1_numVal = trainingData.attribute(attr1_Index).numValues();
		int attr2_numVal = trainingData.attribute(attr2_Index).numValues();
		int label_numVal = trainingData.numClasses();

		// Initialize three layers hashMap
		for (int i = 0; i < label_numVal; i++) {
			String key1 = 
					trainingData.attribute(trainingData.classIndex()).value(i);
			hMap.put(key1, new HashMap<String, HashMap<String, Integer>>());

			for (int j = 0; j < attr1_numVal; j++) {
				String key2 = trainingData.attribute(attr1_Index).value(j);
				hMap.get(key1).put(key2, new HashMap<String, Integer>());

				for (int k = 0; k < attr2_numVal; k++) {
					String key3 = trainingData.attribute(attr2_Index).value(k);
					hMap.get(key1).get(key2).put(key3, 0);
				}
			}
		}

		// count instance has the same value of attribute 1, attribute 2 and
		// label. Store counts in hashMap
		for (int i = 0; i < trainingData.numInstances(); i++) {
			//Current instance 
			Instance curr = trainingData.instance(i);
			//Current instance's value at attr1_Index 
			String attr1_val = curr.stringValue(attr1_Index);
			//Current instance's value at attr2_Index 
			String attr2_val = curr.stringValue(attr2_Index);
			//Current instance's label 
			String label_val = curr.stringValue(curr.classIndex());
			//Current count value in hashMap 
			int cnt = hMap.get(label_val).get(attr1_val).get(attr2_val);

			hMap.get(label_val).get(attr1_val).put(attr2_val, cnt + 1);
		}
	}

	
	
	/**
	 * 
	 * Calculate the conditional probability P(attritue_1 | label)
	 * 
	 * @param attr1_Index
	 * @param attr1_val
	 * @param attr2_Index
	 * @param label_val
	 * 
	 * @return P(attritue_1 | label)
	 */
	public double attr1_prob_given_label(int attr1_Index, String attr1_val,
			int attr2_Index, String label_val) {
		int cnt = 0;
		int total = 0;
		for (int j = 0; j < trainingData.attribute(attr1_Index).numValues(); 
				j++) {
			String key1 = trainingData.attribute(attr1_Index).value(j);		
			for (int i = 0; i < trainingData.attribute(attr2_Index).numValues();
					i++) {
				String key2 = trainingData.attribute(attr2_Index).value(i);			
				total = total + hMap.get(label_val).get(key1).get(key2);
				if (key1.equals(attr1_val)) {
					cnt = cnt + hMap.get(label_val).get(attr1_val).get(key2);
				}
			}
		}
		// Use Laplace estimates
		double numerator = cnt + 1;
		double denominator = total +
				trainingData.attribute(attr1_Index).numValues();
		return numerator / denominator;
	}

	
	/**
	 * Calculate the conditional probability P(attritue_2 | label)
	 * 
	 * @param attr1_Index
	 * @param attr2_Index
	 * @param attr2_val
	 * @param label_val
	 * 
	 * @return P(attritue_2 | label)
	 */
	public double attr2_prob_given_label(int attr1_Index, int attr2_Index,
			String attr2_val, String label_val) {
		int cnt = 0;
		int total = 0;
		
		for (int j = 0; j < trainingData.attribute(attr1_Index).numValues(); 
				j++) {
			String key1 = trainingData.attribute(attr1_Index).value(j);
			for (int i = 0; i < trainingData.attribute(attr2_Index).numValues();
					i++) {
				String key2 = trainingData.attribute(attr2_Index).value(i);
				total = total + hMap.get(label_val).get(key1).get(key2);
				if (key2.equals(attr2_val)) {
					cnt = cnt + hMap.get(label_val).get(key1).get(key2);
				}
			}
		}
		double numerator = cnt + 1;
		double denominator = total + 
				trainingData.attribute(attr2_Index).numValues();
		return numerator / denominator;
	}

	
	
	/**
	 * Calculate the conditional probability P(attribute_1, attritue_2 | label)
	 * 
	 * @param attr1_Index
	 * @param attr1_val
	 * @param attr2_Index
	 * @param attr2_val
	 * @param label_val
	 * 
	 * @return P(attribute_1, attritue_2 | label)
	 */
	public double attr1and2_prob_given_label(int attr1_Index, String attr1_val,
			int attr2_Index, String attr2_val, String label_val) {
		int cnt = 0;
		int total = 0;
		for (int j = 0; j < trainingData.attribute(attr1_Index).numValues(); 
				j++) {
			String key1 = trainingData.attribute(attr1_Index).value(j);
			
			for (int i = 0; i < trainingData.attribute(attr2_Index).numValues(); 
					i++) {
				String key2 = trainingData.attribute(attr2_Index).value(i);
				total = total + hMap.get(label_val).get(key1).get(key2);
				
				if (key1.equals(attr1_val) && key2.equals(attr2_val)) {
					cnt = cnt + hMap.get(label_val).get(key1).get(key2);
				}
			}
		}
		double numerator = cnt + 1;
		double denominator = total+ trainingData.attribute(attr1_Index).
				numValues() * trainingData.attribute(attr2_Index).numValues();

		return numerator / denominator;
		
		
	}

	
	/**
	 * Calculate the joint probability P(attribute_1, attritue_2, label)
	 * 
	 * @param attr1_Index
	 * @param attr1_val
	 * @param attr2_Index
	 * @param attr2_val
	 * @param label_val
	 * 
	 * @return P(attribute_1, attritue_2, label)
	 */
	public double attr1and2andlabel_prob(int attr1_Index, String attr1_val, 
			int attr2_Index, String attr2_val, String label_val) {
		int cnt = hMap.get(label_val).get(attr1_val).get(attr2_val);
		int total = trainingData.numInstances();
		double numerator = cnt + 1;
		double denominator = total + 
				trainingData.attribute(attr1_Index).numValues()* 
				trainingData.attribute(attr2_Index).numValues() * 
				trainingData.numClasses();
		
		return numerator / denominator;
	}

	
	/**
	 * Calculate the conditional probability P(attribute_2 | attritue_1, label)
	 * @param attr1_Index
	 * @param attr1_val
	 * @param attr2_Index
	 * @param attr2_val
	 * @param label_val
	 * @return P(attribute_2 | attritue_1, label)
	 */
	public double attr2_prob_given_attr1andLabel(int attr1_Index, 
			String attr1_val, int attr2_Index, String attr2_val,
			String label_val) {
		int cnt = hMap.get(label_val).get(attr1_val).get(attr2_val);
		int total = 0;
		
		for (int i = 0; i < trainingData.attribute(attr2_Index).numValues();
				i++) {
			String key = trainingData.attribute(attr2_Index).value(i);
			total = total + hMap.get(label_val).get(attr1_val).get(key);
		}
		double numerator = cnt + 1;
		double denominator = total +
				trainingData.attribute(attr2_Index).numValues();

		return numerator / denominator;
	}

	
	
	/**
	 * Calculate the prior probability P(label)
	 * 
	 * @param label_val
	 * 
	 * @return P(label)
	 */
	public double prior(String label_val) {
		int numerator = 1;
		int denominator = trainingData.numInstances() + 2;

		for (int i = 0; i < trainingData.numInstances(); i++) {
			Instance curr = trainingData.instance(i);
			if (curr.stringValue(curr.classIndex()).equals(label_val)) {
				numerator++;
			}
		}
		return (double) numerator / (double) denominator;
	}

	
	
	/**
	 * Compare a given Edge with current(this) object.If current edge's weight
	 * is larger than given Edge, return positive value. If current edge's
	 * weight is smaller, return negative value. If equal, return zero.
	 */
	public int compareTo(Edge otherEdge) {
		if (this.weight > otherEdge.weight) {
			return 1;
		} else if (this.weight < otherEdge.weight) {
			return -1;
		} else {
			return 0;
		}
	}

}
