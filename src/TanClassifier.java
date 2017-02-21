import java.util.ArrayList;
import java.util.HashMap;
import weka.core.Instance;
import weka.core.Instances;


/**
 *  Use TAN algorithm to train data and get classify result of test data.
 *  
 * @author Rouyi Ding
 *
 */
public class TanClassifier{
	
	/**Count instances have the same value of attribute 1 and 2 and label */
	HashMap<String, HashMap<String, HashMap<String, Integer>>> hMap;

	
	
	/**
	 * Train data and use Prims's algorithm to find a maximal spanning tree.
	 * 
	 * @param trainingData
	 * 
	 * @return a list of nodes with parent attribute index information
	 */
	public ArrayList<Node> train(Instances trainingData){
		MST mst = new MST();
		// store parent information of each attribute, Attritue0 has no parent, 
		// whose parent index is given by default.
		ArrayList<Node> nodes = mst.getMST(trainingData); 
		return nodes;
	}
	
	
	/**
	 * Classify an instance as either of label1 or label2
	 * 
	 * @param trainingData
	 * @param testData
	 * 
	 * @return classify result with predicted label and probability
	 */
	public ClassifyResult classify(Instances trainingData, Instance testData){
		
		ClassifyResult cr = new ClassifyResult();
		ArrayList<Node> nodes = train(trainingData);
		String label1 = trainingData.attribute(trainingData.classIndex()).value(0);
		String label2 = trainingData.attribute(trainingData.classIndex()).value(1);
		
		// first node in nodes
		Edge node1 = new Edge(0, 1, trainingData);
		node1.setHashMap(0,1);
		double pos_label1 = node1.prior(label1) * node1.attr1_prob_given_label(0, testData.stringValue(testData.attribute(0)), 1, label1);
		double pos_label2 = node1.prior(label2) * node1.attr1_prob_given_label(0, testData.stringValue(testData.attribute(0)), 1, label2);

		for (Node i : nodes.subList(1, nodes.size())){
			int parentIndex = i.parentIndex;
			Edge newEdge = new Edge(i.attrIndex, parentIndex, trainingData);
			newEdge.setHashMap(parentIndex,i.attrIndex);
			String test_parentVal = testData.stringValue(testData.attribute(parentIndex));
			String test_val = testData.stringValue(testData.attribute(i.attrIndex));
			pos_label1 = pos_label1 *  newEdge.attr2_prob_given_attr1andLabel(parentIndex, test_parentVal, i.attrIndex, test_val, label1);
			pos_label2 = pos_label2 *  newEdge.attr2_prob_given_attr1andLabel(parentIndex, test_parentVal, i.attrIndex, test_val, label2);		
			
		}
			
		if (pos_label1 >= pos_label2){
			cr.pre_label = label1;
			cr.prob_label = pos_label1/(pos_label1 + pos_label2);
		} else {
			cr.pre_label = label2;
			cr.prob_label = pos_label2/(pos_label1 + pos_label2);
		}
		
		return cr;
	}		
	
}
