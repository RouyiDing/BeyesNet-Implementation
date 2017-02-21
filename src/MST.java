import java.util.ArrayList;
import weka.core.Instances;


/**
 * Use Prims's algorithm to find a maximal spanning tree. The start root is 
 * attribute 0
 * 
 * @author Rouyi Ding
 *
 */
public class MST{

	/**visited attributes*/
	ArrayList<Integer> visit = new ArrayList<Integer>();
	
	/**nodes with parent information*/
	ArrayList<Node> nodes = new ArrayList<Node>();
	
	/**2-D array used to record weight between attribute pairs */
	double[][] wgtMatrix;
	

	/**
	 * Use Prims's algorithm to find a maximal spanning tree.
	 * 
	 * @param trainingData
	 * 
	 * @return a list of nodes with parent attribute index information.
	 */
	public ArrayList<Node> getMST(Instances trainingData){
		/**number of total attributes except class attributes */
		int numAttributes = trainingData.numAttributes() - 1;
		wgtMatrix = new double[numAttributes][numAttributes];
		
		// initialize lower triangular with value -1.0. Then calculate the
		// weight of all attribute pairs and store weight in matrix.
		for (int row = 0; row < numAttributes; row++){
			for (int col = 0; col < numAttributes; col++){
				if (row == col){
					wgtMatrix[row][col] = -1.0;
				} else {
					Edge edge = new Edge(row, col, trainingData);
					edge.setWeight(edge);
					wgtMatrix[row][col] = edge.weight;
				} 
			}
		}
			
		//Initialize nodes list and visit list by adding all attributes to nodes
		//and adding first attribute to visited list
		for (int i = 0; i < numAttributes; i++){
			nodes.add(new Node(i));
		}
		visit.add(0);
		for (int row = 0; row < numAttributes; row ++){
			wgtMatrix[row][0] = -1;
		}
		
		// find MST by prime algorithm
		while (visit.size() < numAttributes){
			double maxWgt = 0.0;
			int nextAttrIndex = 0;
			int parentIndex = 0;
			for (int i:visit){
				for (int col = 0; col < numAttributes; col++){
					if (wgtMatrix[i][col] > maxWgt){
						maxWgt = wgtMatrix[i][col];
						parentIndex = i;
						nextAttrIndex = col;
					}
				}
			}
			for (int row = 0; row < numAttributes; row++){
				wgtMatrix[row][nextAttrIndex] = -1;
			}

			wgtMatrix[nextAttrIndex][parentIndex] = -1;
			visit.add(nextAttrIndex); // record the nodes we already visited
			nodes.get(nextAttrIndex).setParent(parentIndex); 
		}		
		return nodes;
	}	
	
}
