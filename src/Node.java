/**
 * A representation of attribute
 * 
 * @author Rouyi Ding
 *
 */
public class Node {
	
	int attrIndex;
	int parentIndex;
	
	Node(int attrIndex){
		this.attrIndex = attrIndex;
		this.parentIndex = -1;
	}
	
	
	/**
	 * set parent index of the current attribute
	 * @param parentIndex
	 */
	public void setParent(int parentIndex){
		this.parentIndex = parentIndex;
	}
	
	

}
