import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import weka.core.Instances;


/**
 * Generate random samples of size n from training data with replacement.
 *  
 * @author Rouyi Ding
 *
 */
public class RandSampling {
	
	/**
	 * Generate random samples of size n from training data with replacement.
	 * 
	 * @param trainingData
	 * @param n
	 *         sample size
	 * @return  
	 *        sample instances of size n
	 */
	public static Instances getSampling(Instances trainingData, int n){
		
		/**sampleIndex stores the Indexes of n samples*/
		List<Integer> list = new ArrayList<Integer>();
		Random random = new Random();	
		for (int i = 0; i < trainingData.numInstances(); i++){
			list.add(i);
		}
		
		int index = list.get(random.nextInt(list.size()));
		Instances newTrainingData = new Instances(trainingData, index, 1);
		
		for (int i = 1; i < n; i++){
				index = list.get(random.nextInt(list.size()));
				newTrainingData.add(trainingData.get(index));
			}
		return newTrainingData;		
	}
}


	
