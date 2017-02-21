import java.io.IOException;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;




public class LearningCurve extends ApplicationFrame {

	public  LearningCurve( String applicationTitle , String chartTitle ) {
		super(applicationTitle);

	}
	
	public void draw(double[] accuRate, int[] sampleSize, String name) {

		DefaultCategoryDataset dataset = new DefaultCategoryDataset( );
		for (int i = 0; i < accuRate.length; i++){
			dataset.addValue(accuRate[i], "Classify Accuracy",Integer.toString(sampleSize[i]));
		}
	      
	      JFreeChart lineChart = ChartFactory.createLineChart(
	         name,
	         "Training Data Size","Accuracy",
	         dataset,
	         PlotOrientation.VERTICAL,
	         true,true,false);
	         
	      ChartPanel chartPanel = new ChartPanel( lineChart );
	      chartPanel.setPreferredSize( new java.awt.Dimension( 560 , 367 ) );
	      chartPanel.setMinimumDrawHeight(1);
	
	      setContentPane( chartPanel );
		 
	      this.pack( );
	      RefineryUtilities.centerFrameOnScreen(this);
	      this.setVisible( true );
	}
	
	

}
