package solucionpw;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;

public class LinearRegressionModel {

	public LinearRegression LinearRegresionModel;
	public Evaluation TestOnTraining;

	public LinearRegressionModel(Instances DataSet){
		try{			
		        
	        LinearRegresionModel = new LinearRegression();
	        LinearRegresionModel.setOptions(weka.core.Utils.splitOptions("-S 0 -R 1.0E-8 -num-decimal-places 4"));
	        LinearRegresionModel.buildClassifier(DataSet);            
	        
	        TestOnTraining = new Evaluation(DataSet);
	        TestOnTraining.evaluateModel(LinearRegresionModel, DataSet);
	        
	        //System.out.println(eTest.toSummaryString());
	        //System.out.println(eTest.correlationCoefficient());
		}
	    catch (final Exception e)
	    {
	    	throw new RuntimeException(e);
	    }
	}
}