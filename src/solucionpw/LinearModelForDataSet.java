package solucionpw;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Remove;

public class LinearModelForDataSet {
	public Instances DataSet;
	public LinearRegressionModel LinealModel;
	
	public int BestRows;

	public LinearModelForDataSet(Instances trainingData, double MaximunAbsoluteErrorPerRow){
		try{			
		    // Define the linear regression model for training data set
			LinealModel = new LinearRegressionModel(trainingData);		
	        
	        // Calculate Absolute Error for each instance on Training Data Set
	        double[] absolute_errors = new double[trainingData.numInstances()];
	        for (int i = 0; i < trainingData.numInstances(); i++){
	        	Instance instance  = trainingData.get(i);
	        	double[] result = LinealModel.LinearRegresionModel.distributionForInstance(instance);
	        	double predicted_psi = result[0];
	        	double psi = instance.value((instance.classIndex()));
	        	absolute_errors[i] = Math.abs(predicted_psi - psi);
	        }
	        
	        // Include the absolute error in data set and sort by this new attribute
	        Add filter = new Add();
	        filter.setAttributeIndex("last");
	        filter.setAttributeName("absolue_error");
	        filter.setInputFormat(trainingData);
	        trainingData = Filter.useFilter(trainingData, filter);
	        for (int i = 0; i < trainingData.numInstances(); i++){
	        	Instance instance  = trainingData.get(i);
	        	instance.setValue(trainingData.numAttributes()-1, absolute_errors[i]);
	        }
	        trainingData.sort(trainingData.attribute(trainingData.numAttributes()-1));
	        
	        // Define first rows with good accumulate performance SAE/i < MaximunAbsoluteErrorPerRow
	        // Calculate Absolute Error for each instance
	        double SAE = 0.0;
	        for (int i = 0; i < trainingData.numInstances(); i++){
	        	Instance instance  = trainingData.get(i);
	        	double absolute_error = instance.value((trainingData.numAttributes()-1));
	        	SAE += absolute_error;
	        	if (SAE/i < MaximunAbsoluteErrorPerRow){
	        		BestRows = i;
	        	}
	        }
	        
	        //Remove the absolute error attribute
	        Remove removeFilter = new Remove();
	        removeFilter.setAttributeIndices("Last");
	        removeFilter.setInvertSelection(false);
	        removeFilter.setInputFormat(trainingData);
	        trainingData = Filter.useFilter(trainingData, removeFilter);
	        
	        DataSet = new Instances(trainingData);
		}
	    catch (final Exception e)
	    {
	    	throw new RuntimeException(e);
	    }
	}
}