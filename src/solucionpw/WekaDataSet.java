package solucionpw;

import weka.core.Instance;
import weka.core.Instances;

public class WekaDataSet {

	public Instances DataSet;
	public LinearRegressionModel LinealModel;
	
	public WekaDataSet(Instances origin){
		DataSet = new Instances(origin);
	}
	
	public void PreserveFirstRows (int numberOfRows){
		for (int i = DataSet.numInstances() - 1; i >= 0; i--) {
		    if (i >= numberOfRows) {
		    	DataSet.delete(i);
		    }
		}
	}

	public void RemoveFirstRows (int numberOfRows){
		for (int i = 0; i < DataSet.numInstances() - 1; i++) {
		    if (i < numberOfRows) {
		    	DataSet.delete(0);
		    }
		}
	}
	
	public void DefineLinearModel()
	{
		LinealModel = new LinearRegressionModel(DataSet);
	}
	
	public double EvaluateIncludeInstance(Instance theInstance){
		try{
			Instances copyData = new Instances(DataSet);
			double currentCC = LinealModel.TestOnTraining.correlationCoefficient();
			copyData.add(theInstance);
			LinearRegressionModel theLinealRegressionModel = new LinearRegressionModel(copyData);
			double newCC = theLinealRegressionModel.TestOnTraining.correlationCoefficient();
			return newCC - currentCC;
		}
		catch (Exception e){
			throw new RuntimeException(e);
		}
	}
	
	public void IncludeInstance(Instance theInstance){
		DataSet.add(theInstance);
		LinealModel = new LinearRegressionModel(DataSet);
	}
	
	public double distance(Instance instance1, Instance instance2) {
	    double dist = 0.0;

	    for (int i = 0; i < instance1.numAttributes()-1; i++) {
	        double x = instance1.value(i);
	        double y = instance2.value(i);
	        if (Double.isNaN(x) || Double.isNaN(y)) {
	            continue; // Mark missing attributes ('?') as NaN.
	        }
	        dist += (x-y)*(x-y);
	    }
	    return Math.sqrt(dist);
	}
	
	public double MinDistanceToTestInstance(Instance theInstance){
		try{
			double minDistance = Double.MAX_VALUE;
			int numInstances = DataSet.numInstances();
			for (int i = 0; i < numInstances; i++){
				Instance rowInstance = DataSet.get(i);
				double dis = distance(theInstance, rowInstance);
				if (dis < minDistance){
					minDistance  = dis;
				}
			}
			return minDistance;
		}
		catch (Exception e){
			throw new RuntimeException(e);
		}
	}
	
	public double PredictPSI(Instance theInstance){
		try {
			double[] predictedPSI = LinealModel.LinearRegresionModel.distributionForInstance(theInstance);
			return predictedPSI[0];
		}
		catch (Exception e){
			throw new RuntimeException(e);
		}
	}
}