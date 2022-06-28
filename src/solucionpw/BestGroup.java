package solucionpw;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;

import weka.core.Instance;
import weka.core.Instances;

public class BestGroup {
	public List<WekaDataSet> AcceptedGroups = new ArrayList<WekaDataSet>();
	
	public void add(WekaDataSet theWekaDatasSet){
		AcceptedGroups.add(theWekaDatasSet);
	}
	
	public void DefineGroup(Instance theInstance){
		int bestGroup = 0;
		double bestCC = 0.0;
		for (int i = 0; i < AcceptedGroups.size(); i++){
			double value = (AcceptedGroups.get(i)).EvaluateIncludeInstance(theInstance);
			if (i ==0) {
				bestGroup = i;
				bestCC = value;
			}
			if (value > bestCC){
				bestGroup = i;
				bestCC = value;				
			}
		}
		(AcceptedGroups.get(bestGroup)).IncludeInstance(theInstance);
	}

	public String TablaResultados;
	
	public double[] Evaluate(Instances TestInstances){
		int numInstances = TestInstances.numInstances();
		int Inside15 = 0;
		double MSE = 0.0;
		double MAE = 0.0;
		double psimax = Double.MIN_VALUE;
		double psimin = Double.MAX_VALUE;
		TablaResultados = "";
		for (int ins = 0; ins < numInstances; ins++){
			Instance theInstance = TestInstances.get(ins);
			
			int bestGroup = 0;
			double minDistance = Double.MAX_VALUE;
			for (int i = 0; i < AcceptedGroups.size(); i++){
				double value = (AcceptedGroups.get(i)).MinDistanceToTestInstance(theInstance);
				if (value < minDistance){
					bestGroup = i;
					minDistance = value;				
				}
			}			
			double psi = theInstance.value(TestInstances.numAttributes()-1);
			if (psi > psimax){psimax = psi;}
			if (psi < psimin){psimin = psi;}
			double predicted_psi = (AcceptedGroups.get(bestGroup)).PredictPSI(theInstance);
			
			TablaResultados += Interfaz.Round(psi, 6) + "|" + Interfaz.Round(predicted_psi, 6) + "\n";
			if (predicted_psi > psi*0.85 && predicted_psi < psi*1.15){
				Inside15++;
			}
			MSE += Math.pow(psi-predicted_psi, 2);
			MAE += Math.abs(psi-predicted_psi);
		}
		double inside = (100.0*Inside15)/numInstances;
		MSE = MSE / numInstances;
		MAE = MAE / numInstances;
		double RMSE = Math.sqrt(MSE);
		double NRMSE = RMSE/ (psimax-psimin);
		double[] result = new double[5];
		result[0] = inside;
		result[1] = MSE;
		result[2] = MAE;
		result[3] = RMSE;
		result[4] = NRMSE;
		return result;
	}

	public String GroupsReport(){
		String result = "";
		for (int i = 0; i < AcceptedGroups.size(); i++){
			WekaDataSet theGroup = AcceptedGroups.get(i);
			result += "Group 1 data \n";
			result += theGroup.DataSet.toString() + "\n";
			result += "Group 1 linear regression model: \n";
			result += theGroup.LinealModel.LinearRegresionModel.toString() + "\n";
		}
		return result;
	}
	
	public String Report(boolean all){
		try{
			double avgCC = 0.0;
			int sumRows = 0;
			String rowsByGroup = "";
			for (int i = 0; i < AcceptedGroups.size(); i++){
				double CC = (AcceptedGroups.get(i)).LinealModel.TestOnTraining.correlationCoefficient();
				int rowsGroup = (AcceptedGroups.get(i)).DataSet.numInstances();
				avgCC += CC * rowsGroup;
				sumRows += rowsGroup;
				rowsByGroup = rowsByGroup.concat(Integer.toString(rowsGroup));
				rowsByGroup = rowsByGroup.concat("(");
				rowsByGroup = rowsByGroup.concat(Round(CC, 3));
				rowsByGroup = rowsByGroup.concat("):");
			}
			
			String result = Integer.toString(AcceptedGroups.size());
			result = result.concat(" | ");
			result = result.concat(Round(avgCC/sumRows, 3));
			result = result.concat(" | ");
			result = result.concat(Integer.toString(sumRows));
			if (all){
				result = result.concat(" | ");
				result = result.concat(rowsByGroup);
			}			
			return result;
		}
		catch (Exception e){
			throw new RuntimeException(e);
		}
	}
        public static String Round (double num, int places){
		BigDecimal bd = new BigDecimal(num);
		bd = bd.setScale(places, RoundingMode.HALF_UP);	// 3 decimals
		return Double.toString(bd.doubleValue());
	}
}