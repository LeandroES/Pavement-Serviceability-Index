package solucionpw;

import java.io.File;
import java.io.FileWriter;
import java.math.BigDecimal;
import java.math.RoundingMode;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class Interfaz {

	public static void main(String[] args) throws Exception{
            
		String DataDirectory = "datos/";

		try{

			int MinNumberOfInstancesByGroup = 800;

			ConverterUtils.DataSource sourceTest = new ConverterUtils.DataSource(DataDirectory.concat("data_test.arff"));
			Instances TestData = sourceTest.getDataSet();
	        if (TestData.classIndex() == -1)
	        	TestData.setClassIndex(TestData.numAttributes() - 1); // psi is the last column

	        int afinamiento = 1;
			for (double maepr = 0.03; maepr <0.04; maepr+=0.005){
				
				for (double qD = 0.03; qD <0.07; qD+=0.005){
					long inicio = System.currentTimeMillis();
					double MaximunAbsoluteErrorPerRow = maepr;
			        double qualityDegrade = qD;
			        					
					// Read the data file and define the index for class attribute
					ConverterUtils.DataSource source = new ConverterUtils.DataSource(DataDirectory.concat("data_training.arff"));
					Instances trainingData = source.getDataSet();
			        if (trainingData.classIndex() == -1)
			            trainingData.setClassIndex(trainingData.numAttributes() - 1); // psi is the last column
				
			        BestGroup bg = new BestGroup();
			        while (true){
						
						LinearModelForDataSet lmd = new LinearModelForDataSet(trainingData, MaximunAbsoluteErrorPerRow);
						//System.out.println(lmd.BestRows);
						
						if (lmd.BestRows < MinNumberOfInstancesByGroup) {
							if (lmd.DataSet.numInstances() < MinNumberOfInstancesByGroup){
								lmd.BestRows = lmd.DataSet.numInstances();
							}
							else{
								lmd.BestRows = MinNumberOfInstancesByGroup;
							}
						}
						
						WekaDataSet bestRowsInDataSet = new WekaDataSet(lmd.DataSet);
						bestRowsInDataSet.PreserveFirstRows(lmd.BestRows);
						bestRowsInDataSet.DefineLinearModel();
						
						if (bestRowsInDataSet.LinealModel.TestOnTraining.correlationCoefficient() > 0.88){
							bg.add(bestRowsInDataSet);
						}
						else{
							break;
						}
						
						WekaDataSet worstRowsInDataSet = new WekaDataSet(lmd.DataSet);
						worstRowsInDataSet.RemoveFirstRows(lmd.BestRows);
			
						trainingData = new Instances(worstRowsInDataSet.DataSet);
						MaximunAbsoluteErrorPerRow += qualityDegrade;
					}
			        
			        String resItermedio = bg.Report(false);
			        
			        int numInstances = trainingData.numInstances();
					for (int i = 0; i < numInstances; i++ ){
						Instance theInstance = trainingData.get(i);
						bg.DefineGroup(theInstance);
						//if (i==500) break;
					}
					
					long fin = System.currentTimeMillis();
					double tiempo = (double) ((fin - inicio)/1000);
					
					System.out.print(afinamiento);
					System.out.print(" | ");
					System.out.print(Round(tiempo, 3));
					System.out.print(" | ");
					System.out.print(Round(maepr, 3));
					System.out.print(" | ");
					System.out.print(Round(qD, 3));
					System.out.print(" | ");
					
					double results[] = bg.Evaluate(TestData);
					System.out.print(Round(results[0], 6)); // inside15
					System.out.print(" | ");
					System.out.print(Round(results[1], 6)); // MSE
					System.out.print(" | ");
					System.out.print(Round(results[2], 6)); // MAE
					System.out.print(" | ");
					System.out.print(Round(results[3], 6)); // RMSE
					System.out.print(" | ");
					System.out.print(Round(results[4], 6)); // NRMSE
					System.out.print(" | ");
					System.out.print(resItermedio);
					System.out.print(" | ");
					System.out.println(bg.Report(true));
					
					// Save models for each found solution ... directory tuning in datos
					String pathDir = DataDirectory + "/tuning/" + Round(maepr, 6) + "-" + Round(qD, 6);
					File directory = new File(pathDir);
					directory.mkdir();					
					File file = new File(pathDir + "/Resultados.txt");
					FileWriter fr = new FileWriter(file);
		            fr.write(bg.TablaResultados);
		            fr.write(bg.GroupsReport());
		            fr.close();

					afinamiento++;
				}
			}
			System.out.println("Fin");
		}
        catch (Exception e)
        {
        	throw new RuntimeException(e);
        }
	}
	
	public static String Round (double num, int places){
		BigDecimal bd = new BigDecimal(num);
		bd = bd.setScale(places, RoundingMode.HALF_UP);	// 3 decimals
		return Double.toString(bd.doubleValue());
	}
}