package weka.api;

import weka.core.Instances;
import weka.core.Instance;
import weka.core.converters.*;
import java.io.File;
import weka.classifiers.bayes.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.evaluation.*;

public class classifyInstance {
	
	public static void main(String[] args) throws Exception{
		
		CSVLoader loader = new CSVLoader(); 
		loader.setSource(new File("iris.csv"));
		Instances data = loader.getDataSet();
		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data); 
		saver.setFile(new File("iris.arff"));
		saver.writeBatch();
		
		DataSource source = new DataSource("iris.arff");
		Instances dataset = source.getDataSet();
		
		int trainSize = (int)Math.round(dataset.numInstances() * 0.8);
		int testSize = dataset.numInstances() - trainSize;
		
		Instances trainingData = new Instances(dataset, 0, trainSize);
		trainingData.setClassIndex(trainingData.numAttributes()-1);
		
		Instances testData = new Instances(dataset, trainSize, testSize);
		testData.setClassIndex(testData.numAttributes()-1);
				
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(trainingData);
		
		
		for(int i = 0; i < testData.numInstances(); i++) {
			
			double actualClass = testData.instance(i).classValue();
			String actual = testData.classAttribute().value((int)actualClass);
			
			Instance newInst = testData.instance(i);
			
			double predNB = nb.classifyInstance(newInst); 
			String predString = testData.classAttribute().value((int) predNB);
			
			System.out.println(actual + ", " + predString);
			}
		
		
		Evaluation eval = new Evaluation(trainingData);
		eval.evaluateModel(nb, testData);
		System.out.println(eval.toSummaryString("\nEvaluation results:\n", false));
		System.out.println(eval.toMatrixString("=== Overall Confusion Matrix ===\n"));
		
		
		
		
		
		}
}
