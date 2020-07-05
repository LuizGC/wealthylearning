package com.wealthy.learning.failure;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

public class StockLearningAnalysis {

	public static void main(String[] args) throws IOException {
		var file = new File("C:\\Users\\luiza\\Documents\\My Projects\\wealthymachinedata\\PETR4_ANALYSIS\\datafile");
		var creator = new TechnicalAnalysisToDataSetCreator(file, 146, 219);
		var featureArray = creator.getFeatureArray();
		var labelArray = creator.getLabelArray();
		var feature = Nd4j.create(featureArray);
		var label = Nd4j.create(labelArray);

		var dataset = new DataSet(feature, label);

		System.out.println(dataset);
		//machineLearning(featureArray, labelArray[0], dataset);

	}

//	private static void machineLearning(double[][] featureArray, double[] doubles, DataSet dataset) {
//		var conf = new NeuralNetConfiguration.Builder()
//				.seed(123)
//				.updater(new Adam())
//				.list()
//				.layer(new DenseLayer.Builder()
//						.nIn(featureArray[0].length)
//						.nOut(200)
//						.activation(Activation.SIGMOID)
//						.weightInit(WeightInit.XAVIER)
//						.build())
//				.layer(new LSTM.Builder()
//						.nIn(200) // Number of input datapoints.
//						.nOut(25) // Number of output datapoints.
//						.activation(Activation.SIGMOID) // Activation function.
//						.weightInit(WeightInit.XAVIER) // Weight initialization.
//						.build())
//				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
//						.nIn(25)
//						.nOut(doubles.length)
//						.activation(Activation.SIGMOID)
//						.weightInit(WeightInit.XAVIER)
//						.build())
//				.build();
//
//		var model = new MultiLayerNetwork(conf);
//		model.init();
//
////		var wrapper = new ParallelWrapper.Builder(model)
////				.prefetchBuffer(60)
////				.workers(60)
////				.averagingFrequency(100)
////				.build();
//
//		var numEpochs = 15;
//		var train = new IteratorDataSetIterator(dataset.getRange(0, featureArray.length - 5).iterator(), 1);
//		for (int i = 0; i < numEpochs; i++) {
//			model.fit(train);
//		}
//
//		var testData = dataset.getRange(featureArray.length - 5, featureArray.length).iterator();
//		while (testData.hasNext()) {
//			var batch = testData.next();
//			var output = model.output(batch.getFeatures());
//			System.out.println(batch.getLabels() + " - " + output);;
//		}
//	}

}
