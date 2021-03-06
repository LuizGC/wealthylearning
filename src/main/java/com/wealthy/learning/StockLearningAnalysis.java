package com.wealthy.learning;

import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class StockLearningAnalysis {

	public static void main(String[] args) throws IOException {
		MultiLayerNetwork model = createModel();
		var petrFolder = new File("C:\\Users\\luiza\\Downloads\\petr");
		boolean firstEpoch = true;
		for(var file : petrFolder.listFiles()) {
			var creator = new TechnicalAnalysisToDataSetCreator(file);
			var datasetBuilder = creator.createLSTMFeatureLabelDatasetBuilder(20);
			var dataset = datasetBuilder.createTrainDataSet(146);

			if (firstEpoch) {
				firstEpoch = false;
			} else {
				var eval = model.evaluateRegression(new IteratorDataSetIterator(dataset.iterator(), 1));
				var fr = new FileWriter(new File("append.txt"), true);
				fr.write(file + System.lineSeparator() + eval.rSquared(0) + System.lineSeparator() + " ------" + System.lineSeparator());
				fr.close();
			}

			int numEpoch = 1;
			for (int i = 0; i < numEpoch; i++) {
				model.fit(new IteratorDataSetIterator(dataset.iterator(), 1));
			}

//			var testData = dataset.getRange(trainLength, trainLength + 1).iterator();
//			while (testData.hasNext()) {
//				var batch = testData.next();
//				var output = model.output(batch.getFeatures());
//				var fr = new FileWriter(new File("append.txt"), true);
//				fr.write(file + System.lineSeparator() + output + System.lineSeparator() + " ------");
//				fr.close();
//			}
		}

	}

	private static MultiLayerNetwork createModel() {
		MultiLayerConfiguration conf = new NeuralNetConfiguration
				.Builder()
				.updater(new Adam())
				.weightInit(WeightInit.XAVIER)
				.seed(123)
				.list()
				.layer(new LSTM.Builder()
						.activation(Activation.TANH)
						.nIn(292)
						.nOut(111)
						.build())
				.layer(new LSTM.Builder()
						.activation(Activation.TANH)
						.nIn(111)
						.nOut(41)
						.build())
				.layer(new LSTM.Builder()
						.activation(Activation.TANH)
						.nIn(41)
						.nOut(23)
						.build())
				.layer(new LSTM.Builder()
						.activation(Activation.TANH)
						.nIn(23)
						.nOut(17)
						.build())
				.layer(new LSTM.Builder()
						.activation(Activation.TANH)
						.nIn(17)
						.nOut(7)
						.build())
				.layer(new RnnOutputLayer.Builder()
						.nIn(7)
						.nOut(1)
						.activation(Activation.SIGMOID)
						.lossFunction(LossFunctions.LossFunction.MSE)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		model.addListeners(new ScoreIterationListener(200));
		return model;
	}

}
