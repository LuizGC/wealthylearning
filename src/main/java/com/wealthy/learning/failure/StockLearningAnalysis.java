package com.wealthy.learning.failure;

import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class StockLearningAnalysis {

	public static void main(String[] args) throws IOException {
		var file = new File("C:\\Users\\luiza\\Documents\\My Projects\\wealthylearning\\src\\main\\resources\\PETR4F.txt");
		var creator = new TechnicalAnalysisToDataSetCreator(file);
		var datasetBuilder = creator.createLSTMFeatureLabelDatasetBuilder(20);
		var dataset = datasetBuilder.create();

		MultiLayerConfiguration conf = new NeuralNetConfiguration
				.Builder()
				.updater(new Adam())
				.weightInit(WeightInit.XAVIER)
				.seed(123)
				.list()
				.layer(new LSTM.Builder()
						.activation(Activation.TANH)
						.nIn(20)
						.nOut(5)
						.build())
				.layer(new RnnOutputLayer.Builder()
						.nIn(5)
						.nOut(1)
						.activation(Activation.SIGMOID)
						.lossFunction(LossFunctions.LossFunction.MSE)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();

		var wrapper = new ParallelWrapper.Builder<MultiLayerNetwork>(model)
				.prefetchBuffer(60)
				.workers(60)
				.averagingFrequency(100)
				.build();

		int trainLength = datasetBuilder.getFeatures().length - 5;
		int numEpoch = 200;
		for (int i = 0; i < numEpoch; i++) {
			IteratorDataSetIterator trainDataset = new IteratorDataSetIterator(dataset.getRange(0, trainLength).iterator(), 3);
			wrapper.fit(trainDataset);
		}

		var testData = dataset.getRange(trainLength, trainLength + 5).iterator();
		while (testData.hasNext()) {
			var batch = testData.next();
			var output = model.output(batch.getFeatures());
			System.out.println(output + " - " + batch.getLabels());
		}

		var eval = model.evaluateRegression(new IteratorDataSetIterator(dataset.getRange(trainLength, trainLength + 4).iterator(), 1));

		System.out.println( eval.stats() );
	}

}
