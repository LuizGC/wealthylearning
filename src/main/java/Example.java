import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Example {

	public static void main(String[] args) throws IOException, InterruptedException {
		var file = new File("C:\\Users\\luiza\\Documents\\My Projects\\wealthymachinedata\\PETR4F\\PETR4F.txt");
		List<double[]> rows = getDataFromJson(file);
		INDArray features = createFeatures(rows);
		INDArray labels = createLabels(rows);
		DataSet dataset = new DataSet(features, labels);
		NormalizerMinMaxScaler normalizer = createNormalizer(dataset);
		normalizer.preProcess(dataset);

		MultiLayerConfiguration conf = new NeuralNetConfiguration
				.Builder()
				.updater(new Adam())
				.weightInit(WeightInit.XAVIER)
				.seed(123)
				.list()
				.layer(new LSTM.Builder()
						.activation(Activation.TANH)
						.nIn(22)
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

		int trainLength = (int) features.size(0) - 5;
		int numEpoch = 200;
		for (int i = 0; i < numEpoch; i++) {
			IteratorDataSetIterator trainDataset = new IteratorDataSetIterator(dataset.getRange(0, trainLength).iterator(), 10);
			wrapper.fit(trainDataset);
		}

		var testData = dataset.getRange(trainLength, trainLength + 4).iterator();
		while (testData.hasNext()) {
			var batch = testData.next();
			var output = model.output(batch.getFeatures());
			normalizer.revertLabels(output);
			normalizer.revertLabels(batch.getLabels());
			System.out.println(output + " - " + batch.getLabels());
		}

		var eval = model.evaluateRegression(new IteratorDataSetIterator(dataset.getRange(trainLength, trainLength + 4).iterator(), 1));

		System.out.println( eval.stats() );
	}

	private static NormalizerMinMaxScaler createNormalizer(DataSet dataSet) {
		NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
		normalizer.fitLabel(true);
		normalizer.fit(dataSet);
		return normalizer;
	}

	private static INDArray createLabels(List<double[]> rows) {
		double[][][] labels = new double[rows.size() - 21][1][4];
		for (int i = 0; i < labels.length - 1; i++) {
			var future = rows.get(i + 22);
			labels[i][0][0] = future[0];
			labels[i][0][1] = future[1];
			labels[i][0][2] = future[2];
			labels[i][0][3] = future[3];
		}
		return Nd4j.create(labels);
	}

	private static INDArray createFeatures(List<double[]> rows) {
		double[][][] features = new double[rows.size() - 21][][];
		for (int i = 0; i < features.length; i++) {
			double[][] rowsDays = new double[22][4];
			for (int j = 0; j < 22; j++) {
				rowsDays[j] = rows.get(i + j);
			}
			features[i] = rowsDays;
		}
		return Nd4j.create(features);
	}

	private static List<double[]> getDataFromJson(File file) throws IOException {
		JsonNode jsonTree = new ObjectMapper().readTree(file);
		ArrayList<double[]> rows = new ArrayList<>();
		for (JsonNode arrayItem : jsonTree) {
			double[] cols = new double[4];
			cols[0] = arrayItem.get("openPrice").asDouble();
			cols[1] = arrayItem.get("closePrice").asDouble();
			cols[2] = arrayItem.get("lowPrice").asDouble();
			cols[3] = arrayItem.get("highPrice").asDouble();
			rows.add(cols);
		}
		return rows;
	}
}
