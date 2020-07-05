import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Example {

	public static void main(String[] args) throws IOException, InterruptedException {
		var file = new File("C:\\Users\\luiza\\Documents\\My Projects\\wealthymachinedata\\PETR4F\\datafile");
		List<double[]> rows = getDataFromJson(file);
		INDArray features = createFeatures(rows);
		INDArray labels = createLabels(rows);
		DataSet dataset = new DataSet(features, labels);
		NormalizerMinMaxScaler normalizer = createNormalizer(dataset);
		normalizer.preProcess(dataset);

		MultiLayerConfiguration conf = new NeuralNetConfiguration
				.Builder()
				.seed(123)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.weightInit(WeightInit.XAVIER)
				.l2(1e-4)
				.list()
				.layer(new LSTM.Builder()
						.activation(Activation.TANH)
						.nIn(5)
						.nOut(256)
						.activation(Activation.TANH)
						.gateActivationFunction(Activation.HARDSIGMOID)
						.dropOut(0.2)
						.build())
				.layer(new DenseLayer.Builder()
						.nIn(256)
						.nOut(30)
						.activation(Activation.RELU)
						.build())
				.layer(new RnnOutputLayer.Builder()
						.nIn(30)
						.nOut(1)
						.activation(Activation.IDENTITY)
						.lossFunction(LossFunctions.LossFunction.MSE)
						.build())
				.backpropType(BackpropType.TruncatedBPTT)
				.build();
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();

		int trainLength = (int) features.size(0) - 5;
		IteratorDataSetIterator trainDataset = new IteratorDataSetIterator(dataset.getRange(0, trainLength).iterator(), 12);
		int numEpoch = 5;
		for (int i = 0; i < numEpoch; i++) {
			model.fit(trainDataset);
		}

		var testData = dataset.getRange(trainLength, trainLength + 5).iterator();
		while (testData.hasNext()) {
			var batch = testData.next();
			var output = model.output(batch.getFeatures());
			System.out.println(batch.getLabels() + " - " + output);;
		}
	}

	private static NormalizerMinMaxScaler createNormalizer(DataSet dataSet) {
		NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
		normalizer.fitLabel(true);
		normalizer.fit(dataSet);
		return normalizer;
	}

	private static INDArray createLabels(List<double[]> rows) {
		double[][][] labels = new double[rows.size() - 4][1][1];
		for (int i = 0; i < labels.length - 1; i++) {
			labels[i][0][0] = rows.get(i + 5)[3];
		}
		return Nd4j.create(labels);
	}

	private static INDArray createFeatures(List<double[]> rows) {
		double[][][] features = new double[rows.size() - 4][][];
		for (int i = 0; i < features.length; i++) {
			double[][] rows5Days = new double[5][4];
			for (int j = 0; j < 5; j++) {
				rows5Days[j] = rows.get(i + j);
			}
			features[i] = rows5Days;
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
