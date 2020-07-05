import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class SeaTemperatureConvolutionalLSTM {
	public static void main(String[] args) throws IOException, InterruptedException {
		var sea_temp = new ClassPathResource("sea_temp").getFile();
		var featureBaseDir = new File(sea_temp, "features"); // set feature directory
		var targetsBaseDir = new File(sea_temp, "targets"); // set label directory


		var numSkipLines = 1;
		var regression = true;
		var batchSize = 32;

		var trainFeatures = new CSVSequenceRecordReader(numSkipLines, ",");
		trainFeatures.initialize( new NumberedFileInputSplit(featureBaseDir + "/%d.csv", 1, 1736));
		var trainTargets = new CSVSequenceRecordReader(numSkipLines, ",");
		trainTargets.initialize(new NumberedFileInputSplit(targetsBaseDir + "/%d.csv", 1, 1736));

		var train = new SequenceRecordReaderDataSetIterator(trainFeatures, trainTargets, batchSize,
				10, regression, SequenceRecordReaderDataSetIterator.AlignmentMode.EQUAL_LENGTH);


		var testFeatures = new CSVSequenceRecordReader(numSkipLines, ",");
		testFeatures.initialize( new NumberedFileInputSplit(featureBaseDir + "/%d.csv", 1937, 2089));
		var testTargets = new CSVSequenceRecordReader(numSkipLines, ",");
		testTargets.initialize(new NumberedFileInputSplit(targetsBaseDir + "/%d.csv", 1937, 2089));

		var test = new SequenceRecordReaderDataSetIterator(testFeatures, testTargets, batchSize,
				10, regression, SequenceRecordReaderDataSetIterator.AlignmentMode.EQUAL_LENGTH);

		var V_HEIGHT = 13;
		var V_WIDTH = 4;
		var kernelSize = 2;
		var numChannels = 1;

		var conf = new NeuralNetConfiguration.Builder()
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.seed(12345)
				.weightInit(WeightInit.XAVIER)
				.updater(new AdaGrad(0.005))
				.list()
				.layer(0, new ConvolutionLayer.Builder(kernelSize, kernelSize)
						.nIn(1) //1 channel
						.nOut(7)
						.stride(2, 2)
						.activation(Activation.RELU)
						.build())
				.layer(1, new LSTM.Builder()
						.activation(Activation.SOFTSIGN)
						.nIn(84)
						.nOut(200)
						.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
						.gradientNormalizationThreshold(10)
						.build())
				.layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
						.activation(Activation.IDENTITY)
						.nIn(200)
						.nOut(52)
						.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
						.gradientNormalizationThreshold(10)
						.build())
				.inputPreProcessor(0, new RnnToCnnPreProcessor(V_HEIGHT, V_WIDTH, numChannels))
				.inputPreProcessor(1, new CnnToRnnPreProcessor(6, 2, 7 ))
				.build();

		var net = new MultiLayerNetwork(conf);
		net.init();

		net.fit(train , 25);

		var eval = net.evaluateRegression(test);

		test.reset();
		System.out.println(eval.stats());

	}

}
