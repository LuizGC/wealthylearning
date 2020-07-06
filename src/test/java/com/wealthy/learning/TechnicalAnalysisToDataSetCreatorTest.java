package com.wealthy.learning;

import com.wealthy.learning.failure.DatasetBuilder;
import com.wealthy.learning.failure.TechnicalAnalysisToDataSetCreator;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class TechnicalAnalysisToDataSetCreatorTest {

	double[][] featuresExpected = new double[][] {
			new double[]{0.4, 1, 0},
			new double[]{0, 1, 0},
			new double[]{0.8, 0, 1},
			new double[]{1.0, 0.0, 0.2857142857142857},
			new double[]{0.0, 0.4, 1.0}
	};

	@Test
	void getDataset_HappyPath_ShouldReturnCorrectFeatureSet() throws IOException {
		var file = createFile();
		var creator = new TechnicalAnalysisToDataSetCreator(file);
		var features = creator.getFeatureArray();
		for (int i = 0; i < features.length; i++) {
			var expectedFeatureColumns = featuresExpected[i];
			assertArrayEquals(expectedFeatureColumns, features[i]);
		}
	}

	private File createFile() throws IOException {
		var tempFile = Files.createTempFile("happy_path_file", ".txt");
		var fileText = "[[4.0,7.0,2.0],[7.0,8.0,7.0],[6.0,2.0,7.0],[7.0,0.0,2.0],[3.0,5.0,8.0]]";
		Files.write(tempFile, fileText.getBytes());
		return tempFile.toFile();
	}

	@Test
	void createLSTMFeatureLabelArray_HappyPath_ShouldReturnCorrectFeatureLabel() throws IOException {
		var file = createFile();
		var creator = new TechnicalAnalysisToDataSetCreator(file);
		DatasetBuilder datasetBuilder = creator.createLSTMFeatureLabelDatasetBuilder(2);
		var featureExpected = "[[[0.4, 1.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0], [0.8, 0.0, 1.0]], [[0.8, 0.0, 1.0], [1.0, 0.0, 0.2857142857142857]], [[1.0, 0.0, 0.2857142857142857], [0.0, 0.4, 1.0]]]";
		var featureReturned = Arrays.deepToString(datasetBuilder.getFeatures());
		assertEquals(featureExpected, featureReturned);
		var labelsExpected = "[[[0.8, 0.0, 1.0]], [[1.0, 0.0, 0.2857142857142857]], [[0.0, 0.4, 1.0]], [[0.0, 0.0, 0.0]]]";
		var labelsReturned = Arrays.deepToString(datasetBuilder.getLabels());
		assertEquals(labelsExpected, labelsReturned);
	}

}