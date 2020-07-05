package com.wealthy.learning;

import com.wealthy.learning.failure.TechnicalAnalysisToDataSetCreator;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class TechnicalAnalysisToDataSetCreatorTest {

	@Test
	void getDataset_HappyPath_ShouldReturnCorrectDataSet() throws IOException {
		var file = createFile();
		var creator = new TechnicalAnalysisToDataSetCreator(file, 2, 3);
		var featuresExpected = new double[][] {
				new double[]{0.4, 1, 0},
				new double[]{0, 1, 0},
				new double[]{0.8, 0, 1}
		};
		var labelsExpected = new double[][] {
				new double[]{1, 1},
				new double[]{0, 0.2857142857142857},
				new double[]{0, 1}
		};
		for (int i = 0; i < featuresExpected.length; i++) {
			var featureColumns = creator.getFeatureArray()[i];
			var expectedFeatureColumns = featuresExpected[i];
			assertArrayEquals(expectedFeatureColumns, featureColumns);
			var labelColumns = creator.getLabelArray()[i];
			var expectedLabelColumns = labelsExpected[i];
			assertArrayEquals(expectedLabelColumns, labelColumns);

		}
	}

	private File createFile() throws IOException {
		var tempFile = Files.createTempFile("happy_path_file", ".txt");
		var fileText = "[[4.0,7.0,2.0],[7.0,8.0,7.0],[6.0,2.0,7.0],[7.0,0.0,2.0],[3.0,5.0,8.0]]";
		Files.write(tempFile, fileText.getBytes());
		return tempFile.toFile();
	}

}