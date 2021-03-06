package com.wealthy.learning;

import org.apache.commons.math3.stat.StatUtils;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public final class TechnicalAnalysisToDataSetCreator {

	private final double[][] normalizedFeatureArray;

	public TechnicalAnalysisToDataSetCreator(final File file) throws IOException {
		var rows = getRows(file);
		var features = Stream
				.of(rows)
				.map(this::getStringArrayToDouble)
				.collect(Collectors.toUnmodifiableList());
		var featureNormalizedArray = createNormalizeFeatureArray(features);
		var size = features.size();
		var columnSize = features.get(0).length;
		this.normalizedFeatureArray = new double[size][columnSize];
		for (var i = 0; i < size; i++) {
			System.arraycopy(featureNormalizedArray[i], 0, this.normalizedFeatureArray[i], 0, columnSize);
		}
	}

	private double[][] createNormalizeFeatureArray(List<double[]> features) {
		var featureArray = new double[features.size()][];
		for (var i = 0; i < features.size(); i++) {
			var feature = features.get(i);
			var min = StatUtils.min(feature);
			var max = StatUtils.max(feature);
			featureArray[i] = new double[feature.length];
			for (int j = 0; j < feature.length; j++) {
				featureArray[i][j] = scaling(feature[j], min, max);
			}
		}
		return featureArray;
	}

	private double scaling(double value, double min, double max) {
		return (value - min)/(max - min);
	}

	private double[] getStringArrayToDouble(String row) {
		return Stream.of(row.split(","))
				.mapToDouble(Double::parseDouble)
				.toArray();
	}

	private String[] getRows(File file) throws IOException {
		var fileText = Files.readString(file.toPath());
		fileText = fileText.substring(2, fileText.length() -2);
		return fileText.split("],\\[");
	}

	public double[][] getNormalizedFeatureArray() {
		return normalizedFeatureArray;
	}

	public DatasetBuilder createLSTMFeatureLabelDatasetBuilder(int timelap) {
		return new DatasetBuilder(normalizedFeatureArray, timelap);
	}

}
