package com.wealthy.learning.failure;

import org.apache.commons.math3.stat.StatUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public final class TechnicalAnalysisToDataSetCreator {

	private final double[][] featureArray;
	private final double[][] labelArray;

	public TechnicalAnalysisToDataSetCreator(final File file, final int buyColumn, final int sellColumn) throws IOException {
		var rows = getRows(file);
		var features = Stream
				.of(rows)
				.map(this::getStringArrayToDouble)
				.collect(Collectors.toUnmodifiableList());
		var featureNormalizedArray = createFeatureArray(features);
		var size = features.size() - 2;
		var columnSize = features.get(0).length;
		this.featureArray = new double[size][columnSize];
		this.labelArray = new double[size][2];
		for (var i = 0; i < size; i++) {
			for (var j = 0; j < columnSize; j++) {
				this.featureArray[i][j] = featureNormalizedArray[i][j];
			}
			this.labelArray[i][0] = featureNormalizedArray[i+1][buyColumn-1];
			this.labelArray[i][1] = featureNormalizedArray[i+2][sellColumn-1];
		}
	}

	private double[][] createFeatureArray(List<double[]> features) {
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

	public double scaling(double value, double min, double max) {
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

	public double[][] getFeatureArray() {
		return featureArray;
	}

	public double[][] getLabelArray() {
		return labelArray;
	}

}
