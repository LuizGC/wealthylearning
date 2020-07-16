package com.wealthy.learning;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class DatasetBuilder {
	private final double[][][] features;
	private final double[][][] labels;

	public DatasetBuilder(double[][] featureArray, int totalTimelap) {
		int columns = featureArray[0].length;
		int rows = featureArray.length - totalTimelap + 1;
		this.features = new double[rows][columns][totalTimelap];
		this.labels = new double[rows][columns][totalTimelap];
		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < columns; column++) {
				for (int timelap = 0; timelap < totalTimelap; timelap++) {
					this.features[row][column][timelap] = featureArray[row + timelap][column];
					if (row+timelap+1 < featureArray.length) {
						this.labels[row][column][timelap] = featureArray[row+timelap+1][column];
					}
				}
			}
		}
	}

	public double[][][] getFeatures() {
		return this.features;
	}

	public double[][][] getLabels() {
		return this.labels;
	}

	public double[][][] getLabels(int columnLabel) {
		var rowsNumber = this.labels.length;
		var timelaps = this.labels[0][0].length;
		var indexLabel = new double[rowsNumber][1][timelaps];
		for (int i = 0; i < rowsNumber; i++) {
			indexLabel[i][0] = this.labels[i][columnLabel];
		}
		return indexLabel;
	}

	public DataSet createTrainDataSet(int columnLabel) {
		var features = Nd4j.create(Arrays.copyOf(this.features, this.features.length - 1));
		var labelArray = getLabels(columnLabel);
		var labels = Nd4j.create(Arrays.copyOf(labelArray, labelArray.length - 1));
		return new DataSet(features, labels);
	}
}
