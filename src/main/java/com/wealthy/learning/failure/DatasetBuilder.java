package com.wealthy.learning.failure;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class DatasetBuilder {
	private double [][][] features;
	private double [][][] labels;

	public DatasetBuilder(double[][] featureArray, int totalTimelap) {
		int columns = featureArray[0].length;
		int rows = featureArray.length - totalTimelap + 1;
		this.features = new double[rows][columns][totalTimelap];
		this.labels = new double[rows][columns][totalTimelap];
		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < columns; column++) {
				for (int timelap = 0; timelap < totalTimelap; timelap++) {
					this.features[row][column][timelap] = featureArray[row+timelap][column];
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

	public DataSet create() {
		INDArray features = Nd4j.create(this.features);
		INDArray labels = Nd4j.create(this.labels);
		return new DataSet(features, labels);
	}
}
