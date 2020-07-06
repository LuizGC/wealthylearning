package com.wealthy.learning.failure;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class DatasetBuilder {
	private double [][][] features;
	private double [][][] labels;

	public DatasetBuilder(double[][] featureArray, int timelap) {
		int columns = featureArray[0].length;
		int rows = featureArray.length - timelap + 1;
		this.features = new double[rows][timelap][columns];
		this.labels = new double[rows][1][columns];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < timelap; j++) {
				this.features[i][j] = featureArray[i+j];
			}
			if (i+timelap < featureArray.length) {
				this.labels[i][0] = featureArray[i+timelap];
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
