package de.htw.ml;

import java.io.IOException;

import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.ranges.IntervalRange;
import org.jblas.ranges.Range;
import org.jblas.util.Random;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;

public class ML_Ue06_Solihin_Martin_Christian {

	// TODO change the names of the axis
	public static final String title = "Line Chart";
	public static final String xAxisLabel = "Iteration";
	public static final String yAxisLabel = "Credit Amount";
	
	public static void main(String[] args) throws IOException {
		FloatMatrix read_data = FloatMatrix.loadCSVFile("german_credit_jblas.csv");
		
		//read the data
		Range rangeX1 = new IntervalRange(0, 5);
		Range rangeX2 = new IntervalRange(6, 21);
		FloatMatrix x1 = read_data.getColumns(rangeX1);
		FloatMatrix x2 = read_data.getColumns(rangeX2);
		FloatMatrix x = FloatMatrix.concatHorizontally(x1,x2);
		FloatMatrix y = read_data.getColumn(5);
		int sumCar = read_data.rows;
		int sumColumn = x.columns;
		int iteration = 100;
		
		//initialize values
		Random.seed(7);
		FloatMatrix theta = FloatMatrix.rand(sumColumn,1);
		FloatMatrix theta1 = theta;
		int m = read_data.rows;
		float alpha = 0.3f;
		float [] rmse = new float [iteration];
		
		//normalize the data input
		FloatMatrix n_x = normalize(x);
		FloatMatrix n_y = normalize(y);
		
		//start linear regression
		FloatMatrix hypothese = n_x.mulRowVector(theta).rowSums();
			//plotting the first hypothese (e=0)
			FloatMatrix prediction = denormalize(y,hypothese);
			FloatMatrix rmse_Matrix = MatrixFunctions.sqrt(((MatrixFunctions.pow(prediction.sub(y),2)).columnSums()).div(sumCar));
				float valueRMSE = rmse_Matrix.get(0);
				rmse[0] = valueRMSE;
		FloatMatrix diff = hypothese.sub(n_y);
		FloatMatrix diff1 = diff;
		
		FloatMatrix n_xT = n_x.transpose();
		for(int i=1; i<iteration; i++) {
			FloatMatrix theta_delta = n_xT.mulRowVector(diff1).rowSums();
			FloatMatrix normalize_theta_delta = theta_delta.mul(alpha/m);
			FloatMatrix new_theta = theta1.sub(normalize_theta_delta);	
			FloatMatrix new_hypothese = n_x.mulRowVector(new_theta).rowSums();
			FloatMatrix new_diff = new_hypothese.sub(n_y); 
				diff1 = new_diff;
				theta1 = new_theta;
			// denormalize the data input
			FloatMatrix new_prediction = denormalize(y, new_hypothese);
			rmse_Matrix = MatrixFunctions.sqrt(((MatrixFunctions.pow(new_prediction.sub(y),2)).columnSums()).div(sumCar));
				valueRMSE = rmse_Matrix.get(0);
				rmse[i] = valueRMSE;
		}
		System.out.println("Best RMSE from " + iteration + " iteration with alpha " + alpha + " is " + rmse[iteration-1]);
		
		// plot the RMSE values
		FXApplication.plot(rmse); 	
		
		Application.launch(FXApplication.class);
		
	}
	
	public static FloatMatrix normalize(FloatMatrix n_data){
		FloatMatrix xmin = n_data.columnMins();
		FloatMatrix xmax = n_data.columnMaxs();
		
		FloatMatrix normalize = (n_data.subRowVector(xmin)).divRowVector(xmax.sub(xmin));
		return normalize;
	}
	
	public static FloatMatrix denormalize(FloatMatrix n_data, FloatMatrix norm_data){
		FloatMatrix xmin = n_data.columnMins();
		FloatMatrix xmax = n_data.columnMaxs();
		
		FloatMatrix denormalize = norm_data.mulRowVector(xmax.sub(xmin)).addRowVector(xmin);
		return denormalize;
	}
	
	
	
	// ---------------------------------------------------------------------------------
	// ------------ Alle Änderungen ab hier geschehen auf eigene Gefahr ----------------
	// ---------------------------------------------------------------------------------
	
	
	/**
	 * We need a separate class in order to trick Java 11 to start our JavaFX application without any module-path settings.
	 * https://stackoverflow.com/questions/52144931/how-to-add-javafx-runtime-to-eclipse-in-java-11/55300492#55300492
	 * 
	 * @author Nico Hezel
	 *
	 */
	public static class FXApplication extends Application {
	
		/**
		 * equivalent to linspace in Octave
		 * 
		 * @param lower
		 * @param upper
		 * @param num
		 * @return
		 */
		private static FloatMatrix linspace(float lower, float upper, int num) {
	        float[] data = new float[num];
	        float step = Math.abs(lower-upper) / (num-1);
	        for (int i = 0; i < num; i++)
	            data[i] = lower + (step * i);
	        data[0] = lower;
	        data[data.length-1] = upper;
	        return new FloatMatrix(data);
	    }
		
		// y-axis values of the plot 
		private static float[] dataY;
		
		/**
		 * Draw the values and start the UI
		 */
		public static void plot(float[] yValues) {
			dataY = yValues;
		}
		
		/**
		 * Draw the UI
		 */
		@SuppressWarnings("unchecked")
		@Override 
		public void start(Stage stage) {
	
			stage.setTitle(title);
			
			final NumberAxis xAxis = new NumberAxis();
			xAxis.setLabel(xAxisLabel);
	        final NumberAxis yAxis = new NumberAxis();
	        yAxis.setLabel(yAxisLabel);
	        
			final LineChart<Number, Number> sc = new LineChart<>(xAxis, yAxis);
	
			XYChart.Series<Number, Number> series1 = new XYChart.Series<>();
			series1.setName("Data");
			for (int i = 0; i < dataY.length; i++) {
				series1.getData().add(new XYChart.Data<Number, Number>(i, dataY[i]));
			}
	
			sc.setAnimated(false);
			sc.setCreateSymbols(true);
	
			sc.getData().addAll(series1);
	
			Scene scene = new Scene(sc, 500, 400);
			stage.setScene(scene);
			stage.show();
	    }
	}
}
