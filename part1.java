import java.io.FileWriter;
import java.io.IOException;

/* Name: Aidan Weinreber
 * Date: 10/13/23
 * Description: Neural Network framework to build on for assignment 2
 *              Program takes sample inputs and trains on mini-batches over multiple epochs
 *              Generates an output file that will record all inputs to outputs as well as all
 *              newly generated weights and biases for each layer.
 * 
 */
public class part1 {
    
    //given inputs
    public static double[][] inputs = {{0, 1, 0, 1},
                                        {1, 0, 1, 0},
                                        {0, 0, 1, 1},
                                        {1, 1, 0, 0}};

    //expected outputs
    public static double[][] outputs = {{0, 1},
                                            {1, 0},
                                            {0, 1}, 
                                            {1, 0}};

    public static int eta = 10;
    

    //weights and biases for input to hidden layer
    public static double[][] weightsOne = new double[3][4];
    public static double[] weightOneWhole = {-0.21, 0.72, -0.25, 1, -0.94, -0.41, -0.47, 0.63, 0.15, 0.55, -0.49, -0.75};
    public static double[] biasOne = {0.1, -0.36, -0.31};

    //weights and biases for hidden to output layer
    public static double[][] weightsTwo = new double[2][3];
    public static double[] weightTwoWhole = {0.76, 0.48, -0.73, 0.34, 0.89, -0.23};
    public static double[] biasTwo = {0.16, -0.46};


    public static void main(String[] args){
        try{
            FileWriter toFile = new FileWriter("output.txt");
            int trainingSet = 1;
            //loads in the original weights
            weightsOne = loadWeights(weightOneWhole, weightsOne);
            weightsTwo = loadWeights(weightTwoWhole, weightsTwo);

            //cycles through specific number of epochs
            for(int epoch = 0; epoch< 6; epoch++){
                //goes through the given number of training data
                for(int set = 0; set<=3; set = set+2){
                double[] setOnResultOne = calculateActivation(weightsOne, inputs[set], biasOne);

                double[] setOneResultTwo = calculateActivation(weightsTwo, setOnResultOne, biasTwo);

                double[] setOneGradientLast = calculateBiasGradientOutputLayer(setOneResultTwo, outputs[set]);

                double[] setOneGradientHidden = calculateBiasGradientHiddenLayer(weightsTwo, setOnResultOne, setOneGradientLast);

                double[][] setOneWeightGradientLast = calculateGradientOfWeights(setOneGradientLast, setOnResultOne);

                double[][] setOneWeightGradientHidden = calculateGradientOfWeights(setOneGradientHidden, inputs[set]);

                double[] setTwoResultOne = calculateActivation(weightsOne, inputs[set+1], biasOne);

                double[] setTwoResultTwo = calculateActivation(weightsTwo, setTwoResultOne, biasTwo);

                double[] setTwoGradientLast = calculateBiasGradientOutputLayer(setTwoResultTwo, outputs[set+1]);

                double[] setTwoGradientHidden = calculateBiasGradientHiddenLayer(weightsTwo, setTwoResultOne, setTwoGradientLast);

                double[][] setTwoWeightGradientLast = calculateGradientOfWeights(setTwoGradientLast, setTwoResultOne);

                double[][] setTwoWeightGradientHidden = calculateGradientOfWeights(setTwoGradientHidden, inputs[set+1]);

                //calculating new biases and weights through back propagation
                biasOne = calculateNewBias(biasOne, eta, setOneGradientHidden, setTwoGradientHidden);

                biasTwo = calculateNewBias(biasTwo, eta, setOneGradientLast, setTwoGradientLast);

                weightsOne = calculateNewWeights(weightsOne, eta, setOneWeightGradientHidden, setTwoWeightGradientHidden);

                weightsTwo = calculateNewWeights(weightsTwo, eta, setOneWeightGradientLast, setTwoWeightGradientLast);

                //file and command prompt output, including input, output and new weights and biases generated
                toFile.write("Results for Epoch: " + (epoch+1) + ", Mini-batch: " + trainingSet + "\n");
                System.out.println("Results for Epoch: " + (epoch+1) + ", Mini-batch: " + trainingSet);
                if(trainingSet ==2){
                    trainingSet = 1;
                }
                else{
                    trainingSet++;
                }


                toFile.write("Inputs for Training Set: " + (set+1) + "\n");
                System.out.println("Inputs for Training Set: " + (set+1));
                for(double x: inputs[set]){
                    toFile.write(x + "\n");
                    System.out.println(x + "\t");
                }
                toFile.write("\n");
                System.out.println();

                toFile.write("Outputs for Training Set: " + (set+1) + "\n");
                System.out.println("Outputs for Training Set: " + (set+1));
                for(double x: setOneResultTwo){
                    toFile.write(x + "\n");
                    System.out.println(x + "\t");
                }
                toFile.write("\n");
                System.out.println();

                toFile.write("Inputs for Training Set: " + (set+2) + "\n");
                System.out.println("Inputs for Training Set: " + (set+2));
                for(double x: inputs[set+1]){
                    toFile.write(x + "\n");
                    System.out.println(x + "\t");
                }
                toFile.write("\n");
                System.out.println();

                toFile.write("Outputs for Training Set: " + (set+2) + "\n");
                System.out.println("Outputs for Training Set: " + (set+2));
                for(double x: setTwoResultTwo){
                    toFile.write(x + "\n");
                    System.out.println(x + "\t");
                }
                toFile.write("\n");
                System.out.println();

                toFile.write("New Weights for Inputs to Hidden \n");
                System.out.println("New Weights for Inputs to Hidden ");
                for(double[] x: weightsOne){
                    for(double y:x){
                        toFile.write(y + "\t");
                        System.out.print(y + "\t");
                    }
                    toFile.write("\n");
                    System.out.println();
                    
                }
                toFile.write("\n");

                toFile.write("New Weights for Hidden to Output\n");
                System.out.println("New Weights for Hidden to Output");
                for(double[] x: weightsTwo){
                    for(double y:x){
                        toFile.write(y + "\t");
                        System.out.print(y + "\t");
                    }
                    toFile.write("\n");
                    System.out.println();
                }
                toFile.write("\n");

                toFile.write("New Hidden Layer Bias\n");
                System.out.println("New Hidden Layer Bias");
                for(double x:biasOne){
                    toFile.write(x + "\n");
                    System.out.print(x + "\n");
                }
                toFile.write("\n");
                System.out.println();

                toFile.write("New Output Layer Bias\n");
                System.out.println("New Output Layer Bias");
                for(double x:biasTwo){
                    toFile.write(x + "\n");
                    System.out.print(x + "\n");
                }
                toFile.write("\n");
                System.out.println("\n");
                }
                
            }
            
            toFile.write("Final Weights for Inputs to Hidden\n");
            System.out.println("Final Weights for Inputs to Hidden");
                for(double[] x: weightsOne){
                    for(double y:x){
                        toFile.write(y + "\t");
                        System.out.print(y + "\t");
                    }
                    toFile.write("\n");
                    System.out.println();
                }
            toFile.write("\n");

            toFile.write("Final Weights for Hidden to Output\n");
            System.out.println("Final Weights for Hidden to Ouput");
            for(double[] x: weightsTwo){
                for(double y:x){
                    toFile.write(y + "\t");
                    System.out.print(y + "\t");
                }
                toFile.write("\n");
                System.out.println();
            }
            toFile.write("\n");

            toFile.write("Final Hidden Layer Bias\n");
            System.out.println("Final Hidden Layer Bias");
            for(double x:biasOne){
                toFile.write(x + "\n");
                System.out.print(x + "\n");
            }
            toFile.write("\n");
            System.out.println();

            toFile.write("Final Output Layer Bias\n");
            System.out.println("Final Output Layer Bias");
            for(double x:biasTwo){
                toFile.write(x + "\n");
                System.out.print(x + "\n");
            }
            toFile.write("\n");
            System.out.println();

            toFile.close();
        }
        catch (IOException e){
            //needed to write to output file
            System.out.println("an error has occured");
        }
        
    }
    
    //function to lead list of weights into 2D matrix
    public static double[][] loadWeights(double [] oldMatrix, double [][] frame){
        double[][] newMatrix = new double[frame.length][frame[0].length];
        int numberInMatrix = 0;

        for(int i = 0; i <= newMatrix.length -1; i++){
            for(int j =0; j<=newMatrix[0].length -1;j++){
                newMatrix[i][j] = oldMatrix[numberInMatrix];
                numberInMatrix += 1;
            }
            
        }
        
        return newMatrix;
    }

    //calculates activation of next layer and stores them in a nX1 matrix
    public static double[] calculateActivation(double[][] weights, double[] inputs, double[] bias){
        double[] intermidiateMatrix = new double[weights.length];
        double[] resultMatrix = new double[weights.length];
        double sum = 0;

        for(int i = 0; i <= weights.length-1; i++){
            for(int j = 0; j <= weights[0].length-1; j++){
                sum = sum + (inputs[j] * weights[i][j]);
            }
            intermidiateMatrix[i] = sum+ bias[i]; 
            sum = 0;
        }
        
        for(int i = 0; i <= resultMatrix.length-1; i++){
            resultMatrix[i] = 1 / (1+ Math.pow(2.71828, -intermidiateMatrix[i]));
        }

        return resultMatrix;
    }

    //calculates the bias gradient of the output layer for back propagation and stores them in a nX1 matrix
    public static double[] calculateBiasGradientOutputLayer(double[] calculatedOutput, double[] correctOutput){
        double[] biasGradients = new double[calculatedOutput.length];

        for(int i = 0; i <= calculatedOutput.length-1; i++){
            biasGradients[i] = (calculatedOutput[i]- correctOutput[i]) * calculatedOutput[i] * (1 - calculatedOutput[i]);
        }

        return biasGradients;
    }

    //calculates the bias gradient of the hidden layer for back propagation and stores them in a nX1 matrix
    public static double[] calculateBiasGradientHiddenLayer(double[][] weights, double[] activation, double[] previousGradient ){
        double[] biasGradients = new double[activation.length];
        double sum = 0;

        for(int i = 0; i <= weights[0].length-1; i++){
            for(int j = 0; j<= weights.length-1; j++){
                sum = sum +  (weights[j][i] * previousGradient[j]);
            }
            biasGradients[i] = sum * (activation[i] * (1-activation[i]));
            sum = 0;
        }

        return biasGradients;
    }

    //calculates the weight gradient for nodes and stores them in mXn matrix
    public static double[][] calculateGradientOfWeights(double[] biasGradient, double[] activation){
        double[][] weightGradient = new double[biasGradient.length][activation.length];

        for(int i = 0; i<=biasGradient.length-1;i++){
            for(int j = 0; j<=activation.length-1;j++){
                weightGradient[i][j] = biasGradient[i] * activation[j];
            }
        }

        return weightGradient;
    }

    //calculates the new biases for each layer and stores them in a nX1 matrix
    public static double[] calculateNewBias(double[] originalBias, int trainingRate, double[] biasGradientOne, double[] biasGradientTwo){
        double[] newBias = new double[originalBias.length];

        for(int i = 0; i<=originalBias.length-1;i++){
            newBias[i] = originalBias[i] - ((trainingRate/2) * (biasGradientOne[i] + biasGradientTwo[i]));
        }
        return newBias;
    }

    //calucaltes the new weights for each layer and stores them in a mXn matrix
    public static double[][] calculateNewWeights(double[][] originalWeights, int trainingRate, double[][] weightGradientOne, double[][] weightGradientTwo){
        double[][] newWeights = new double[originalWeights.length][originalWeights[0].length];

        for(int i = 0; i <=originalWeights.length-1;i++){
            for(int j = 0; j<=originalWeights[0].length-1;j++){
                 newWeights[i][j] = originalWeights[i][j] - ((trainingRate/2) * (weightGradientOne[i][j] + weightGradientTwo[i][j]));
            }
        }
        
        return newWeights;
    }

}