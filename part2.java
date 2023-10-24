
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;

/* Name: Aidan Weinreber
 * Date: 10/23/23
 * Description: Neural network that can train randomly or input a previously saved weights and biases
 *              User is capable of inputting to determine what they wish to do and gain more options once they have a trained network
 *              Network contains a 784-20-10 node network to evaluate a 28 by 28 image to determine what number is contained.
 *              The output for each image is transferred into a 1 hot vector to compare to the label that is also a 1 hot vector.
 *              Secondary user input allows the user to iterate over the training and test sets, as well as display the ascii representation of the numbers being read, and 
 *              the ability to save the current state of the network.
 * 
 */

public class part2 {

    //layer variables, can be changed to update inputs, hidden layer, or output. 
    private static int layerOneNodes = 784;
    private static int layerTwoNodes = 15;
    private static int layerOutputNodes = 10;

    //sets are loaded at all times to be used
    private static double[][] trainingSet = new double[60000][layerOneNodes];
    private static double[] trainingOutput = new double[60000];
    private static double[][] testSet = new double[10000][layerOneNodes];
    private static double[] testOutput = new double[10000];

    //weights and biases for each layer
    private static double[][] weightsToHidden = new double[layerTwoNodes][layerOneNodes];
    private static double[][] weightsToOutput = new double[layerOutputNodes][layerTwoNodes];
    private static double[] biasForHidden = new double[layerTwoNodes];
    private static double[] biasForOutput = new double[layerOutputNodes];

    private static String saveState = "saveState.csv";

    private static Scanner userInput;


    private static double eta = 1;
    
    //main will load in all inputs from training and test set, then go to user input
    public static void main(String[] args) throws IOException{
        loadTrainingSet(trainingSet, trainingOutput);
        scaleInputs(trainingSet);
        loadTestSet(testSet, testOutput);
        scaleInputs(testSet);
        terminalOne();
    }
    //
    // functions to load sets and scale them appropriately
    //
    
    //funciton to load training set. Divides the training set into labels and pixels
    public static void loadTrainingSet(double[][] trainingInput, double[] trainingOutput)throws IOException{
        double[][] trainingSetFinal = new double[60000][785];
        Scanner sc = new Scanner(new File("mnist_train.csv"));
        sc.useDelimiter(",|\\n");

        for(int i = 0; i<= trainingSetFinal.length-1;i++){
            for(int j= 0;j<=trainingSetFinal[0].length-1;j++){
                trainingSetFinal[i][j] = Integer.parseInt(sc.next());
            }
        }
        sc.close();

        for(int i = 0; i <= trainingSetFinal.length-1;i++){
            trainingOutput[i] = trainingSetFinal[i][0];
            for(int j=1; j<=trainingSetFinal[0].length-1;j++){
                trainingInput[i][j-1] = trainingSetFinal[i][j];
            }
        }
    }

    //function to load test set and separates the labels and pixels
    public static void loadTestSet(double[][] testInput, double[] testOutput)throws IOException{
        double[][] testSetFinal = new double[10000][785];
        Scanner sc = new Scanner(new File("mnist_test.csv"));
        sc.useDelimiter(",|\\n");

        for(int i = 0; i<= testSetFinal.length-1;i++){
            for(int j= 0;j<=testSetFinal[0].length-1;j++){
                testSetFinal[i][j] = Integer.parseInt(sc.next());
            }
        }
        sc.close();

        for(int i = 0; i <= testSetFinal.length-1;i++){
            testOutput[i] = testSetFinal[i][0];
            for(int j=1; j<=testSetFinal[0].length-1;j++){
                testInput[i][j-1] = testSetFinal[i][j];
            }
        }
    }

    //function to scale to between 0 and 1
    public static void scaleInputs(double[][] inputs){
        for(int i = 0; i<=inputs.length-1;i++){
            for(int j = 0; j<=inputs[0].length-1;j++){
                inputs[i][j] = inputs[i][j]/255;
            }
        }
    }
    //
    // end of section
    //

    //
    // functions to display user interface
    //
    
    //function to generate statements for terminal
    public static void terminalOne()throws IOException{
        userInput = new Scanner(System.in);
        while(true){
            System.out.println("[1] Train the Network.");
            System.out.println("[2] Load a Pre-trained Network.");
            System.out.println("[0] Exit Program.");
                
            var input = userInput.nextLine();

            if(input.equals("0")){
                break;
            }
            else if(input.equals("1")){
                //train the network
                trainNetwork();
                //second output terminal
                terminalTwo();
                break;
            }
            else if(input.equals("2")){
                //load pretrained network
                loadNetworkState();
                //second output terminal
                terminalTwo();
                break;
            }
            else{
                System.out.println("Invalid Input, Try Again.");
            }
        }
        userInput.close();
    }

    //function to generate secondary level statements
    public static void terminalTwo()throws IOException{
        while(true) {
            System.out.println("[3] Display network accuracy on TRAINING data");
            System.out.println("[4] Display network accuracy on TESTING data");
            System.out.println("[5] Run network on TESTING data showing images and labels");
            System.out.println("[6] Display the misclassified TESTING images");
            System.out.println("[7] Save the network state to file");
            System.out.println("[0] Exit");

            var input = userInput.nextLine();

            if(input.equals("0")){
                break;
            }
            else if(input.equals("3")){
                runTrainingSet();
            }
            else if(input.equals("4")){
                runTestSet();
            }
            else if(input.equals("5")){
                printTestingNumbers(true);
            }
            else if(input.equals("6")){
                printTestingNumbers(false);
            }
            else if(input.equals("7")){
                saveNetworkState();
            }
            else{
                System.out.println("Invalid input. Just input the number.");
            }
        }
    }

    //print out image and if correct or incorrect
    public static void printTestingNumbers(boolean printAll){
        int dimensions = (int) Math.pow(testSet[0].length, 0.5);
        for(int test = 0; test <=testSet.length-1; test++){
            double[] resultOne = calculateActivation(weightsToHidden, testSet[test], biasForHidden);

            double[] resultTwo = calculateActivation(weightsToOutput, resultOne, biasForOutput);

            double[] oneHotVector = generateOneHotVectorOutput(resultTwo);

            double[] expectedOneHotVector = generateOneHotVector(testOutput[test]);
            
            int currentPixel = 0;
            int actualOutputNumber = 0;
            int outputNumber = 0;
            for(int i = 1; i<=expectedOneHotVector.length-1;i++){
                if(expectedOneHotVector[outputNumber]<expectedOneHotVector[i]){
                    outputNumber = i;
                }
            }
            for(int i = 1; i<=oneHotVector.length-1;i++){
                if(oneHotVector[actualOutputNumber]<oneHotVector[i]){
                    actualOutputNumber = i;
                }
            }
            if(printAll){
                System.out.print("Testing Case #" + (test+1) + ":\tCorrect Classification = ");
                System.out.print(outputNumber + "\tNetwork Output = ");
                System.out.print(actualOutputNumber + "\t");
                System.out.print(actualOutputNumber == outputNumber ? "Correct." : "Incorrect");
                for(int i = 0; i<=dimensions-1;i++){
                    System.out.print("\t");
                    for(int j = 0; j<=dimensions-1;j++){
                        if(testSet[test][currentPixel] == 0){
                            System.out.print("  ");
                        }
                        else{
                            char asciiRepresentation = (char) (int) (testSet[test][currentPixel] * 255);
                            System.out.print(asciiRepresentation + " ");
                        }
                        currentPixel++;
                    }
                    System.out.println();
                }
                System.out.println("Press [1] to continue. All other inputs will lead back to the menu.");
                String s = userInput.nextLine();
                if(s.equals("1")){
                    continue;
                }
                else{
                    break;
                }
                
            }
            else if(actualOutputNumber != outputNumber){
                System.out.print("Testing Case #" + (test+1) + ":\tCorrect Classification = ");
                System.out.print(outputNumber + "\tNetwork Output = ");
                System.out.print(actualOutputNumber + "\t");
                System.out.print(actualOutputNumber == outputNumber ? "Correct." : "Incorrect");
                for(int i = 0; i<=dimensions-1;i++){
                    System.out.print("\t");
                    for(int j = 0; j<=dimensions-1;j++){
                        if(testSet[test][currentPixel] == 0){
                            System.out.print("  ");
                        }
                        else{
                            char asciiRepresentation = (char) (int) (testSet[test][currentPixel] * 255);
                            System.out.print(asciiRepresentation + " ");
                        }
                        currentPixel++;
                    }
                    System.out.println();
                }
                System.out.println("Press [1] to continue. All other inputs will lead back to the menu.");
                String s = userInput.nextLine();
                if(s.equals("1")){
                    continue;
                }
                else{
                    break;
                }
            }
        }
    }
    //
    //   end of section
    //

    //function to train network by generating random weights and biases and adjusting off those.
    public static void trainNetwork()throws IOException{
        weightsToHidden = generateRandomWeights(weightsToHidden);
        weightsToOutput = generateRandomWeights(weightsToOutput);

        biasForHidden = generateRandomBiases(biasForHidden);
        biasForOutput = generateRandomBiases(biasForOutput);

        int miniBatchSize = 10;

        double[][] biasGradientStorageHidden = new double[miniBatchSize][layerTwoNodes];
        double[][] biasGradientStorageOutput = new double[miniBatchSize][layerOutputNodes];

        double[][][] weightGradientStorageHidden = new double[miniBatchSize][layerTwoNodes][layerOneNodes];
        double[][][] weightGradientStorageOutput = new double[miniBatchSize][layerOutputNodes][layerTwoNodes];

        Random rand = new Random();
        int randInt;

        //epoch and minibatch(10 training sets each) each epoch goes over all inputs, each mini batch  runs the same amount of inputs(though different)
        //I found that the best accuracy that I was getting was found with large mini-batch sizes
        for(int epoch = 0; epoch<=59; epoch++){
            for(int mini = 0; mini<=1999; mini++){
                for(int set = 0; set<=miniBatchSize-1;set++){
                    randInt = rand.nextInt(59999);

                    double[] resultOne = calculateActivation(weightsToHidden, trainingSet[randInt], biasForHidden);

                    double[] resultTwo = calculateActivation(weightsToOutput, resultOne, biasForOutput);

                    double[] expectedOneHotVector = generateOneHotVector(trainingOutput[randInt]);

                    biasGradientStorageOutput[set] = calculateBiasGradientOutputLayer(resultTwo, expectedOneHotVector);

                    biasGradientStorageHidden[set] = calculateBiasGradientHiddenLayer(weightsToOutput, resultOne, biasGradientStorageOutput[set]);

                    weightGradientStorageOutput[set] = calculateGradientOfWeights(biasGradientStorageOutput[set], resultOne);

                    weightGradientStorageHidden[set] = calculateGradientOfWeights(biasGradientStorageHidden[set], trainingSet[randInt]);

                }
                //update weights and biases
                biasForHidden = calculateNewBias(biasForHidden, eta, biasGradientStorageHidden);

                biasForOutput = calculateNewBias(biasForOutput, eta, biasGradientStorageOutput);

                weightsToHidden = calculateNewWeights(weightsToHidden, eta, weightGradientStorageHidden);

                weightsToOutput = calculateNewWeights(weightsToOutput, eta, weightGradientStorageOutput);
            }
            //run through training set to test current accuracy
            runTrainingSet();
        }
        
    }

    //run over training set once and determines the accuracy of the current weights and biases
    public static void runTrainingSet(){
        int[] totalNumbers = {0,0,0,0,0,0,0,0,0,0};
        int[] correctNumbers = {0,0,0,0,0,0,0,0,0,0};

        for(int test = 0; test <=trainingSet.length-1; test++){
            double[] resultOne = calculateActivation(weightsToHidden, trainingSet[test], biasForHidden);

            double[] resultTwo = calculateActivation(weightsToOutput, resultOne, biasForOutput);

            double[] oneHotVector = generateOneHotVectorOutput(resultTwo);

            double[] expectedOneHotVector = generateOneHotVector(trainingOutput[test]);

            comparison(oneHotVector, expectedOneHotVector, totalNumbers, correctNumbers);
        }

         accuracyGenerator(correctNumbers, totalNumbers);
    }

    //run over test set once and determinese the accuracy of the current weights and biases
    public static void runTestSet(){
        int[] totalNumbers = {0,0,0,0,0,0,0,0,0,0};
        int[] correctNumbers = {0,0,0,0,0,0,0,0,0,0};

        for(int test = 0; test <=testSet.length-1; test++){
            double[] resultOne = calculateActivation(weightsToHidden, testSet[test], biasForHidden);

            double[] resultTwo = calculateActivation(weightsToOutput, resultOne, biasForOutput);

            double[] oneHotVector = generateOneHotVectorOutput(resultTwo);

            double[] expectedOneHotVector = generateOneHotVector(testOutput[test]);

            comparison(oneHotVector, expectedOneHotVector, totalNumbers, correctNumbers);
        }

        accuracyGenerator(correctNumbers, totalNumbers);
    }

    //print accuracy and occurances
    public static void accuracyGenerator(int[] correctNumbers, int[] totalNumbers){
        System.out.print("0 = " + correctNumbers[0] + "/" + totalNumbers[0] + "\t");
        System.out.print("1 = " + correctNumbers[1] + "/" + totalNumbers[1] + "\t");            
        System.out.print("2 = " + correctNumbers[2] + "/" + totalNumbers[2] + "\t");
        System.out.print("3 = " + correctNumbers[3] + "/" + totalNumbers[3] + "\t");
        System.out.print("4 = " + correctNumbers[4] + "/" + totalNumbers[4] + "\t");
        System.out.print("5 = " + correctNumbers[5] + "/" + totalNumbers[5] + "\n");
        System.out.print("6 = " + correctNumbers[6] + "/" + totalNumbers[6] + "\t");
        System.out.print("7 = " + correctNumbers[7] + "/" + totalNumbers[7] + "\t");
        System.out.print("8 = " + correctNumbers[8] + "/" + totalNumbers[8] + "\t");
        System.out.print("9 = " + correctNumbers[9] + "/" + totalNumbers[9] + "\t");
        
        int correctAnswers = 0;
        int totalAnswers = 0;
        for(int i = 0; i <= correctNumbers.length-1;i++){
            correctAnswers += correctNumbers[i];
            totalAnswers += totalNumbers[i];
        }
        double accuracy = ((double) correctAnswers * 100) /(double) totalAnswers;

        System.out.print("Accuracy = " + correctAnswers + "/" + totalAnswers + " = " + String.format("%.3f",accuracy) + "%\n\n");

        for(int i = 0; i <=correctNumbers.length-1;i++){
            correctNumbers[i] = 0;
            totalNumbers[i] = 0;
        }
    }

    //increase occurance and correct input based on the two one hot vectors.
    public static void comparison(double[] actualOneHot, double[] expectedOneHot, int[] totalOccurances, int[] correctOccurances){
        int onePosition = 0;

        for(int i = 1; i <= expectedOneHot.length-1;i++){
            if(expectedOneHot[i] == 1){
                onePosition = i;
            }
        }

        if(expectedOneHot[onePosition] == actualOneHot[onePosition]){
            correctOccurances[onePosition] +=1;
        }
        totalOccurances[onePosition] += 1;
    }

    //generate one hot vector for actual output
    public static double[] generateOneHotVectorOutput(double[] actualInputs){
        double[] result = {0,0,0,0,0,0,0,0,0,0};
        int max = 0;
        for(int i = 1; i<= actualInputs.length-1; i++){
            if(actualInputs[max]< actualInputs[i]){
                max = i;
            }
        }
        result[max] = 1;

        return result;
    }

    //generate one hot vector for expected output
    public static double[] generateOneHotVector(double expectedOutput){
        double[] result = {0,0,0,0,0,0,0,0,0,0};
        result[(int) expectedOutput] = 1;
        return result;
    }

    //function to generate random weights for training the network
    public static double[][] generateRandomWeights(double[][] weightFrame){
        double[][] weights = new double[weightFrame.length][weightFrame[0].length];
        Random rand = new Random();

        for(int i = 0; i<=weights.length-1;i++){
            for(int j= 0; j<= weights[0].length-1;j++){
                weights[i][j] = rand.nextDouble() * 2 -1;
            }
        }

        return weights;
    }

    //function to generate random biases for training the network
    public static double[] generateRandomBiases(double[] biasFrame){
        double[] biases = new double[biasFrame.length];
        Random rand = new Random();

        for(int i = 0; i<= biases.length-1; i++){
            biases[i] = rand.nextDouble() * 2 -1;
        }

        return biases;
    }

    //function to load weights from a file, will give an error if the file does not exist.
    public static void loadNetworkState(){
        try{
            Scanner sc = new Scanner(new File("saveState.csv"));
            sc.useDelimiter(",|\\n");
    
            for(int i = 0; i<= weightsToHidden.length-1;i++){
                for(int j= 0;j<=weightsToHidden[0].length-1;j++){
                    weightsToHidden[i][j] = Double.parseDouble(sc.next());
                }
            }
            for(int i = 0; i<= weightsToOutput.length-1;i++){
                for(int j= 0;j<=weightsToOutput[0].length-1;j++){
                    weightsToOutput[i][j] = Double.parseDouble(sc.next());
                }
            }
            for(int i = 0; i <= biasForHidden.length-1; i++){
                biasForHidden[i] = Double.parseDouble(sc.next());
            }
            for(int i = 0; i <= biasForOutput.length-1; i++){
                biasForOutput[i] = Double.parseDouble(sc.next());
            }
            sc.close();
        }
        catch(IOException e){
            System.out.println("Either an error occurred, the file does not exist, or the file is not formatted correctly.");
        }
        
    }

    //saves the current network into a csv format that can be loaded back into this file, files is save with one-hidden weights, hidden-output weights, one-hidden biases, hidden-output biases
    public static void saveNetworkState(){
        try{
            FileWriter toFile = new FileWriter(saveState);
            String outputString = "";
            for(int i =0; i<=weightsToHidden.length-1;i++){
                for(int j = 0; j<=weightsToHidden[0].length-1;j++){
                    outputString += weightsToHidden[i][j];
                    if(j == weightsToHidden[0].length-1){
                        outputString += "\n";
                    }
                    else{
                        outputString += ",";
                    }

                }
            }
            for(int i =0; i<=weightsToOutput.length-1;i++){
                for(int j=0; j<=weightsToOutput[0].length-1;j++){
                    outputString += weightsToOutput[i][j];
                    if(j == weightsToOutput[0].length-1){
                        outputString += "\n";
                    }
                    else{
                        outputString += ",";
                    }
                }
            }
            for(int i = 0; i<=biasForHidden.length-1; i++){
                outputString += biasForHidden[i];
                if(i == biasForHidden.length-1){
                    outputString += "\n";
                }
                else{
                    outputString += ",";
                }
            }
            for(int i = 0; i<=biasForOutput.length-1;i++){
                outputString += biasForOutput[i];
                if(i == biasForOutput.length-1){
                    outputString += "\n";
                }
                else{
                    outputString += ",";
                }
            }
            toFile.write(outputString);
            toFile.close();
        }
        catch(IOException e){
            System.out.println("Something went wrong saving this file");
        }
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

    //calculates the new biases for each layer and stores them in a nX1 matrix, pulls in a 2d matrix of bias gradients
    public static double[] calculateNewBias(double[] originalBias, double trainingRate, double[][] biasGradient){
        double[] newBias = new double[originalBias.length];
        double sum = 0;

        for(int i = 0; i<=originalBias.length-1;i++){
            for(int j = 0; j<=biasGradient.length-1; j++){
                sum = sum + biasGradient[j][i];
            }
            newBias[i] = originalBias[i] - ((trainingRate/((double)biasGradient.length)) * (sum));
            sum = 0;
        }
        return newBias;
    }

    //calucaltes the new weights for each layer and stores them in a mXn matrix, pulls in a 3d matrix of weight gradients
    public static double[][] calculateNewWeights(double[][] originalWeights, double trainingRate, double[][][] weightGradients){
        double[][] newWeights = new double[originalWeights.length][originalWeights[0].length];
        double sum = 0;

        for(int i = 0; i <=originalWeights.length-1;i++){
            for(int j = 0; j<=originalWeights[0].length-1;j++){
                for(int k = 0;k<=weightGradients.length-1;k++){
                    sum += weightGradients[k][i][j];
                }
                newWeights[i][j] = originalWeights[i][j] - ((trainingRate/((double)weightGradients.length)) * (sum));
                sum = 0;
            }
        }
        
        return newWeights;
    }
}