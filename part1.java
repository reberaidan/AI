

public class part1 {

    public static double[][] inputs = {{0, 1, 0, 1},
                                        {1, 0, 1, 0},
                                        {0, 0, 1, 1},
                                        {1, 1, 0, 0}};

    public static double[][] outputs = {{0, 1},
                                            {1, 0},
                                            {0, 1}, 
                                            {1, 0}};
    

    public static double[][] weightsOneToTwo = new double[3][4];
    public static double[] weightOneWhole = {-0.21, 0.72, -0.25, 1, -0.94, -0.41, -0.47, 0.63, 0.15, 0.55, -0.49, -0.75};
    public static double[] biasOne = {0.1, -0.36, -0.31};

    public static double[][] weightsTwoToThree = new double[2][3];
    public static double[] weightTwoWhole = {0.76, 0.48, -0.73, 0.34, 0.89, -0.23};
    public static double[] biasTwo = {0.16, -0.46};

    

    public static void main(String[] args){
        double[][] weightsOne = loadWeights(weightOneWhole, weightsOneToTwo);

        double[] resultOne = calculateActivation(weightsOne, inputs[0], biasOne);

        // for(double x: resultOne){
        //     System.out.println(x);
        // }

        double[][] weightsTwo = loadWeights(weightTwoWhole, weightsTwoToThree);

        double[] resultTwo = calculateActivation(weightsTwo, resultOne, biasTwo);

        for(double x: resultTwo){
            System.out.println(x);
        }
    }

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

    public static double[] calculateActivation(double[][] weights, double[] inputs, double[] bias){
        double[] intermidiateMatrix = new double[weights.length];
        double[] resultMatrix = new double[weights.length];
        double sum = 0;

        for(int i = 0; i <= weights.length-1; i++){
            for(int j = 0; j <= weights[0].length-1; j++){
                sum = sum + (inputs[j] * weights[i][j]);
            }
            intermidiateMatrix[i] = sum;
            sum = 0;
        }

        for(int i = 0; i <= intermidiateMatrix.length-1; i++){
            resultMatrix[i] = intermidiateMatrix[i] + bias[i];
        }

        return resultMatrix;
        
    }

}