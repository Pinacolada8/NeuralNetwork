//// See https://aka.ms/new-console-template for more information
//using System.Text.Json;
//using CSVUtils;
//using MathNet.Numerics.LinearAlgebra;
//using MathNet.Numerics.LinearAlgebra.Double;
//using NeuralNetwork;
//using NeuralNetwork.ChartUtils;
//using NeuralNetwork.Models;

//Console.WriteLine("Started Execution");

///* =========================================
// * READING DATA
// * ========================================= */

//DataReader.CSV_CONFIGURATION.Delimiter = " ";
//DataReader.CSV_CONFIGURATION.WhiteSpaceChars = Array.Empty<char>();

//var unformattedData = DataReader.ReadData<dynamic>("Files/column_3C.dat");

//var data = unformattedData
//           .Select(x => ((IDictionary<string, object>)x).Values)
//           .Select(x => new Sample()
//           {
//               Label = (string)x.Last(),
//               X = Vector.Build.Dense(x.SkipLast(1).Select(Convert.ToDouble).ToArray()),
//               ExpectedResult = GetExpectedResultFromLabel((string)x.Last())
//           }).ToList();

//// Print input data
////foreach(var o in data)
////    Console.WriteLine(JsonSerializer.Serialize(o));

///* =========================================
// * Calculate using Step evaluation function
// * ========================================= */

//Console.WriteLine($"====> Neural Network Step Function");

//var perceptronStep = new Perceptron(3, 6)
//{
//    ResultTransformFunc = (result) =>
//    {
//        var maxValue = result.EnumerateIndexed().MaxBy(x => x.Item3);
//        result.Clear();
//        result[maxValue.Item1, maxValue.Item2] = 1;
//        return result;
//    }
//};

//perceptronStep.Train(data);

//Console.WriteLine($"Weights: {perceptronStep.Weights}");
//Console.WriteLine($"Bias: {perceptronStep.Bias}");

//ChartUtils.PlotNeuralNetwork(perceptronStep, "Neural Network Step Function");

///* =========================================
// * Calculate using Sigmoid evaluation function
// * ========================================= */

//Console.WriteLine($"====> Neural Network Sigmoid Function");

//var perceptronSigmoid = new Perceptron(3, 6)
//{
//    ResultTransformFunc = (result) =>
//    {
//        result.MapInplace(x => 1/(1 + Math.Exp(-x)) , Zeros.Include);
//        return result;
//    }
//};

//perceptronSigmoid.Train(data);

//Console.WriteLine($"Weights: {perceptronSigmoid.Weights}");
//Console.WriteLine($"Bias: {perceptronSigmoid.Bias}");

//ChartUtils.PlotNeuralNetwork(perceptronSigmoid, "Neural Network Sigmoid Function");


///* =========================================
// * Finishing Program
// * ========================================= */

//Console.WriteLine("Finished Execution");

//// =========================================================================

//static Vector<double> GetExpectedResultFromLabel(string label)
//{
//    var result = Vector.Build.Dense(new[]
//    {
//        "NO".Equals(label) ? 1.0 : 0.0,
//        "SL".Equals(label) ? 1.0 : 0.0,
//        "DH".Equals(label) ? 1.0 : 0.0,
//    });

//    return result;
//}