// See https://aka.ms/new-console-template for more information
using System.Text.Json;
using CSVUtils;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuralNetwork;
using NeuralNetwork.ChartUtils;
using NeuralNetwork.Models;


Console.WriteLine("Started Execution");

/* =========================================
 * READING DATA
 * ========================================= */

DataReader.CSV_CONFIGURATION.Delimiter = "\t";
//DataReader.CSV_CONFIGURATION.WhiteSpaceChars = Array.Empty<char>();

var unformattedData = DataReader.ReadData<dynamic>("Files/seeds_dataset.txt");

var data = unformattedData
           .Select(x => ((IDictionary<string, object>)x).Values)
           .Select(x => new Sample()
           {
               Label = (string)x.Last(),
               X = Vector.Build.Dense(x.SkipLast(1).Select(Convert.ToDouble).ToArray()),
               ExpectedResult = GetExpectedResultFromLabel((string)x.Last())
           }).ToList();

/* =========================================
 * Choosing training and testing data
 * ========================================= */

const double trainingDataPercentage = 0.7;




/* =========================================
 * Executing Neural Network
 * ========================================= */

const string title = $"====> Neural Network Wheat Detection <====";
Console.WriteLine(title);

var perceptron = new Perceptron(3, 7)
{
    ResultTransformFunc = (result) =>
    {
        var maxValue = result.EnumerateIndexed().MaxBy(x => x.Item3);
        result.Clear();
        result[maxValue.Item1, maxValue.Item2] = 1;
        return result;
        //result.MapInplace(x => 1 / (1 + Math.Exp(-x)), Zeros.Include);
        //return result;
    }
};

perceptron.Train(data);

Console.WriteLine($"Weights: {perceptron.Weights}");
Console.WriteLine($"Bias: {perceptron.Bias}");

ChartUtils.PlotNeuralNetwork(perceptron, title);

/* =========================================
 * Finishing Program
 * ========================================= */

Console.WriteLine("Finished Execution");

// =========================================================================

static Vector<double> GetExpectedResultFromLabel(string label)
{
    var result = Vector.Build.Dense(new[]
    {
        "1".Equals(label) ? 1.0 : 0.0,
        "2".Equals(label) ? 1.0 : 0.0,
        "3".Equals(label) ? 1.0 : 0.0,
    });

    return result;
}