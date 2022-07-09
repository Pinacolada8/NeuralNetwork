// See https://aka.ms/new-console-template for more information
using System.Text.Json;
using CSVUtils;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuralNetwork;
using NeuralNetwork.ChartUtils;
using NeuralNetwork.Models;
using Utils;


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

var dataSeparatedByLabel = data.ToLookup(x => x.Label);
var trainingData = new List<Sample>();
var testData = new List<Sample>();

foreach (var dataByLabel in dataSeparatedByLabel)
{
    var trainingQty = Convert.ToInt32(Math.Ceiling(dataByLabel.Count() * trainingDataPercentage));
    var trainingDataByLabel = dataByLabel.PickRandom(trainingQty).AsList()!;
    var testDataByLabel = dataByLabel.Except(trainingDataByLabel);

    trainingData.AddRange(trainingDataByLabel);
    testData.AddRange(testDataByLabel);
}


/* =========================================
 * Executing Neural Network for Step function
 * ========================================= */

const string title = $"====> (Step) Neural Network 'wheat' Detection <====";
Console.WriteLine(title);

var perceptronStep = new Perceptron(3, 7)
{
    ResultTransformFunc = (result) =>
    {
        var maxValue = result.EnumerateIndexed().MaxBy(x => x.Item3);
        result.Clear();
        result[maxValue.Item1, maxValue.Item2] = 1;
        return result;
    }
};

perceptronStep.Train(trainingData);

Console.WriteLine($"Weights: {perceptronStep.Weights}");
Console.WriteLine($"Bias: {perceptronStep.Bias}");

var testResults = testData.Select(x => perceptronStep.CalcResultStatistics(x)).AsList()!;

Console.WriteLine($"Accuracy: {testResults.Average(x => x.CorrectResult ? 1 : 0)}");

ChartUtils.PlotNeuralNetwork(perceptronStep, title, testResults);

/* =========================================
 * Executing Neural Network for Sigmoid function
 * ========================================= */

const string titleSig = $"====>(Sigmoid) Neural Network 'wheat' Detection <====";
Console.WriteLine(titleSig);

var perceptronSig = new Perceptron(3, 7)
{
    ResultTransformFunc = (result) =>
    {
        result.MapInplace(x => 1 / (1 + Math.Exp(-x)), Zeros.Include);
        return result;
    }
};

perceptronSig.Train(trainingData);

Console.WriteLine($"Weights: {perceptronSig.Weights}");
Console.WriteLine($"Bias: {perceptronSig.Bias}");

var testResultsSig = testData.Select(x => perceptronSig.CalcResultStatistics(x)).AsList()!;

Console.WriteLine($"Accuracy: {testResultsSig.Average(x => x.CorrectResult ? 1 : 0)}");

ChartUtils.PlotNeuralNetwork(perceptronSig, titleSig, testResultsSig);

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