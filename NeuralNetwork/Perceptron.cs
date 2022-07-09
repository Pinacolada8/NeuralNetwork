using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Models;
using Utils;

namespace NeuralNetwork;
public class Perceptron
{
    public int NumberOfNeurons { get; }
    public int NumberOfInputs { get; }

    public int DefaultIterationCount { get; set; } = 1000;
    public double LearningRate { get; set; } = 0.01;
    public Func<Matrix<double>, Matrix<double>> ResultTransformFunc = matrix => matrix;

    public Matrix<double> Weights { get; private set; }
    public Matrix<double> Bias { get; private set; }

    public Dictionary<int, double> SquaredErrorHistory { get; } = new();

    public Perceptron(int numberOfNeurons, int numberOfInputs)
    {
        NumberOfNeurons = numberOfNeurons;
        NumberOfInputs = numberOfInputs;

        var mBuilder = Matrix<double>.Build;
        Weights = mBuilder.Random(NumberOfNeurons, NumberOfInputs);
        Bias = mBuilder.Random(NumberOfNeurons, 1);
    }

    public Matrix<double> CalcResult(Sample sample)
    {
        var sampleX = sample.X.ToColumnMatrix();

        var result = (Weights * sampleX) + Bias;
        var transformedResult = ResultTransformFunc(result);
        return transformedResult;
    }

    public class NNResultStatistics
    {
        public Matrix<double> Result { get; set; } = null!;

        public Matrix<double> Error { get; set; } = null!;

        public bool CorrectResult { get; set; }

        public double SquaredError { get; set; }
    }

    public NNResultStatistics CalcResultStatistics(Sample sample)
    {
        var calcResult = CalcResult(sample);
        var expectedResult = sample.ExpectedResult.ToColumnMatrix();

        var resultError = expectedResult - calcResult;

        var expectedResultMaxIndex = expectedResult.EnumerateIndexed().MaxBy(x => x.Item3);
        var resultMaxIndex = calcResult.EnumerateIndexed().MaxBy(x => x.Item3);
        


        var statistics = new NNResultStatistics()
        {
            Result = calcResult,
            Error = resultError,
            CorrectResult = resultMaxIndex.Item1 == expectedResultMaxIndex.Item1, 
            SquaredError = resultError.PointwisePower(2).ColumnSums().First()
    };
        
        return statistics;
    }

    /// <summary>
    /// Trains the neural network using a input sample.
    /// </summary>
    /// <param name="sample">Sample to use in training</param>
    /// <param name="learningRateOverride"></param>
    /// <returns>The error of the training</returns>
    public NNResultStatistics Train(Sample sample, double? learningRateOverride = null)
    {
        var sampleX = sample.X.ToRowMatrix();

        var statistics = CalcResultStatistics(sample);
        var resultError = statistics.Error;

       var learningRate = learningRateOverride ?? LearningRate;

        Weights += learningRate * resultError * sampleX;
        Bias += learningRate * resultError;

        return statistics;
    }

    public void Train(IEnumerable<Sample> samples, int? iterations = null)
    {
        iterations ??= DefaultIterationCount;

        var samplesList = samples.AsList() ?? new List<Sample>();
        var squaredErrorSum = 1.0;

        for(var i = 0; i < iterations && squaredErrorSum > 0; i++)
        {
            squaredErrorSum = 0.0;
            foreach(var sample in samplesList)
            {
                var error = Train(sample, LearningRate / ((i + 1) * 0.1));
                //var error = Train(sample, LearningRate);
                squaredErrorSum += error.SquaredError;
            }

            SquaredErrorHistory[i] = squaredErrorSum;
        }
    }
}
