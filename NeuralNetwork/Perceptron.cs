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
    public double LearningRate { get; set; } = 0.1;
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

    /// <summary>
    /// Trains the neural network using a input sample.
    /// </summary>
    /// <param name="sample">Sample to use in training</param>
    /// <param name="learningRateOverride"></param>
    /// <returns>The error of the training</returns>
    public Matrix<double> Train(Sample sample, double? learningRateOverride = null)
    {
        var calcResult = CalcResult(sample);
        var sampleX = sample.X.ToRowMatrix();
        var expectedResult = sample.ExpectedResult.ToColumnMatrix();

        var learningRate = learningRateOverride ?? LearningRate;

        var resultError = expectedResult - calcResult;

        Weights += learningRate * resultError * sampleX;
        Bias += learningRate * resultError;

        return resultError;
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
                var squaredError = error.PointwisePower(2).ColumnSums().First();
                squaredErrorSum += squaredError;
            }

            SquaredErrorHistory[i] = squaredErrorSum;
        }
    }
}
